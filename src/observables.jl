# Observables: expectation values ⟨ψ|O|ψ⟩ / ⟨ψ|ψ⟩ of a saved wave function ψ = Σ_i c_i φ_i (`load_wavefunction`).
#
# An operator O is handled like H: its action on a basis key, Oφ_j = Σ (monomial, exponent set, coefficient) (`apply_op`), is
# looked up in the same element table as the assembly (rows = basis, column shifts = expansion terms). No matrix is formed:
#   g_u = Σ_i c_i ⟨φ_i | shift u⟩,   ⟨ψ|O|ψ⟩ = Σ_j c_j Σ_{(u, a) ∈ Oφ_j} a g_u
# (the cost of one table). Multi-component L ≥ 1 states keep the same "key → expansion" form.

"Operator for expectation values; `apply_op(op, hd, es, key)` expands its action on a basis key."
# keys of wave functions saved without components are 6-tuples (L = 0)
_key7(t::NTuple{7,Int}) = t
_key7(t::NTuple{6,Int}) = (t..., 0)
abstract type Observable end

"""
    DistancePower(p13, p23, p12)

Distance powers r₁₃^p13 r₂₃^p23 r₁₂^p12 (input numbering, in the order of `PowerTerm`: 1–3, 2–3, 1–2). Powers ≥ −1
(r^{−2} is finite in three dimensions, but the element integrals do not take negative-power keys).
"""
struct DistancePower <: Observable
    pow::NTuple{3,Int}
    function DistancePower(p13::Integer, p23::Integer, p12::Integer)
        all(>=(-1), (p13, p23, p12)) || throw(ArgumentError("distance powers must be ≥ −1 ($p13, $p23, $p12)"))
        new((Int(p13), Int(p23), Int(p12)))
    end
end
"Total energy H (the operator of `run_fc`; ⟨H⟩ reproduces the saved E)."
struct TotalEnergy <: Observable end
"Kinetic energy (including mass polarization)."
struct KineticEnergy <: Observable end
"Potential V (sum of the system terms)."
struct PotentialEnergy <: Observable end

"""
    Delta(i, j)

Coalescence density ⟨δ³(r_i − r_j)⟩ of two particles (input numbering), from the closed-form one-dimensional integral of ψ
restricted to the coalescence line. Works for ln(r₁+r₂) and split log variables.
"""
struct Delta <: Observable
    pair::NTuple{2,Int}
    Delta(i::Integer, j::Integer) = (i != j && i in 1:3 && j in 1:3) ? new(minmax(Int(i), Int(j))) :
                                    throw(ArgumentError("particle pair ($i, $j)"))
end
"""
    Cusp(i, j)

Cusp ν = ⟨δ³(r_ij) ∂/∂r_ij⟩ / ⟨δ³(r_ij)⟩ at coalescence; Kato's value q_i q_j μ_ij for the exact wave function (μ_ij the
reduced mass). Also for L ≥ 1 and the r₁₂ Gaussian. Meaningless (0/0) for pairs with ⟨δ⟩ = 0 (e.g. triplet electron pairs).
"""
struct Cusp <: Observable
    pair::NTuple{2,Int}
    Cusp(i::Integer, j::Integer) = (i != j && i in 1:3 && j in 1:3) ? new(minmax(Int(i), Int(j))) :
                                   throw(ArgumentError("particle pair ($i, $j)"))
end

describe(io::IO, o::Delta) = print(io, "⟨δ(r", o.pair[1], o.pair[2], ")⟩")
describe(io::IO, o::Cusp) = print(io, "ν", o.pair[1], o.pair[2])
describe(io::IO, o::DistancePower) = print(io, "⟨r₁₃^", o.pow[1], " r₂₃^", o.pow[2], " r₁₂^", o.pow[3], "⟩")
describe(io::IO, ::TotalEnergy) = print(io, "⟨H⟩")
describe(io::IO, ::KineticEnergy) = print(io, "⟨T⟩")
describe(io::IO, ::PotentialEnergy) = print(io, "⟨V⟩")

# Expansion of the action, as Vector{Tuple{monomial, exponent set index, component, coefficient}} (same form as apply_H).
# The pow of DistancePower is converted to internal numbering before (by `expectation`).
_mono_shift(m::Key, p, ::Val{false}) = (m[1] + p[1], m[2] + p[2], m[3] + p[3], m[4], m[5])
_mono_shift(m::Key, p, ::Val{true}) = (m[1] + p[1], m[2] + p[2], m[3] + p[3], m[4], 0, m[5])
function apply_op(o::DistancePower, hd::HamiltonianData{T}, es::ExponentSets, key::Key) where {T}
    K = hd.split ? Mono6 : Mono
    Tuple{K,Int,Int,T}[(_mono_shift(key, o.pow, Val(hd.split)), key[6], key[7], one(T))]
end
apply_op(::TotalEnergy, hd::HamiltonianData, es::ExponentSets, key::Key) = apply_H(hd, es, key)
apply_op(::PotentialEnergy, hd::HamiltonianData, es::ExponentSets, key::Key) = apply_H(hd, es, key; potential_only=true)
function apply_op(::KineticEnergy, hd::HamiltonianData{T}, es::ExponentSets, key::Key) where {T}
    h0 = HamiltonianData{T}(hd.invμ1, hd.invμ2, hd.invm3, Tuple{NTuple{3,Int},T}[], Tuple{NTuple{3,Int},NTuple{3,Float64},T}[],
                            hd.βσ, hd.split, hd.L, hd.unnatural, hd.w2)   # keep unnatural (else the neighbor component a+1 is generated) and w₂
    apply_H(h0, es, key)
end
_internal(o::Observable, π) = o
_internal(o::DistancePower, π) = (dm = _distance_map(π); p = ntuple(d -> o.pow[dm[d]], 3); DistancePower(p...))

# --- Coalescence (δ and cusp) -----------------------------------------------------------
# On the line where internal distance d is 0, the other two distances are equal (call them r), and σ = r₁ + r₂ + βσ r₁₂ is
#   d=1 (r₁=0): σ = (1+βσ) r, d=2 (r₂=0): σ = (1+βσ) r, d=3 (r₁₂=0): σ = 2r.
# A basis function becomes r^p σ^D (ln σ)^j e^{−κr} (0 if the power of the d coordinate is positive). Expanding (ln λr)^j in
# (ln r)^q gives a one-dimensional expansion (p, q, κ) → coefficient, integrated with
# G(N, J, κ) = ∫₀^∞ r^N (ln r)^J e^{−κr} dr = d^J/ds^J [Γ(s) κ^{−s}] at s = N+1.

# key of a one-dimensional term (p, q, κ, η): r^p (ln r)^q e^{−κr − ηr²} (η when the r₁₂ Gaussian remains on the coalescence line)
const _CKey = Tuple{Int,Int,BigFloat,BigFloat}

"""
Expand the monomial m = (A,B,C,D,R,j) (coefficient a) of exponent set k on the coalescence line of internal distance d into
one-dimensional terms added to `acc`. The function is r₁^A r₂^B r₁₂^C X^D σ^{−R} (ln σ)^j (ln(r₁+r₂): X = σ, R = 0; split:
X = s = r₁+r₂). On the line: d = 1, 2: s = r, σ = (1+βσ) r; d = 3: s = σ = 2r.
"""
function _restrict!(acc::Dict{_CKey,BigFloat}, m::NTuple{6,Int}, k::Int, a::BigFloat, es::ExponentSets, d::Int,
                    βσ::Float64, split::Bool)
    A, B, C, D, R, j = m
    pw = (A, B, C)
    pw[d] > 0 && return acc
    pw[d] < 0 && throw(ArgumentError("term diverges at coalescence (power $(pw[d]) of coordinate $d)"))
    p = sum(pw) - pw[d] + D - R
    # sum of the remaining two exponents exactly in BigFloat (Float64 sums of non-binary exponents round, and cancellations of
    # coefficients ~1e20 amplify it)
    ex = (BigFloat(es.α[k]), BigFloat(es.β[k]), BigFloat(es.γ[k]))
    κ = sum(ex) - ex[d]
    λσ = d == 3 ? BigFloat(2) : 1 + BigFloat(βσ)                           # σ = λσ r
    λX = split ? (d == 3 ? BigFloat(2) : BigFloat(1)) : λσ                 # X = λX r
    lλ = log(λσ)
    f = (D >= 0 ? λX^D : inv(λX^(-D))) / λσ^R
    η = d == 3 ? BigFloat(0) : BigFloat(es.η[k])                            # r₁₂ = r remains for d = 1, 2
    for q in 0:j
        key = (p, q, κ, η)
        acc[key] = get(acc, key, BigFloat(0)) + a * f * binomial(j, q) * lλ^(j - q)
    end
    acc
end

"G(N, J, κ) = ∫₀^∞ r^N (ln r)^J e^{−κr} dr (integer N ≥ 0): the J-th derivative of F(s) = exp(g(s)), g = lnΓ(s) − s ln κ, by recurrence."
function _gint(N::Int, J::Int, κ::BigFloat, ζ::Vector{BigFloat})
    N >= 0 || throw(ArgumentError("∫ r^$N … diverges"))
    m = N + 1
    F0 = factorial(big(N)) / κ^m
    J == 0 && return F0
    # g' = ψ(m) − ln κ, g^{(k+1)} = ψ^{(k)}(m) = (−1)^{k+1} k! (ζ(k+1) − Σ_{i<m} i^{−(k+1)})
    γE = BigFloat(Base.MathConstants.eulergamma)
    gd = Vector{BigFloat}(undef, J)
    gd[1] = -γE + sum((inv(BigFloat(i)) for i in 1:m-1); init=BigFloat(0)) - log(κ)
    for k in 1:J-1
        gd[k+1] = (-1)^(k + 1) * factorial(big(k)) * (ζ[k+1] - sum((inv(BigFloat(i))^(k + 1) for i in 1:m-1); init=BigFloat(0)))
    end
    F = Vector{BigFloat}(undef, J + 1); F[1] = F0           # F[r+1] = F^{(r)}
    for r in 1:J
        F[r+1] = sum(binomial(r - 1, k) * gd[k+1] * F[r-k] for k in 0:r-1)
    end
    F[J+1]
end

"""
G(N, J, κ, η) = ∫₀^∞ r^N (ln r)^J e^{−κr − ηr²} dr for terms where the r₁₂ Gaussian remains, for all N ≤ Nmax, J ≤ Jmax per (κ, η),
by the half-line DE rule (r = s·exp(t − e^{−t})), halving the step until all entries change by less than 2^{−(bits − 16)}. κ may be negative (η > 0).
"""
function _gauss_moments(κ::BigFloat, η::BigFloat, Nmax::Int, Jmax::Int)
    s = max(-κ / (2η), inv(sqrt(η)), BigFloat(1) / 64)      # peak position or width
    tol = BigFloat(2)^(-(precision(BigFloat) - 16))
    function sweep(h)
        G = zeros(BigFloat, Nmax + 1, Jmax + 1)
        t = -BigFloat(7)
        while t <= 6
            e = exp(-t); x = s * exp(t - e); w = h * s * (1 + e) * exp(t - e) * exp(-(κ + η * x) * x)
            if w != 0
                lx = log(x); xn = w
                for N in 0:Nmax
                    lj = xn
                    for J in 0:Jmax; G[N+1, J+1] += lj; lj *= lx; end
                    xn *= x
                end
            end
            t += h
        end
        G
    end
    h = BigFloat(1) / 16; G = sweep(h)
    while true
        h /= 2; G2 = sweep(h)
        maximum(abs(G2[i] - G[i]) / max(abs(G2[i]), floatmin(Float64)) for i in eachindex(G)) < tol && return G2
        h < BigFloat(2)^-14 && error("G(N,J,κ,η) does not converge (κ=$κ, η=$η)")
        G = G2
    end
end

"One-dimensional ⟨a|b⟩: Σ a_x b_y G(2 + p_x + p_y + shift, q_x + q_y, κ_x + κ_y, η_x + η_y) (r² is the 3D volume element at coalescence; shift is the power of the angular weight)."
function _pair_sum(a::Dict{_CKey,BigFloat}, b::Dict{_CKey,BigFloat}; shift::Int=0)
    Jmax = maximum(k[2] for k in keys(a); init=0) + maximum(k[2] for k in keys(b); init=0)
    ζ = [k >= 2 ? (z = BigFloat(); ccall((:mpfr_zeta_ui, :libmpfr), Cint, (Ref{BigFloat}, Culong, Int32), z, k, 0); z) : BigFloat(0)
         for k in 1:max(Jmax, 1)]
    memo = Dict{Tuple{Int,Int,BigFloat},BigFloat}()
    gm = Dict{Tuple{BigFloat,BigFloat},Matrix{BigFloat}}()
    Nmax = 2 + shift + maximum(k[1] for k in keys(a); init=0) + maximum(k[1] for k in keys(b); init=0)
    acc = BigFloat(0)
    for ((p1, q1, k1, η1), x) in a, ((p2, q2, k2, η2), y) in b
        N = 2 + p1 + p2 + shift; J = q1 + q2; κ = k1 + k2; η = η1 + η2
        G = if iszero(η)
            get!(() -> _gint(N, J, κ, ζ), memo, (N, J, κ))
        else
            N >= 0 || throw(ArgumentError("∫ r^$N … diverges"))
            get!(() -> _gauss_moments(κ, η, Nmax, Jmax), gm, (κ, η))[N+1, J+1]
        end
        acc += x * y * G
    end
    acc
end

"One-dimensional expansions of ψ and ∂ψ/∂r_d on the coalescence line of internal distance d."
function _coalescence(basis, es::ExponentSets, c, d::Int, βσ::Float64, split::Bool; derivative::Bool=false, ncomp::Int=1, comps::Bool=ncomp > 1, w2::Float64=1.0)
    isone(w2) || throw(ArgumentError("coalescence quantities (Delta, Cusp) require w₂ = 1"))
    ψs = [Dict{_CKey,BigFloat}() for _ in 1:ncomp]          # per component a (one for L = 0)
    dψs = [Dict{_CKey,BigFloat}() for _ in 1:ncomp]
    ψ = ψs[1]; dψ = dψs[1]
    bσ = BigFloat(βσ)
    for (i, key) in enumerate(basis)
        A, B, C, D, j, k, ac = key
        ci = BigFloat(c[i])
        _restrict!(ψs[ac+1], (A, B, C, D, 0, j), k, ci, es, d, βσ, split)
        derivative || continue
        dψ = dψs[ac+1]
        η = BigFloat(es.η[k])
        if split
            f6 = Poly6{BigFloat}((A, B, C, D, 0, j) => ci)
            g6 = d == 1 ? _d1(f6, BigFloat(es.α[k])) : d == 2 ? _d2(f6, BigFloat(es.β[k])) : _d12(f6, BigFloat(es.γ[k]), bσ, η)
            for (m, a) in g6
                m[d] < 0 && continue       # r^{−1}·(A times a term with A=0) vanishes (the coefficient carries A)
                _restrict!(dψ, m, k, a, es, d, βσ, split)
            end
        else
            f = Poly{BigFloat}((A, B, C, D, j) => ci)
            g = d == 1 ? _d1(f, BigFloat(es.α[k])) : d == 2 ? _d2(f, BigFloat(es.β[k])) : _d12(f, BigFloat(es.γ[k]), bσ, η)
            for (m, a) in g
                m[d] < 0 && continue
                _restrict!(dψ, (m[1], m[2], m[3], m[4], 0, m[5]), k, a, es, d, βσ, split)
            end
        end
    end
    comps ? (ψs, dψs) : (ψs[1], dψs[1])
end

"""
    expectation(sys, wf, ops; T=BigFloat, logvar=LogVariable(), engine=nothing, ntasks) -> Vector{T}

Expectation values of the operators `ops` (`Observable`s) for the saved wave function `wf` (`load_wavefunction(outdir, n)`).
`sys` is the system passed to `run_fc` (input numbering); `DistancePower` uses the input numbering (converted with `wf.order`).
Pass the same `logvar` as the run (it is not stored in the wave function file). The working precision is `T` (for BigFloat set
`setprecision` at least to that of the run). Checks: `TotalEnergy()` reproduces `wf.E`; for pure Coulomb systems
`PotentialEnergy()` / `TotalEnergy()` = 2 (virial theorem, up to the non-optimality of the exponents).
"""
function expectation(sys::ThreeBody, wf, ops::AbstractVector{<:Observable}; T::Type=BigFloat, logvar::LogVariable=LogVariable(),
                     engine=nothing, ntasks::Int=Threads.nthreads(), _norm::Bool=false)
    π = hasproperty(wf, :order) ? wf.order : (1, 2, 3)
    sysi = π == (1, 2, 3) ? sys : permute(sys, π)
    es = ExponentSets(copy(wf.α), copy(wf.β), copy(wf.γ); η=copy(get(wf, :η, zeros(length(wf.α)))))
    basis = _key7.(wf.basis); M = length(basis)
    L = get(wf, :L, 0); un = get(wf, :unnatural, false)
    length(wf.c) == M || throw(ArgumentError("$(length(wf.c)) coefficients for $M basis functions"))
    hd = HamiltonianData(sysi, logvar, T; L=L, unnatural=un)
    direct = L > 0 || _gaussian(es)      # elements from the direct table (angular weights, Gaussian integrals; _element_table_direct)
    close_exponent_sets!(hd, es, basis)
    ctx = IntegralContext(T, engine === nothing ? KernelQuadrature(T) : engine, logvar)
    c = T.(wf.c)
    K = hd.split ? Mono6 : Mono
    out = T[]
    Snorm = Ref{Any}(nothing)
    for op in ops
        if op isa Delta || op isa Cusp
            # Cusp for L ≥ 1 and Gaussians: radial derivatives of the angular part vanish in the spherical average around the
            # coalescence point, so the same W_ac sums as δ: numerator Σ W_ac ∫ r² f_a ∂f_c, denominator Σ W_ac ∫ r² f_a f_c
            dm = _distance_map(π)
            din = findfirst(==(op.pair), _PAIR_PARTICLES)
            d = findfirst(==(din), dm)                         # input pair → internal distance
            ncomp = L == 0 ? 1 : _maxcomp(hd) + 1
            ψ, dψ = _coalescence(basis, es, wf.c, d, logvar.βσ, logvar.split; derivative=op isa Cusp, ncomp=ncomp, comps=L > 0, w2=logvar.w2)
            if op isa Delta
                Snorm[] === nothing && (Snorm[] = expectation(sys, wf, [DistancePower(0, 0, 0)]; T=T, logvar=logvar, engine=engine,
                                                              ntasks=ntasks, _norm=true)[1])
                # ⟨δ⟩ = 4π ∫ r² ψ² dr / (8π² · Σ c c S) (the element integrals drop 8π²). For L ≥ 1, averaged over M: numerator
                # (4π/(2L+1)) Σ_ac ∫ r² f_a f_c W_ac|coalescence, denominator 8π²/(2L+1) S' (S' with W), the same form as L = 0.
                # Only W_ac terms with power 0 in the coalescence coordinate survive, as r^{sum of powers} (all vanish for unnatural parity)
                num = if ncomp == 1 && L == 0
                    _pair_sum(ψ, ψ)
                else
                    acc = BigFloat(0)
                    for a in 0:ncomp-1, cc in 0:ncomp-1
                        (isempty(ψ[a+1]) || isempty(ψ[cc+1])) && continue
                        for (pw, w) in Angular.angular_weight(L, a, cc; unnatural=un)
                            pw[d] == 0 || continue
                            acc += BigFloat(w) * _pair_sum(ψ[a+1], ψ[cc+1]; shift=sum(pw))
                        end
                    end
                    acc
                end
                push!(out, T(num / (2 * BigFloat(Base.MathConstants.pi) * BigFloat(Snorm[]))))
            elseif ncomp == 1 && L == 0
                push!(out, T(_pair_sum(ψ, dψ) / _pair_sum(ψ, ψ)))
            else
                nu = BigFloat(0); de = BigFloat(0)
                for a in 0:ncomp-1, cc in 0:ncomp-1
                    (isempty(ψ[a+1]) || (isempty(ψ[cc+1]) && isempty(dψ[cc+1]))) && continue
                    for (pw, w) in Angular.angular_weight(L, a, cc; unnatural=un)
                        pw[d] == 0 || continue
                        isempty(dψ[cc+1]) || (nu += BigFloat(w) * _pair_sum(ψ[a+1], dψ[cc+1]; shift=sum(pw)))
                        isempty(ψ[cc+1]) || (de += BigFloat(w) * _pair_sum(ψ[a+1], ψ[cc+1]; shift=sum(pw)))
                    end
                end
                iszero(de) && throw(ArgumentError("Cusp: the coalescence density vanishes identically (e.g. unnatural parity)"))
                push!(out, T(nu / de))
            end
            continue
        end
        oi = _internal(op, π)
        Oφ = Vector{Vector{Tuple{K,Int,Int,T}}}(undef, M)
        Threads.@threads for j in 1:M
            Oφ[j] = apply_op(oi, hd, es, basis[j])
        end
        if direct
            tab = _element_table_direct(Oφ, es, basis, 1:M, ntasks, L, ctx; unnatural=un)
        else
            tab = _element_table(Oφ, es, basis, 1:M, ntasks)
            _compute_elements!(tab, ctx, ntasks, 5_000_000; prefetch=!ctx.split)
        end
        nU = size(tab.idx, 1)
        # g_u = Σ_i c_i ⟨φ_i|u⟩ (per shift u; partial sums per thread)
        parts = [zeros(T, nU) for _ in 1:ntasks]
        @sync for (t, rr) in enumerate(Iterators.partition(1:M, cld(M, ntasks)))
            Threads.@spawn begin
                local gp = parts[t]            # different name from the outer variable (shared through the closure otherwise)
                @inbounds for i in rr, u in 1:nU
                    gp[u] += c[i] * tab.vals[tab.idx[u, i]]
                end
            end
        end
        gsum = reduce(+, parts)
        num = sum(c[j] * sum((a * gsum[u] for (u, a) in tab.hterms[j]); init=zero(T)) for j in 1:M)
        den = sum(c[j] * gsum[tab.sofs[j]] for j in 1:M)
        push!(out, _norm ? den : num / den)
    end
    out
end
expectation(sys::ThreeBody, wf, op::Observable; kwargs...) = expectation(sys, wf, [op]; kwargs...)[1]

# --- Variance σ² = ⟨(H−E)ψ|(H−E)ψ⟩/⟨ψ|ψ⟩ (direct 3D quadrature) -----------------------------------------------
# Element integrals of ⟨Hψ|Hψ⟩ need negative powers of distances (r₁^{−2} etc.) that the integral layer cannot form. σ² needs
# only ~1e-6 relative accuracy, so (H−E)ψ is evaluated pointwise from the expansion and integrated in perimetric coordinates
# (u,v,w) (r₁ = v+w, r₂ = u+w, r₁₂ = u+v). Near coalescence lines |R|² r₁r₂r₁₂ has corner singularities of type 1/(v+w), so the
# domain is split by the largest coordinate (3 parts) × the order of the other two (2 triangles); with the largest = s and the
# others s·ρ, s·ρ·ξ (ρ, ξ ∈ [0,1]), du dv dw = s²ρ ds dρ dξ removes the singularity. s uses the half-line DE rule, ρ, ξ tanh-sinh.

_vterm(m::Mono, k) = (m[1], m[2], m[3], m[4], 0, m[5], k)          # (A,B,C, d, R, j, k): σ^d (ln σ)^j for ln(r₁+r₂)
_vterm(m::Mono6, k) = (m[1], m[2], m[3], m[4], m[5], m[6], k)      # split: s^d σ^{−R} (ln σ)^j

function _powtab(x::T, lo::Int, hi::Int) where {T}
    v = Vector{T}(undef, hi - lo + 1); o = 1 - lo
    v[o] = one(T)
    for a in 1:hi; v[a+o] = v[a-1+o] * x; end
    ix = inv(x); for a in -1:-1:lo; v[a+o] = v[a+1+o] * ix; end
    v
end

# Pointwise evaluation of ψ and (H−E)ψ from the expansion (used by variance and distribution). Terms are (A,B,C, D, R, j, k, coefficient):
# r₁^A r₂^B r₁₂^C X^D σ^{−R} (ln σ)^j e^{−α_k r₁ − β_k r₂ − γ_k r₁₂} (X = σ for ln(r₁+r₂), s = r₁+r₂ for split).
function _point_terms(sys::ThreeBody, wf, logvar::LogVariable, ::Type{T}; residual::Bool) where {T}
    π = hasproperty(wf, :order) ? wf.order : (1, 2, 3)
    sysi = π == (1, 2, 3) ? sys : permute(sys, π)
    es = ExponentSets(copy(wf.α), copy(wf.β), copy(wf.γ))
    hd = HamiltonianData(sysi, logvar, BigFloat)
    get(wf, :L, 0) == 0 || throw(ArgumentError("L = 0 only"))
    any(!iszero, get(wf, :η, [0.0])) && throw(ArgumentError("the r₁₂ Gaussian (η > 0) is not supported here"))
    close_exponent_sets!(hd, es, _key7.(wf.basis))
    E = BigFloat(wf.E)
    Pd = Dict{NTuple{7,Int},BigFloat}(); Rd = Dict{NTuple{7,Int},BigFloat}()
    for (i, key) in enumerate(_key7.(wf.basis))
        key[7] == 0 || throw(ArgumentError("L = 0 only"))
        ci = BigFloat(wf.c[i])
        tk = logvar.split ? _vterm((key[1], key[2], key[3], key[4], 0, key[5]), key[6]) : _vterm(key[1:5], key[6])
        Pd[tk] = get(Pd, tk, BigFloat(0)) + ci
        residual || continue
        Rd[tk] = get(Rd, tk, BigFloat(0)) - E * ci
        for (m, kk, _, a) in apply_H(hd, es, key)
            t = _vterm(m, kk); Rd[t] = get(Rd, t, BigFloat(0)) + a * ci
        end
    end
    tP = [(t..., T(c)) for (t, c) in Pd if !iszero(c)]; tR = [(t..., T(c)) for (t, c) in Rd if !iszero(c)]
    all_ = vcat(tP, tR)
    # power tables start from the 0th power, so the range includes 0
    lo = ntuple(i -> min(0, minimum(t -> t[i], all_)), 6); hi = ntuple(i -> max(0, maximum(t -> t[i], all_)), 6)
    ρmin = minimum(min(es.β[k] + es.γ[k], es.α[k] + es.γ[k], es.α[k] + es.β[k]) for k in eachindex(es.α))
    (tP=tP, tR=tR, lo=lo, hi=hi, o=ntuple(i -> 1 - lo[i], 6), α=T.(es.α), β=T.(es.β), γ=T.(es.γ), βσ=T(logvar.βσ),
     split=logvar.split, E=E, ρmin=ρmin, w2=T(logvar.w2))
end

function _eval_terms(pt, ts, P1, P2, P12, PSd, PR, lσ, ek::Vector{T}) where {T}
    o = pt.o
    sacc = zero(T)
    @inbounds for (A, B, C, D, R, j, k, c) in ts
        x = c * P1[A+o[1]] * P2[B+o[2]] * P12[C+o[3]] * PSd[D+o[4]] * PR[R+o[5]] * ek[k]
        for _ in 1:j; x *= lσ; end
        sacc += x
    end
    sacc
end

"(ψ, (H−E)ψ) at the point (r₁, r₂, r₁₂); the second is 0 for terms with residual=false."
function _psi_at(pt, r1::T, r2::T, r12::T) where {T}
    lo, hi = pt.lo, pt.hi
    ss = r1 + r2; σ = (isone(pt.w2) ? ss : r1 + pt.w2 * r2) + pt.βσ * r12    # σ = r₁ + w₂ r₂ + βσ r₁₂
    P1 = _powtab(r1, lo[1], hi[1]); P2 = _powtab(r2, lo[2], hi[2]); P12 = _powtab(r12, lo[3], hi[3])
    PSd = _powtab(pt.split ? ss : σ, lo[4], hi[4]); PR = _powtab(inv(σ), lo[5], hi[5])
    ek = [exp(-(pt.α[k] * r1 + pt.β[k] * r2 + pt.γ[k] * r12)) for k in eachindex(pt.α)]
    lσ = log(σ)
    (_eval_terms(pt, pt.tP, P1, P2, P12, PSd, PR, lσ, ek), isempty(pt.tR) ? zero(T) : _eval_terms(pt, pt.tR, P1, P2, P12, PSd, PR, lσ, ek))
end

# nodes and weights of the half-line DE rule (x = L·exp(t − e^{−t}), t ∈ [−3, 4.5]) and of tanh-sinh on [0,1]
function _de_half(::Type{T}, N::Int, L) where {T}
    # lower end x ≈ 1e-10·L (smaller values overflow negative powers of σ to Inf·0 = NaN; the contribution there is ~x⁵)
    h = 7.5 / (N - 1)
    map(0:N-1) do i
        t = -3.0 + i * h; xx = exp(t - exp(-t))
        (T(L) * T(xx), T(L) * T(h) * T((1 + exp(-t)) * xx))
    end
end
function _tanhsinh01(::Type{T}, N::Int) where {T}
    h = 6.4 / (N - 1)
    map(0:N-1) do i
        b = BigFloat(-3.2 + i * h); q = BigFloat(Base.MathConstants.pi) / 2 * sinh(b)
        (T((1 + tanh(q)) / 2), T(BigFloat(h) * BigFloat(Base.MathConstants.pi) / 4 * cosh(b) / cosh(q)^2))
    end
end

"""
    variance(sys, wf; logvar=LogVariable(), N=50, T=nothing) -> (σ2, dE, E)

σ² = ⟨(H−E)ψ|(H−E)ψ⟩/⟨ψ|ψ⟩ (E is the saved value) with 6N³ points. `dE` = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ − E of the same quadrature (its
accuracy: σ² is good to δ relative if dE ≪ σ²·δ). Check convergence with two or more N. Temple's lower bound is `temple(E, σ2, E1)`.
L ≥ 1 and the r₁₂ Gaussian use the pointwise evaluation of `expectation_quad` ((H−E)Ψ^M in Cartesian coordinates including the
angular derivatives; power terms only); both paths agree for L = 0.
"""
function variance(sys::ThreeBody, wf; logvar::LogVariable=LogVariable(), N::Int=50, T::Union{Type,Nothing}=nothing, coords=:perimetric)
    T = _resolve_quad_type(T, wf; floor=Float64x4)
    if get(wf, :L, 0) > 0 || any(!iszero, get(wf, :η, [0.0])) || coords !== :perimetric
        # L ≥ 1 and the r₁₂ Gaussian: the pointwise evaluation of the quadrature operators (power terms only)
        v = expectation_quad(sys, wf, [_Residual(1), _Residual(2)]; logvar=logvar, N=N, T=T, coords=coords)
        return (σ2=v[2], dE=v[1], E=BigFloat(wf.E))
    end
    pt = _point_terms(sys, wf, logvar, T; residual=true)
    S = _de_half(T, N, 1 / pt.ρmin); Q = _tanhsinh01(T, N)
    acc = [zeros(T, 3) for _ in 1:N]
    Threads.@threads :dynamic for is in 1:N
        local ac = acc[is]
        local sv = S[is][1]
        local wsv = S[is][2]
        for (ρ, wρ) in Q, (ξ, wξ) in Q
            local wt = wsv * wρ * wξ * sv^2 * ρ
            for (yy, zz) in ((ρ, ρ * ξ), (ρ * ξ, ρ))       # two triangles × largest coordinate (u, v, w)
                for (u, v, w) in ((sv, sv * yy, sv * zz), (sv * yy, sv, sv * zz), (sv * yy, sv * zz, sv))
                    local r1 = v + w
                    local r2 = u + w
                    local r12 = u + v
                    local ψ, Rv = _psi_at(pt, r1, r2, r12)
                    local jac = r1 * r2 * r12 * wt
                    ac[1] += jac * ψ * ψ; ac[2] += jac * ψ * Rv; ac[3] += jac * Rv * Rv
                end
            end
        end
    end
    tot = reduce(+, acc)
    (σ2=BigFloat(tot[3] / tot[1]), dE=BigFloat(tot[2] / tot[1]), E=pt.E)
end

"""
    distribution(sys, wf, (i, j), Rs; logvar=LogVariable(), N=60, T=nothing) -> Vector

Distribution P(R) of the pair distance r_ij (∫₀^∞ P(R) dR = 1, input numbering) at the points `Rs`; ⟨δ³(r_ij)⟩ = lim P(R)/(4πR²).
Two-dimensional quadrature (N² points) over the sum a ∈ [R, ∞) (half-line DE) and difference b ∈ [−R, R] (tanh-sinh) of the
other two distances; the normalization ⟨ψ|ψ⟩ is exact from the element integrals.
"""
function distribution(sys::ThreeBody, wf, pair::NTuple{2,Int}, Rs::AbstractVector; logvar::LogVariable=LogVariable(), N::Int=60,
                      T::Union{Type,Nothing}=nothing)
    T = _resolve_quad_type(T, wf; floor=Float64x4)
    π = hasproperty(wf, :order) ? wf.order : (1, 2, 3)
    din = findfirst(==(minmax(pair...)), _PAIR_PARTICLES)
    din === nothing && throw(ArgumentError("particle pair $pair"))
    d = findfirst(==(din), _distance_map(π))
    # L ≥ 1 and the Gaussian: pointwise evaluation of the quadrature operators (Σ_M w_M|Ψ^M|² = Σ W_ac f_a f_c; the closed-form denominator also has W)
    general = get(wf, :L, 0) > 0 || any(!iszero, get(wf, :η, [0.0]))
    pt = general ? _point_expansions(sys, wf, logvar, T) : _point_terms(sys, wf, logvar, T; residual=false)
    den = expectation(sys, wf, [DistancePower(0, 0, 0)]; T=BigFloat, logvar=logvar, _norm=true)[1]
    A = _de_half(T, N, 1 / pt.ρmin); Q = _tanhsinh01(T, N)
    out = Vector{BigFloat}(undef, length(Rs))
    Threads.@threads :dynamic for iR in eachindex(Rs)
        local R = T(Rs[iR])
        local acc = zero(T)
        local wk = general ? _quadwork(pt, T) : nothing
        for (x, wx) in A, (q, wq) in Q
            local a = R + x                      # sum of the other two distances (≥ R)
            local b = R * (2q - 1)               # difference (|b| ≤ R)
            local p1 = (a + b) / 2
            local p2 = (a - b) / 2
            local wt = wx * wq * 2R / 2          # db = 2R dq, dp₁ dp₂ = da db / 2
            # internal distance d at R, the other two at (p₁, p₂)
            local r1, r2, r12 = d == 1 ? (R, p1, p2) : d == 2 ? (p1, R, p2) : (p1, p2, R)
            local ψψ = if general
                _expansions_at!(wk, pt, r1, r2, r12); _invariants(pt, wk.acc, r1, r2, r12).ψψ
            else
                ψ = _psi_at(pt, r1, r2, r12)[1]; ψ * ψ
            end
            acc += wt * r1 * r2 * r12 * ψψ
        end
        out[iR] = BigFloat(acc) / den
    end
    out
end

"""
    temple(E, σ2, E1)

Temple's lower bound E₀ ≥ E − σ²/(E₁ − E); E₁ is a lower bound of the second eigenvalue of the same symmetry (E < E₁ required).
"""
function temple(E, σ2, E1)
    E < E1 || throw(ArgumentError("E < E₁ required"))
    E - σ2 / (E1 - E)
end

# --- Operators by quadrature (parts of relativistic corrections) --------------------------------------------------
# Operators whose element integrals need negative distance powers (−1 including the volume element), by the pointwise evaluation and perimetric quadrature of σ².
# In internal coordinates r⃗ᵢ = R⃗ᵢ − R⃗₃ (i = 1, 2), the CM-frame momenta are Pᵢ = −i∇ᵢ (i = 1, 2), P₃ = −(P₁ + P₂) = i(∇₁ + ∇₂) (mass independent).

"""
    P4(i)

⟨p_i⁴⟩ = ⟨(∇_i²ψ)²⟩ (CM-frame momentum of input particle i; (∇₁+∇₂)²ψ for internal particle 3), a form that avoids the δ-type
singularity at coalescence (second form of Fang et al., arXiv:2301.05442 Eq. 23). Used with `expectation_quad`.
"""
struct P4 <: Observable
    i::Int
end
"⟨∇₁²ψ|∇₂²ψ⟩ (product of the Laplacians of the two light particles; part of the regularization Σp⁴ = 4⟨(E−V)²⟩ − 2⟨∇₁²∇₂²⟩)."
struct LaplacianProduct <: Observable end
"⟨(E − V)²⟩ (E is the saved value)."
struct EnergyMinusVSquared <: Observable end
"""
    Retardation()

Retardation (orbit–orbit) ⟨p₁ⁱ (δᵢⱼ/r₁₂ + r₁₂ⁱ r₁₂ʲ/r₁₂³) p₂ʲ⟩ (form of Puchalski–Komasa–Pachucki, arXiv:1704.07153 Eq. 13).
The Breit–Pauli H₂ expectation value is −½ × this (two electrons of mass 1).
"""
struct Retardation <: Observable
    pair::NTuple{2,Int}
    Retardation(i::Integer, j::Integer) = (i != j && i in 1:3 && j in 1:3) ? new(minmax(Int(i), Int(j))) :
                                          throw(ArgumentError("particle pair ($i, $j)"))
end
Retardation() = Retardation(1, 2)
"""
    QTerm(i, j)

Araki–Sucher Q term Q(r_ij) = lim_{ρ→0} ⟨Θ(r − ρ)/(4πr³) + (ln ρ + γ)δ(r)⟩ (1810.07929 Eq. 14; input numbering, any masses) in the
global form (μ_ij/2π)⟨[2(E−V)ψ² − G](ln r_ij + 1 + γ)/r_ij⟩ (G is the mass-weighted |∇ψ|² of `DrachmanDelta`), from
−(1/2μ)⟨∇²f⟩ = 2⟨(E−V)ψ² f⟩ − ⟨G f⟩ with f = ln r/r and 1/r. For He, ⟨1/r₁₂³⟩_AS = 4π Q(r₁₂).
"""
struct QTerm <: Observable
    pair::NTuple{2,Int}
    QTerm(i::Integer, j::Integer) = (p = (min(i, j), max(i, j)); p in _PAIR_PARTICLES ? new(p) : throw(ArgumentError("pair must be (1,3), (2,3) or (1,2)")))
end
describe(io::IO, o::QTerm) = print(io, "Q(r", o.pair[1], o.pair[2], ")")
"⟨1/r_ij²⟩ (input numbering)."
struct InverseSquare <: Observable
    pair::NTuple{2,Int}
    InverseSquare(i::Integer, j::Integer) = (i != j && i in 1:3 && j in 1:3) ? new(minmax(Int(i), Int(j))) :
                                            throw(ArgumentError("particle pair ($i, $j)"))
end
"""
    Uehling(i, j; inv_alpha="137.035999177")

Vacuum polarization (Uehling) potential of pair (i, j) (input numbering), ⟨V_U⟩ in a.u. including the factor α:
V_U(r) = (q_i q_j / r)(α/3π) ∫₁^∞ dt (2t² + 1)√(t² − 1)/t⁴ · e^{−2rt/α} (electron loop only; Compton wavelength α a.u.).
q_i q_j is the coefficient of the Coulomb term 1/r_ij (0 if absent). Evaluated pointwise with t = cosh u and the trapezoidal rule in u
(error ~e^{−π²/h}, analytic in |Im u| < π/2). Dominant for muon–nucleus pairs. Used with `expectation_quad` (the form of Eskin et al.,
EPJ WoC 222, 03009 (2019) Eq. 17).
"""
struct Uehling <: Observable
    pair::NTuple{2,Int}
    inv_alpha::String
    Uehling(i::Integer, j::Integer; inv_alpha::AbstractString="137.035999177") =
        (p = (min(i, j), max(i, j)); p in _PAIR_PARTICLES ? new(p, String(inv_alpha)) : throw(ArgumentError("pair must be (1,3), (2,3) or (1,2)")))
end
describe(io::IO, o::Uehling) = print(io, "⟨V_Uehling(r", o.pair[1], o.pair[2], ")⟩")
describe(io::IO, o::P4) = print(io, "⟨p", o.i, "⁴⟩")
describe(io::IO, ::LaplacianProduct) = print(io, "⟨∇₁²ψ|∇₂²ψ⟩")
describe(io::IO, ::EnergyMinusVSquared) = print(io, "⟨(E−V)²⟩")
describe(io::IO, o::Retardation) = print(io, "⟨p", o.pair[1], "(δ/r+rr/r³)p", o.pair[2], "⟩")
describe(io::IO, o::InverseSquare) = print(io, "⟨1/r", o.pair[1], o.pair[2], "²⟩")

"""
    ArakiSucherPart()

Quadrature part of the regularized Araki–Sucher ⟨1/r₁₂³⟩ (Fang et al. arXiv:2301.05442 Eq. 13):
−Σᵢ⟨∇ᵢψ|r₁₂⁻¹ ln r₁₂|∇ᵢψ⟩ + ⟨ψ|2(E−V) ln r₁₂/r₁₂|ψ⟩; the whole is `araki_sucher` (+4π(1+γ)⟨δ³(r₁₂)⟩).
"""
struct ArakiSucherPart <: Observable end
"""
    DrachmanDelta12()

Drachman's global form 4π⟨δ³(r₁₂)⟩ = 2⟨r₁₂⁻¹(E−V)⟩ − Σᵢ⟨∇ᵢψ|r₁₂⁻¹|∇ᵢψ⟩ (arXiv:2301.05442 Eq. 9), a check of the closed-form `Delta(1, 2)`.
"""
struct DrachmanDelta12 <: Observable end
"""
    DrachmanDelta(i, j)

⟨δ³(r_ij)⟩ in global form for three bodies of any masses (input numbering), from T(1/r_ij) = (2π/μ_ij) δ³(r_ij) and the Schrödinger
equation: ⟨δ(r_ij)⟩ = (μ_ij/2π) ⟨[2(E−V)ψ² − G]/r_ij⟩, G = |∇₁ψ|²/μ₁ + |∇₂ψ|²/μ₂ + 2∇₁ψ·∇₂ψ/m₃ (internal coordinates).
Its difference from the closed-form `Delta(i, j)` indicates the wave-function error (it converges faster than the local value).
"""
struct DrachmanDelta <: Observable
    pair::Tuple{Int,Int}
    DrachmanDelta(i::Integer, j::Integer) = (p = (min(i, j), max(i, j)); p in _PAIR_PARTICLES ? new(p) : throw(ArgumentError("pair must be (1,3), (2,3) or (1,2)")))
end
describe(io::IO, o::DrachmanDelta) = print(io, "⟨δ(r", o.pair[1], o.pair[2], ")⟩ (global form)")
describe(io::IO, ::ArakiSucherPart) = print(io, "quadrature part of ⟨1/r₁₂³⟩")
describe(io::IO, ::DrachmanDelta12) = print(io, "4π⟨δ(r₁₂)⟩ (Drachman)")

"""
    KineticQuad(form)

⟨T⟩ by quadrature (for checks). `form = :laplacian`: −½(1/μ₁)⟨ψ|∇₁²ψ⟩ − ½(1/μ₂)⟨ψ|∇₂²ψ⟩ − (1/m₃)⟨ψ|∇₁·∇₂ψ⟩; `:gradient`: the
integrated-by-parts ½(1/μ₁)⟨|∇₁ψ|²⟩ + ½(1/μ₂)⟨|∇₂ψ|²⟩ + (1/m₃)⟨∇₁ψ·∇₂ψ⟩. Should equal the closed-form `KineticEnergy()` (checks
the angular derivatives for L ≥ 1).
"""
struct KineticQuad <: Observable
    form::Symbol
end
describe(io::IO, o::KineticQuad) = print(io, "⟨T⟩ (quadrature, ", o.form, ")")
# ⟨ψ|(H−E)ψ⟩ (k = 1) and ⟨(H−E)ψ|(H−E)ψ⟩ (k = 2), for the L ≥ 1 and Gaussian path of `variance` (internal)
struct _Residual <: Observable
    k::Int
end

"""
    LDot((i, j), k; power=-3)

⟨L·(r_ij × p_k) r_ij^power⟩ (input numbering; r_ij = R_i − R_j, p_k the CM-frame momentum of particle k, L the total angular
momentum), a rotational invariant summed over M, used for spin–orbit terms by the projection ⟨V·S⟩_J = ⟨L·V⟩/(L(L+1))·⟨L·S⟩_J (`fine_structure`).
"""
struct LDot <: Observable
    pair::NTuple{2,Int}
    k::Int
    power::Int
    LDot(p::NTuple{2,<:Integer}, k::Integer; power::Integer=-3) =
        (q = (min(p...), max(p...)); q in _PAIR_PARTICLES && k in 1:3 ? new(q, Int(k), Int(power)) : throw(ArgumentError("pair $(p), particle $(k)")))
end
"""
    LTensor((i, j))

⟨Σ_ab L_a Q_ab L_b⟩, Q_ab = (δ_ab r² − 3 r_a r_b)/r⁵ (r = r_ij). Within an L multiplet Q_ab = κ(½{L_a,L_b} − δ_ab L²/3),
κ = ⟨LQL⟩/c_L, c_L = (2/3)(L(L+1))² − L(L+1)/2; used for spin–spin tensor terms (`fine_structure`).
"""
struct LTensor <: Observable
    pair::NTuple{2,Int}
    LTensor(p::NTuple{2,<:Integer}) = (q = (min(p...), max(p...)); q in _PAIR_PARTICLES ? new(q) : throw(ArgumentError("pair $p")))
end
describe(io::IO, o::LDot) = print(io, "⟨L·(r", o.pair[1], o.pair[2], "×p", o.k, ")r^", o.power, "⟩")
describe(io::IO, o::LTensor) = print(io, "⟨L Q(r", o.pair[1], o.pair[2], ") L⟩")

const QuadObservable = Union{P4,LaplacianProduct,EnergyMinusVSquared,Retardation,InverseSquare,ArakiSucherPart,DrachmanDelta12,DrachmanDelta,KineticQuad,_Residual,QTerm,LDot,LTensor,Uehling}

# Expansions of ψ, ∂ψ/∂r₁, ∂ψ/∂r₂, ∂ψ/∂r₁₂, ∇₁²ψ, ∇₂²ψ, ∇₁·∇₂ψ (internal numbering). For L ≥ 1 seven per component a
# (expansion indices 7a + 1..7), derivatives of f_a only (those of the angular part 𝒴_a are added pointwise, `_angular_fields`).
function _point_expansions(sys::ThreeBody, wf, logvar::LogVariable, ::Type{T}) where {T}
    isone(logvar.w2) || throw(ArgumentError("pointwise evaluation for L ≥ 1 and Gaussians requires w₂ = 1"))
    π = hasproperty(wf, :order) ? wf.order : (1, 2, 3)
    sysi = π == (1, 2, 3) ? sys : permute(sys, π)
    L = get(wf, :L, 0); unnat = get(wf, :unnatural, false)
    ncomp = L == 0 ? 1 : L - unnat + 1
    es = ExponentSets(copy(wf.α), copy(wf.β), copy(wf.γ); η=copy(get(wf, :η, zeros(length(wf.α)))))
    hd = HamiltonianData(sysi, logvar, BigFloat)
    close_exponent_sets!(hd, es, _key7.(wf.basis))
    D = [Dict{NTuple{7,Int},BigFloat}() for _ in 1:7ncomp]
    add!(dd, t, c) = (dd[t] = get(dd, t, BigFloat(0)) + c)
    bσ = BigFloat(logvar.βσ)
    for (i, key) in enumerate(_key7.(wf.basis))
        ci = BigFloat(wf.c[i]); k = key[6]; o7 = 7key[7]
        a, b, g, η = BigFloat(es.α[k]), BigFloat(es.β[k]), BigFloat(es.γ[k]), BigFloat(es.η[k])
        m = logvar.split ? (key[1], key[2], key[3], key[4], 0, key[5]) : key[1:5]
        f = Dict(m => ci)
        add!(D[o7+1], _vterm(m, k), ci)
        for (dd, w) in ((2, _d1(f, a)), (3, _d2(f, b)), (4, _d12(f, g, bσ, η)))
            for (mm, c) in w; add!(D[o7+dd], _vterm(mm, k), c); end
        end
        l1, l2, mp = kinetic_terms(m, a, b, g, bσ, η)
        for (mm, c) in l1; add!(D[o7+5], _vterm(mm, k), c * ci); end
        for (mm, c) in l2; add!(D[o7+6], _vterm(mm, k), c * ci); end
        for (mm, c) in mp; add!(D[o7+7], _vterm(mm, k), c * ci); end
    end
    ts = [[(t..., T(c)) for (t, c) in dd if !iszero(c)] for dd in D]
    allt = reduce(vcat, ts)
    lo = ntuple(i -> min(0, minimum(t -> t[i], allt)), 6); hi = ntuple(i -> max(0, maximum(t -> t[i], allt)), 6)   # include 0 (as above)
    # quadrature scale: smallest perimetric exponent (only α + β for Gaussian sets, where the two containing r₁₂ may be negative)
    ρmin = minimum(iszero(es.η[k]) ? min(es.β[k] + es.γ[k], es.α[k] + es.γ[k], es.α[k] + es.β[k]) : es.α[k] + es.β[k] for k in eachindex(es.α))
    V = [(p, T(exact(BigFloat, t.c))) for t in sysi.terms if t isa PowerTerm for p in (t.pow,)]
    any(t -> !(t isa PowerTerm), sysi.terms) && throw(ArgumentError("quadrature operators require systems of power terms (PowerTerm) only"))
    # one table for the monomials of the seven expansions: monomial = (r₁^A r₂^B r₁₂^C) × (X^D σ^{−R} (ln σ)^j e_k); each distinct factor
    # is computed once per point, then one product per monomial and one multiply-add per term (with its expansion index)
    U = Dict{NTuple{7,Int},Vector{Tuple{Int16,BigFloat}}}()
    for (i, dd) in enumerate(D), (t, c) in dd
        iszero(c) || push!(get!(() -> Tuple{Int16,BigFloat}[], U, t), (Int16(i), c))
    end
    o = ntuple(i -> 1 - lo[i], 6)
    f3 = Dict{NTuple{3,Int},Int32}(); f4 = Dict{NTuple{4,Int},Int32}()
    mono = Tuple{Int32,Int32}[]; tstart = Int32[1]; tidx = Int16[]; tcoef = T[]
    for (t, cs) in U
        i3 = get!(f3, (t[1] + o[1], t[2] + o[2], t[3] + o[3]), Int32(length(f3) + 1))
        i4 = get!(f4, (t[4] + o[4], t[5] + o[5], t[6] + 1, t[7]), Int32(length(f4) + 1))
        push!(mono, (i3, i4))
        for (e, c) in cs; push!(tidx, e); push!(tcoef, T(c)); end
        push!(tstart, Int32(length(tidx) + 1))
    end
    fac3 = Vector{NTuple{3,Int}}(undef, length(f3)); for (k, v) in f3; fac3[v] = k; end
    fac4 = Vector{NTuple{4,Int}}(undef, length(f4)); for (k, v) in f4; fac4[v] = k; end
    im = inverse_masses(sysi, BigFloat)
    (ts=ts, im=(T(im[1]), T(im[2]), T(im[3])), mono=mono, tstart=tstart, tidx=tidx, tcoef=tcoef, fac3=fac3, fac4=fac4, lo=lo, hi=hi, o=o, α=T.(es.α), β=T.(es.β), γ=T.(es.γ), η=T.(es.η), βσ=T(logvar.βσ),
     split=logvar.split, E=T(wf.E), V=V, ρmin=ρmin, π=π, L=L, ncomp=ncomp,
     ang=L == 0 ? nothing : _angular_tables(L, ncomp, unnat, T))
end

# frame monomials of T_a^M and their derivatives per component (Angular.component_fields) in the working type; ang[a+1][M] = (w_M, 10 fields)
function _angular_tables(L, ncomp, unnat, ::Type{T}) where {T}
    [[(T(w), [[(e, Complex{T}(T(real(c)), T(imag(c)))) for (e, c) in fld] for fld in flds])
      for (w, flds) in Angular.component_fields(L, a; unnatural=unnat)] for a in 0:ncomp-1]
end

# Bilinear invariants at the point (r₁, r₂, r₁₂) (summed over M, rotationally averaged). L = 0 directly from ψ and its radial
# derivatives; L ≥ 1 builds gradients and Laplacians of Ψ^M = Σ_a f_a T_a^M in Cartesian coordinates in the frame
# r₁ = (0,0,r₁), r₂ = (r₂ s, 0, r₂ c) and sums with weights w_M:
#   ψψ = Σ|Ψ|², n1 = Σ|∇₁Ψ|², n2 = Σ|∇₂Ψ|², x12 = Σ Re ∇₁Ψ*·∇₂Ψ, qq = Σ Re (r̂₁₂·∇₁Ψ)*(r̂₁₂·∇₂Ψ),
#   l11 = Σ|∇₁²Ψ|², l22 = Σ|∇₂²Ψ|², l12 = Σ Re (∇₁²Ψ)*∇₂²Ψ, l33 = Σ|(∇₁ + ∇₂)²Ψ|²,
#   pl1 = Σ Re Ψ*∇₁²Ψ, pl2 = Σ Re Ψ*∇₂²Ψ, pm = Σ Re Ψ*∇₁·∇₂Ψ (kinetic checks, `KineticQuad`),
#   ψR = Σ Re Ψ*(H−E)Ψ, RR = Σ|(H−E)Ψ|² (σ², `variance` for L ≥ 1 and Gaussians),
#   q13 = Σ Re (r̂₁·∇₁Ψ)*(r̂₁·∇₃Ψ), q23 = Σ Re (r̂₂·∇₂Ψ)*(r̂₂·∇₃Ψ) (∇₃ = −(∇₁ + ∇₂); `Retardation(i, j)`),
#   lc[3p + k] = Σ Re A*·(r_p × ∇_kΨ) (p = internal pairs (1,3),(2,3),(1,2) as 0,1,2; k = 0,1,2 for ∇₁,∇₂,∇₃),
#   lt[p] = Σ (|A|² r_p² − 3|r_p·A|²) (A = r₁×∇₁Ψ + r₂×∇₂Ψ, LΨ = −iA; `LDot`, `LTensor`, fine structure)
function _invariants(pt, acc, r1::T, r2::T, r12::T) where {T}
    c12 = (r1 * r1 + r2 * r2 - r12 * r12) / (2 * r1 * r2)
    i1, i2, i3 = pt.im; Vp = _potential_at(pt, r1, r2, r12)
    if pt.L == 0
        ψ, g1, g2, g12, L1, L2, MP = acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]
        c1q = (r1 * r1 - r2 * r2 + r12 * r12) / (2 * r1 * r12)
        c2q = (r1 * r1 - r2 * r2 - r12 * r12) / (2 * r2 * r12)
        n1 = g1 * g1 + g12 * g12 + 2 * g1 * g12 * c1q
        n2 = g2 * g2 + g12 * g12 - 2 * g2 * g12 * c2q
        x12 = g1 * g2 * c12 - g1 * g12 * c1q + g12 * g2 * c2q - g12 * g12
        L3 = L1 + L2 + 2MP
        R = -(i1 * L1 + i2 * L2) / 2 - i3 * MP + (Vp - pt.E) * ψ
        a11 = g1 + g12 * c1q; a12 = g2 * c12 - g12 * c1q          # r̂₁·∇₁ψ, r̂₁·∇₂ψ
        b21 = g1 * c12 + g12 * c2q; b22 = g2 - g12 * c2q          # r̂₂·∇₁ψ, r̂₂·∇₂ψ
        return (ψψ=ψ * ψ, n1=n1, n2=n2, x12=x12, qq=(g1 * c1q + g12) * (g2 * c2q - g12), l11=L1 * L1, l22=L2 * L2, l12=L1 * L2, l33=L3 * L3,
                pl1=ψ * L1, pl2=ψ * L2, pm=ψ * MP, ψR=ψ * R, RR=R * R, q13=-(a11 * a11 + a11 * a12), q23=-(b22 * b22 + b22 * b21),
                lc=ntuple(_ -> zero(T), 9), lt=ntuple(_ -> zero(T), 3))        # LΨ = 0 for L = 0
    end
    s = sqrt(max(zero(T), 1 - c12 * c12))
    z1 = r1; x2 = r2 * s; z2 = r2 * c12
    e1 = (zero(T), zero(T), one(T)); e2 = (s, zero(T), c12)
    e12 = (-x2 / r12, zero(T), (z1 - z2) / r12)                       # r̂₁₂ = (r⃗₁ − r⃗₂)/r₁₂
    dot3(u, v) = u[1] * v[1] + u[2] * v[2] + u[3] * v[3]
    z = zero(T)
    ψψ = n1 = n2 = x12 = qq = l11 = l22 = l12 = l33 = pl1 = pl2 = pm = ψR = RR = q13 = q23 = z
    lc = ntuple(_ -> z, 9); lt = ntuple(_ -> z, 3)
    rv = ((zero(T), zero(T), z1), (x2, zero(T), z2), (-x2, zero(T), z1 - z2))   # vectors R_a − R_b of the internal pairs (1,3), (2,3), (1,2)
    cross3(u, v) = (u[2] * v[3] - u[3] * v[2], u[3] * v[1] - u[1] * v[3], u[1] * v[2] - u[2] * v[1])
    nM = length(pt.ang[1])
    for iM in 1:nM
        w = pt.ang[1][iM][1]
        Ψ = zero(Complex{T}); G1 = (Ψ, Ψ, Ψ); G2 = (Ψ, Ψ, Ψ); D1 = Ψ; D2 = Ψ; DM = Ψ
        for a in 1:pt.ncomp
            o = 7(a - 1)
            f, g1, g2, g12, L1, L2, MP = acc[o+1], acc[o+2], acc[o+3], acc[o+4], acc[o+5], acc[o+6], acc[o+7]
            fl = pt.ang[a][iM][2]
            v = ntuple(q -> _frame_eval(fl[q], z1, x2, z2), 10)
            ∇1f = ntuple(i -> g1 * e1[i] + g12 * e12[i], 3); ∇2f = ntuple(i -> g2 * e2[i] - g12 * e12[i], 3)
            ∇1T = (v[2], v[3], v[4]); ∇2T = (v[5], v[6], v[7])
            Ψ += f * v[1]
            G1 = ntuple(i -> G1[i] + ∇1f[i] * v[1] + f * ∇1T[i], 3)
            G2 = ntuple(i -> G2[i] + ∇2f[i] * v[1] + f * ∇2T[i], 3)
            D1 += L1 * v[1] + 2 * dot3(∇1f, ∇1T) + f * v[8]
            D2 += L2 * v[1] + 2 * dot3(∇2f, ∇2T) + f * v[9]
            DM += MP * v[1] + dot3(∇1f, ∇2T) + dot3(∇2f, ∇1T) + f * v[10]
        end
        q1 = dot3(e12, G1); q2 = dot3(e12, G2); D3 = D1 + D2 + 2DM
        ψψ += w * abs2(Ψ); n1 += w * sum(abs2, G1); n2 += w * sum(abs2, G2)
        x12 += w * real(conj(G1[1]) * G2[1] + conj(G1[2]) * G2[2] + conj(G1[3]) * G2[3])
        qq += w * real(conj(q1) * q2)
        l11 += w * abs2(D1); l22 += w * abs2(D2); l12 += w * real(conj(D1) * D2); l33 += w * abs2(D3)
        pl1 += w * real(conj(Ψ) * D1); pl2 += w * real(conj(Ψ) * D2); pm += w * real(conj(Ψ) * DM)
        R = -(i1 * D1 + i2 * D2) / 2 - i3 * DM + (Vp - pt.E) * Ψ
        ψR += w * real(conj(Ψ) * R); RR += w * abs2(R)
        a11 = dot3(e1, G1); a12 = dot3(e1, G2); b21 = dot3(e2, G1); b22 = dot3(e2, G2)
        q13 -= w * (abs2(a11) + real(conj(a11) * a12)); q23 -= w * (abs2(b22) + real(conj(b22) * b21))
        # total angular momentum: LΨ = −i A, A = r₁ × ∇₁Ψ + r₂ × ∇₂Ψ (internal coordinates, from particle 3)
        A = cross3(rv[1], G1) .+ cross3(rv[2], G2)
        G3 = .-(G1 .+ G2)
        Gs = (G1, G2, G3)
        lc = ntuple(9) do q
            pp, k = divrem(q - 1, 3)
            c = cross3(rv[pp+1], Gs[k+1])
            lc[q] + w * real(conj(A[1]) * c[1] + conj(A[2]) * c[2] + conj(A[3]) * c[3])
        end
        AA = sum(abs2, A)
        lt = ntuple(3) do pp
            v = rv[pp]; r2v = v[1] * v[1] + v[2] * v[2] + v[3] * v[3]
            lt[pp] + w * (AA * r2v - 3 * abs2(v[1] * A[1] + v[2] * A[2] + v[3] * A[3]))
        end
    end
    (ψψ=ψψ, n1=n1, n2=n2, x12=x12, qq=qq, l11=l11, l22=l22, l12=l12, l33=l33, pl1=pl1, pl2=pl2, pm=pm, ψR=ψR, RR=RR, q13=q13, q23=q23,
     lc=lc, lt=lt)
end
function _frame_eval(terms, z1::T, x2::T, z2::T) where {T}
    s = zero(Complex{T})
    @inbounds for ((i, j, k), c) in terms
        s += c * (z1^i * x2^j * z2^k)
    end
    s
end

# workspace of the pointwise evaluation (one per task; no allocation per point)
_quadwork(pt, ::Type{T}) where {T} = (P1=Vector{T}(undef, pt.hi[1] - pt.lo[1] + 1), P2=Vector{T}(undef, pt.hi[2] - pt.lo[2] + 1),
    P12=Vector{T}(undef, pt.hi[3] - pt.lo[3] + 1), PSd=Vector{T}(undef, pt.hi[4] - pt.lo[4] + 1),
    PR=Vector{T}(undef, pt.hi[5] - pt.lo[5] + 1), Lj=Vector{T}(undef, pt.hi[6] + 1), ek=Vector{T}(undef, length(pt.α)),
    F3=Vector{T}(undef, length(pt.fac3)), F4=Vector{T}(undef, length(pt.fac4)), acc=Vector{T}(undef, 7pt.ncomp))
function _powtab!(v::Vector{T}, x::T, lo::Int, hi::Int) where {T}
    o = 1 - lo
    v[o] = one(T)
    for a in 1:hi; v[a+o] = v[a-1+o] * x; end
    if lo < 0
        ix = inv(x); for a in -1:-1:lo; v[a+o] = v[a+1+o] * ix; end
    end
    v
end

"The seven values of _expansions_at from the merged monomial table and workspace w (no allocation per point); for L ≥ 1 all component values go to w.acc (component 0's seven are returned)."
function _expansions_at!(w, pt, r1::T, r2::T, r12::T) where {T}
    lo, hi = pt.lo, pt.hi
    ss = r1 + r2; σ = ss + pt.βσ * r12
    _powtab!(w.P1, r1, lo[1], hi[1]); _powtab!(w.P2, r2, lo[2], hi[2]); _powtab!(w.P12, r12, lo[3], hi[3])
    _powtab!(w.PSd, pt.split ? ss : σ, lo[4], hi[4]); _powtab!(w.PR, inv(σ), lo[5], hi[5])
    _powtab!(w.Lj, log(σ), 0, hi[6])
    ek = w.ek
    @inbounds for k in eachindex(pt.α)
        ek[k] = exp(-(pt.α[k] * r1 + pt.β[k] * r2 + (pt.γ[k] + pt.η[k] * r12) * r12))
    end
    P1, P2, P12, PSd, PR, Lj, F3, F4, acc = w.P1, w.P2, w.P12, w.PSd, w.PR, w.Lj, w.F3, w.F4, w.acc
    @inbounds for i in eachindex(pt.fac3)
        A, B, C = pt.fac3[i]; F3[i] = P1[A] * P2[B] * P12[C]
    end
    @inbounds for i in eachindex(pt.fac4)
        D, R, j, k = pt.fac4[i]; F4[i] = PSd[D] * PR[R] * Lj[j] * ek[k]
    end
    fill!(acc, zero(T))
    tstart, tidx, tcoef = pt.tstart, pt.tidx, pt.tcoef
    @inbounds for i in eachindex(pt.mono)
        i3, i4 = pt.mono[i]
        x = F3[i3] * F4[i4]
        for q in tstart[i]:tstart[i+1]-1
            acc[tidx[q]] += tcoef[q] * x
        end
    end
    (acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7])
end
isinf_mass(m) = m == "Inf" || m == "inf" || (m isa Real && isinf(m))

function _expansions_at(pt, r1::T, r2::T, r12::T) where {T}
    lo, hi = pt.lo, pt.hi
    ss = r1 + r2; σ = ss + pt.βσ * r12
    P1 = _powtab(r1, lo[1], hi[1]); P2 = _powtab(r2, lo[2], hi[2]); P12 = _powtab(r12, lo[3], hi[3])
    PSd = _powtab(pt.split ? ss : σ, lo[4], hi[4]); PR = _powtab(inv(σ), lo[5], hi[5])
    ek = [exp(-(pt.α[k] * r1 + pt.β[k] * r2 + (pt.γ[k] + pt.η[k] * r12) * r12)) for k in eachindex(pt.α)]
    lσ = log(σ)
    ntuple(i -> _eval_terms(pt, pt.ts[i], P1, P2, P12, PSd, PR, lσ, ek), 7)
end

_mantissa_bits(::Type{Float64}) = 53
_mantissa_bits(::Type{MultiFloat{Float64,K}}) where {K} = 53K
_mantissa_bits(::Type{BigFloat}) = precision(BigFloat)

const _QUAD_MARGIN = 64

"""
    quad_type(wf; margin=64, floor=Float64x3) -> Type

Working type of the pointwise evaluation ψ = Σ c_k φ_k. The coefficients grow with the stage and the pointwise sum cancels
their magnitude, so the type is the smallest (floor → Float64x4 → BigFloat) keeping `margin` bits after the cancellation. The
absolute pointwise error ~eps·max|c|·max|φ| is nearly uniform, so the relative error is largest where ψ is small (tails).
For BigFloat, a warning is printed if the current `precision(BigFloat)` is not enough.
"""
function quad_type(wf; margin::Int=_QUAD_MARGIN, floor::Type=Float64x3)
    need = ceil(Int, log2(Float64(maximum(abs, wf.c)))) + margin
    for T in (Float64x3, Float64x4)
        _mantissa_bits(T) >= _mantissa_bits(floor) && _mantissa_bits(T) >= need && return T
    end
    precision(BigFloat) < need && @warn "quadrature type: BigFloat precision $(precision(BigFloat)) is not enough for the coefficient cancellation (log₂ max|c| + $margin = $need bits); raise setprecision(BigFloat, …)"
    BigFloat
end

# T = nothing chooses automatically (quad_type); an explicit T that cannot hold the cancellation gets a warning (T is still used)
function _resolve_quad_type(T, wf; floor::Type=Float64x3)
    T === nothing && return quad_type(wf; floor=floor)
    need = ceil(Int, log2(Float64(maximum(abs, wf.c)))) + _QUAD_MARGIN
    _mantissa_bits(T) < need && @warn "quadrature type $T ($(_mantissa_bits(T)) bits) is not enough for the coefficient cancellation (log₂ max|c| + $(_QUAD_MARGIN) = $need bits); omit T (automatic) or use $(quad_type(wf; floor=floor))"
    T
end

"""
    expectation_quad(sys, wf, ops; logvar=LogVariable(), N=50, T=nothing) -> Vector{BigFloat}

Expectation values of operators that element integrals cannot form (`P4`, `LaplacianProduct`, `EnergyMinusVSquared`, `Retardation`,
`InverseSquare`, …) by pointwise evaluation and three-dimensional perimetric quadrature (as σ², 6N³ points); the r₁₂ Gaussian is
allowed. `P4` and `InverseSquare` are mass independent. `DistancePower` is accepted (to check the quadrature against the
closed-form `expectation`). `LaplacianProduct`, `Retardation`, `EnergyMinusVSquared` are parts of the formulas for two-electron
atoms (particle 3 infinitely heavy). Check convergence with two values of N. The working type T defaults (`nothing`) to an
automatic choice from the coefficient magnitudes (`quad_type`: Float64x3 → Float64x4 → BigFloat).
"""
function expectation_quad(sys::ThreeBody, wf, ops::AbstractVector; logvar::LogVariable=LogVariable(), N::Int=50,
                          T::Union{Type,Nothing}=nothing, coords=:perimetric)
    if T === nothing && quad_type(wf) === BigFloat
        # automatic BigFloat is reduced to the needed bits + 32 (full working precision pointwise is several times slower)
        bits = ceil(Int, log2(Float64(maximum(abs, wf.c)))) + _QUAD_MARGIN + 32
        precision(BigFloat) > bits &&
            return setprecision(() -> expectation_quad(sys, wf, ops; logvar=logvar, N=N, T=BigFloat, coords=coords), BigFloat, bits)
    end
    T = _resolve_quad_type(T, wf)
    all(o -> o isa QuadObservable || o isa DistancePower, ops) || throw(ArgumentError("expectation_quad operators must be P4, LaplacianProduct, EnergyMinusVSquared, Retardation, InverseSquare, ArakiSucherPart, DrachmanDelta12, DrachmanDelta, KineticQuad, QTerm, DistancePower"))
    pt = _point_expansions(sys, wf, logvar, T)
    # input pair → internal distance (_distance_map maps internal → input, so use the inverse; they differ for cyclic orders)
    inv_dm = Tuple(invperm(collect(_distance_map(pt.π))))
    internal(i) = findfirst(==(i), pt.π)           # input particle i → internal number
    grid = _quad_grid(sys, wf, pt, coords, N, T, logvar)
    nops = length(ops)
    nout = _grid_outer(grid)
    acc = [zeros(T, nops + 1) for _ in 1:nout]
    opsT = Tuple(map(o -> _prepare_quad(o, pt, inv_dm, T), ops))      # a tuple of operators (different types) keeps the point loop type stable
    _quad_sum!(acc, grid, pt, opsT, inv_dm, internal, T)
    tot = reduce(+, acc)
    [BigFloat(tot[q+1] / tot[1]) for q in 1:nops]
end

function _quad_sum!(acc, grid, pt, opsT::Tuple, inv_dm, internal, ::Type{T}) where {T}
    Threads.@threads :dynamic for io in eachindex(acc)
        local ac = acc[io]
        local wk = _quadwork(pt, T)
        _grid_points(grid, io) do r1, r2, r12, jac
            _expansions_at!(wk, pt, r1, r2, r12)
            local iv = _invariants(pt, wk.acc, r1, r2, r12)
            ac[1] += jac * iv.ψψ
            local vals = map(o -> _quad_integrand(o, pt, inv_dm, internal, iv, r1, r2, r12), opsT)
            for q in eachindex(vals)
                ac[q+1] += jac * vals[q]
            end
        end
    end
    acc
end

# --- Quadrature grids ---------------------------------------------------------------------------------------
# :perimetric (default): perimetric u, v, w as the largest coordinate s (half-line DE, scale 1/ρmin) and two ratios (tanh-sinh), 6N³ points.
# PairAxis: the pair distance R as axis, the other two distances p₁, p₂ as sum a = p₁ + p₂ ∈ [R, ∞) and difference b ∈ [−R, R]. R and a − R use
# the log-sinh map x = x₀ exp(c sinh t) (double-exponential decay at both ends; x₀ and c = width/x₀ from closed-form ⟨x⟩, ⟨x²⟩), b uses tanh-sinh.
"""
    PairAxis(i, j; NR=N, Na=N, Nb=N)

Grid for `expectation_quad(…; coords=PairAxis(i, j))`: the distance R of input particles i, j as axis, R and (sum of the other two
distances) − R each cut at their own scale (centers and widths from closed-form expectation values). For systems with two scales
(the internuclear ridge of H₂⁺, μ–α and electrons in μHe). NR·Na·Nb points (`N` is inherited); coalescence lies at a corner of (a, b).
"""
struct PairAxis
    pair::NTuple{2,Int}
    NR::Int
    Na::Int
    Nb::Int
end
PairAxis(i::Integer, j::Integer; NR::Int=0, Na::Int=0, Nb::Int=0) =
    (p = (min(i, j), max(i, j)); p in _PAIR_PARTICLES ? PairAxis(p, NR, Na, Nb) : throw(ArgumentError("pair must be (1,3), (2,3) or (1,2)")))

# x = x₀ exp(c sinh t), trapezoidal in t ∈ [−T₁, T₂]; lower end x₀e^{−K₁}, upper end x₀ + 80·width (1e-35 for exponential tails).
# The lower end must be close to 0 (the a − R integrand is finite at 0), but not so small that negative distance powers overflow to
# Inf·0 = NaN: K₁ is chosen from the total negative power neg so that e^{K₁·neg} ≤ e^{600} (truncation error relative e^{−K₁}).
function _logsinh_rule(::Type{T}, N::Int, x0, w; neg::Int=0) where {T}
    x0b = BigFloat(x0); c = BigFloat(w) / x0b
    K1 = big(neg == 0 ? 80 : clamp(600 / neg, 20, 80)); K2 = log(1 + 80 / c)
    T1 = asinh(K1 / c); T2 = asinh(K2 / c); h = (T1 + T2) / (N - 1)
    map(0:N-1) do i
        t = -T1 + i * h; x = x0b * exp(c * sinh(t))
        (T(x), T(h * x * c * cosh(t)))
    end
end

_grid_outer(g) = g.kind === :perimetric ? length(g.S) : length(g.R)

function _quad_grid(sys, wf, pt, coords, N, ::Type{T}, logvar) where {T}
    if coords === :perimetric
        return (kind=:perimetric, S=_de_half(T, N, 1 / pt.ρmin), Q=_tanhsinh01(T, N))
    end
    coords isa PairAxis || throw(ArgumentError("coords must be :perimetric or PairAxis(i, j)"))
    NR = coords.NR > 0 ? coords.NR : N; Na = coords.Na > 0 ? coords.Na : N; Nb = coords.Nb > 0 ? coords.Nb : N
    # internal distance indices: axis d and the others (d1, d2)
    dm = _distance_map(pt.π)                                            # internal distance → input distance
    d = findfirst(==(findfirst(==(coords.pair), _PAIR_PARTICLES)), dm)
    d1, d2 = filter(!=(d), (1, 2, 3))
    # centers and widths: ⟨R⟩, ⟨R²⟩ and ⟨a − R⟩, ⟨(a − R)²⟩ (DistancePower in input numbering)
    inp = ntuple(k -> _PAIR_PARTICLES[dm[k]], 3)                         # internal distance k → input pair
    pw(k, e) = (v = zeros(Int, 3); v[findfirst(==(inp[k]), _PAIR_PARTICLES)] += e; v)
    ops = [DistancePower(pw(d, 1)...), DistancePower(pw(d, 2)...), DistancePower(pw(d1, 1)...), DistancePower(pw(d2, 1)...),
           DistancePower(pw(d1, 2)...), DistancePower(pw(d2, 2)...), DistancePower((pw(d1, 1) .+ pw(d2, 1))...),
           DistancePower((pw(d, 1) .+ pw(d1, 1))...), DistancePower((pw(d, 1) .+ pw(d2, 1))...)]
    m = _axis_moments(sys, wf, ops, logvar, pt.π)
    R1, R2 = m[1], m[2]
    u1 = m[3] + m[4] - m[1]
    u2 = m[5] + m[6] + R2 + 2m[7] - 2m[8] - 2m[9]                      # ⟨(p₁ + p₂ − R)²⟩
    wR = sqrt(max(R2 - R1^2, R1^2 / 100)); wu = sqrt(max(u2 - u1^2, u1^2 / 100))
    neg = sum(i -> -min(pt.lo[i], 0), 1:4) + max(pt.hi[5], 0)          # total negative power that can overflow at a point (r^lo, σ^{−R})
    (kind=:axis, d=d, d1=d1, d2=d2, R=_logsinh_rule(T, NR, R1, wR; neg=neg), A=_logsinh_rule(T, Na, u1, wu; neg=neg), Q=_tanhsinh01(T, Nb))
end

# Closed-form moments (nine) for the grid centers and widths, computed once per wave function at a precision reduced to the coefficient
# cancellation (two digits suffice for scales). Keyed by content (E, ends and count of the coefficients, operators, logvar, order),
# since objectid alone can be reused by another stage's array after it is freed.
const _AXIS_MOMENTS = Dict{Any,Vector{BigFloat}}()
const _AXIS_MOMENTS_LOCK = ReentrantLock()
function _axis_moments(sys, wf, ops, logvar, π)
    key = (BigFloat(wf.E), BigFloat(wf.c[1]), BigFloat(wf.c[end]), length(wf.c), get(wf, :L, 0), ops, logvar.βσ, logvar.split, π,
           inverse_masses(sys, Float64))
    hit = lock(() -> get(_AXIS_MOMENTS, key, nothing), _AXIS_MOMENTS_LOCK)
    hit === nothing || return hit
    bits = min(precision(BigFloat), 2 * ceil(Int, log2(Float64(maximum(abs, wf.c)))) + 128)
    m = setprecision(() -> BigFloat.(expectation(sys, wf, ops; logvar=logvar)), BigFloat, max(bits, 128))
    lock(_AXIS_MOMENTS_LOCK) do
        length(_AXIS_MOMENTS) >= 64 && empty!(_AXIS_MOMENTS)
        _AXIS_MOMENTS[key] = m
    end
    m
end

# pass the points of outer index io to f(r₁, r₂, r₁₂, volume element × weight)
function _grid_points(f, g, io)
    if g.kind === :perimetric
        sv, wsv = g.S[io]
        for (ρ, wρ) in g.Q, (ξ, wξ) in g.Q
            wt = wsv * wρ * wξ * sv^2 * ρ
            for (yy, zz) in ((ρ, ρ * ξ), (ρ * ξ, ρ))       # two triangles × largest coordinate (u, v, w)
                for (u, v, w) in ((sv, sv * yy, sv * zz), (sv * yy, sv, sv * zz), (sv * yy, sv * zz, sv))
                    r1 = v + w; r2 = u + w; r12 = u + v
                    f(r1, r2, r12, r1 * r2 * r12 * wt)
                end
            end
        end
    else
        R, wR = g.R[io]
        for (x, wx) in g.A, (q, wq) in g.Q
            a = R + x; b = R * (2q - 1)
            p1 = (a + b) / 2; p2 = (a - b) / 2
            rr = g.d == 1 ? (R, p1, p2) : g.d == 2 ? (p1, R, p2) : (p1, p2, R)     # (d1, d2) in increasing order
            f(rr[1], rr[2], rr[3], rr[1] * rr[2] * rr[3] * wR * wx * wq * R)       # dp₁dp₂ = da db/2, db = 2R dq
        end
    end
    nothing
end

expectation_quad(sys::ThreeBody, wf, op::QuadObservable; kwargs...) = expectation_quad(sys, wf, [op]; kwargs...)[1]

# per-operator tables in the working type before the point loop (default: unchanged)
_prepare_quad(o, pt, dm, ::Type{T}) where {T} = o

# Uehling: I(x) = ∫₁^∞ (2t²+1)√(t²−1)/t⁴ e^{−xt} dt = ∫₀^∞ (2c²+1)s²/c⁴ e^{−x c} du (c = cosh u, s = sinh u), trapezoidal with step h.
# The integrand is even in u, 0 at u = 0, with the only pole at c = 0 (u = iπ/2), so the error is ~e^{−π²/h}; cut where x·c > X (e^{−X})
struct _UehlingTab{T}
    d::Int          # internal distance index
    pref::T         # q_i q_j α/(3π)
    x0::T           # x = x0·r = 2r/α
    X::T
    c::Vector{T}
    w::Vector{T}    # h (2c²+1)s²/c⁴
end
function _prepare_quad(o::Uehling, pt, dm, ::Type{T}) where {T}
    d = dm[findfirst(==(o.pair), _PAIR_PARTICLES)]
    pw = ntuple(k -> k == d ? -1 : 0, 3)
    q = zero(T)
    for (p, c) in pt.V
        Tuple(p) == pw && (q += c)
    end
    α = 1 / parse(BigFloat, o.inv_alpha)
    L = -log(BigFloat(eps(T))) * big"1.15" + 10
    h = BigFloat(π)^2 / L
    K = ceil(Int, 45 / h)                         # u ≤ 45: covers x ≥ 1e-17
    us = (1:K) .* h
    c = cosh.(us); s = sinh.(us)
    _UehlingTab{T}(d, T(q * α / (3 * BigFloat(π))), T(2 / α), T(L), T.(c), T.(h .* (2 .* c .^ 2 .+ 1) .* s .^ 2 ./ c .^ 4))
end
function _uehling_integral(o::_UehlingTab{T}, r::T) where {T}
    x = o.x0 * r
    acc = zero(T)
    @inbounds for k in eachindex(o.c)
        xc = x * o.c[k]
        xc > o.X && break
        acc += o.w[k] * exp(-xc)
    end
    acc
end
_quad_integrand(o::_UehlingTab, pt, dm, internal, iv, r1, r2, r12) =
    (r = (r1, r2, r12)[o.d]; iv.ψψ * o.pref * _uehling_integral(o, r) / r)

# integrand per operator from the point invariants (`_invariants`), independent of L
function _quad_integrand(o::Union{QuadObservable,DistancePower}, pt, dm, internal, iv, r1, r2, r12)
    if o isa P4
        ii = internal(o.i)
        return ii == 1 ? iv.l11 : ii == 2 ? iv.l22 : iv.l33      # P₃ = i(∇₁ + ∇₂)
    elseif o isa LaplacianProduct
        return iv.l12
    elseif o isa EnergyMinusVSquared
        x = pt.E - _potential_at(pt, r1, r2, r12)
        return iv.ψψ * x * x
    elseif o isa ArakiSucherPart || o isa DrachmanDelta12
        grad2 = iv.n1 + iv.n2                                     # |∇₁ψ|² + |∇₂ψ|²
        V = _potential_at(pt, r1, r2, r12)
        o isa DrachmanDelta12 && return (2 * iv.ψψ * (pt.E - V) - grad2) / r12
        return (-grad2 + 2 * iv.ψψ * (pt.E - V)) * log(r12) / r12
    elseif o isa DistancePower        # the same quantity as the closed form (expectation); for checking the quadrature
        rr = (r1, r2, r12); x = iv.ψψ
        for k in 1:3
            p = o.pow[k]; p == 0 && continue
            x *= rr[dm[k]]^p
        end
        return x
    elseif o isa DrachmanDelta || o isa QTerm
        d = dm[findfirst(==(o.pair), _PAIR_PARTICLES)]
        i1, i2, i3 = pt.im
        G = i1 * iv.n1 + i2 * iv.n2 + 2 * i3 * iv.x12
        V = _potential_at(pt, r1, r2, r12)
        invμ = d == 1 ? i1 : d == 2 ? i2 : i1 + i2 - 2i3            # inverse reduced mass of the pair
        r = (r1, r2, r12)[d]
        x = (2 * iv.ψψ * (pt.E - V) - G) / (r * 2 * oftype(r1, π) * invμ)
        o isa DrachmanDelta && return x
        return x * (log(r) + 1 + oftype(r1, Base.MathConstants.eulergamma))
    elseif o isa InverseSquare
        d = dm[findfirst(==(o.pair), _PAIR_PARTICLES)]
        r = (r1, r2, r12)[d]
        return iv.ψψ / (r * r)
    elseif o isa _Residual
        return o.k == 1 ? iv.ψR : iv.RR
    elseif o isa LDot || o isa LTensor
        # input pair (i, j) → internal pair (a, b); R_i − R_j is +v(a,b) if a < b, else −v
        a0, b0 = internal(o.pair[1]), internal(o.pair[2])
        a, b = minmax(a0, b0)
        pidx = (a, b) == (1, 3) ? 1 : (a, b) == (2, 3) ? 2 : 3
        r = (r1, r2, r12)[pidx]
        o isa LTensor && return iv.lt[pidx] / r^5
        sgn = a0 < b0 ? 1 : -1
        return sgn * iv.lc[3(pidx-1)+internal(o.k)] * r^o.power
    elseif o isa KineticQuad
        i1, i2, i3 = pt.im
        o.form === :laplacian && return -(i1 * iv.pl1 + i2 * iv.pl2) / 2 - i3 * iv.pm
        return (i1 * iv.n1 + i2 * iv.n2) / 2 + i3 * iv.x12
    else   # Retardation(i, j): ∇ᵢψ·T·∇ⱼψ, T = (1 + r̂r̂)/r (pair vector; internal (1,2) is r⃗₁ − r⃗₂, (1,3), (2,3) are r⃗₁, r⃗₂)
        a, b = minmax(internal(o.pair[1]), internal(o.pair[2]))
        (a, b) == (1, 2) && return (iv.x12 + iv.qq) / r12
        (a, b) == (1, 3) && return (-(iv.n1 + iv.x12) + iv.q13) / r1        # ∇₃ = −(∇₁ + ∇₂)
        return (-(iv.n2 + iv.x12) + iv.q23) / r2
    end
end
function _potential_at(pt, r1, r2, r12)
    rr = (r1, r2, r12)
    V = zero(r1)
    for (p, c) in pt.V
        V += c * rr[1]^p[1] * rr[2]^p[2] * rr[3]^p[3]
    end
    V
end

"""
    relativistic_correction(sys, wf; regularize=true, N=50, logvar=LogVariable()) -> NamedTuple

Breit–Pauli correction of singlet S states of two-electron atoms (input particles 1, 2 of mass 1 and charge −1, particle 3 of
infinite mass and charge Z): δE_rel/α² = −(⟨p₁⁴⟩+⟨p₂⁴⟩)/8 + (Zπ/2)⟨δ(r₁₃)+δ(r₂₃)⟩ + π⟨δ(r₁₂)⟩ − ½⟨p₁(δ/r+rr/r³)p₂⟩
(arXiv:2301.05442 Eq. 5). Electron–nucleus δ in closed form (`Delta`); electron–electron δ with `delta12=:closed` (default) or
`:drachman` (global form). p⁴ with `regularize=true` as Σp⁴ = 4⟨(E−V)²⟩ − 2⟨∇₁²ψ|∇₂²ψ⟩ (standard regularization), false directly ⟨(∇ᵢ²ψ)²⟩.
"""
function relativistic_correction(sys::ThreeBody, wf; regularize::Bool=true, N::Int=50, logvar::LogVariable=LogVariable(),
                                 T::Union{Type,Nothing}=nothing, delta12::Symbol=:closed)
    (sys.masses[1] == "1" && sys.masses[2] == "1" && isinf_mass(sys.masses[3])) ||
        throw(ArgumentError("relativistic_correction requires a two-electron atom (1, 1, ∞) in this input order"))
    q = Dict(t.pow => exact(BigFloat, t.c) for t in sys.terms if t isa PowerTerm)
    (q[(0, 0, -1)] == 1 && q[(-1, 0, 0)] == q[(0, -1, 0)]) || throw(ArgumentError("must be a Coulomb system of two electrons and one nucleus"))
    delta12 in (:closed, :drachman) || throw(ArgumentError("delta12 must be :drachman or :closed"))
    # wrapper of the general breit_pauli (singlet ⟨s₁·s₂⟩ = −3/4, electron–nucleus δ closed form, electron–electron by delta12)
    r = breit_pauli(sys, wf; spin_half=(true, true, false), spin_dot=Dict((1, 2) => -0.75), N=N, T=T, logvar=logvar,
                    delta=Dict((1, 3) => :closed, (2, 3) => :closed, (1, 2) => delta12), regularize=regularize)
    (Erel=r.E, p4sum=r.p4sum, p4_direct=r.p4[1] + r.p4[2], delta_nuc=(r.delta[(1, 3)], r.delta[(2, 3)]), delta12=r.delta[(1, 2)],
     retardation=r.retardation[(1, 2)], Z=-q[(-1, 0, 0)])
end

"""
    breit_pauli(sys, wf; spin_half=(true, true, true), spin_dot=Dict(), coords=:perimetric, N=50, T=nothing,
                delta=:closed, logvar=LogVariable()) -> NamedTuple

Spin-independent Breit–Pauli correction δE/α² (atomic units, masses in units of m_e) of three-body Coulomb systems of any masses and
charges (general form of 1810.07929 Eqs. 3–6):
  −Σᵢ ⟨pᵢ⁴⟩/(8mᵢ³) − (π/2) Σ_{i<j} zᵢzⱼ (sᵢ/mᵢ² + sⱼ/mⱼ²) ⟨δ(r_ij)⟩ − Σ_{i<j} zᵢzⱼ/(2mᵢmⱼ) ⟨pᵢ(δ/r + r r/r³)pⱼ⟩
(sᵢ = 1 is the Darwin term of a spin-1/2 particle; set per particle with `spin_half` (false for spin-0 or spin-1 nuclei); terms of
infinite masses vanish). `spin_dot = Dict((i, j) => ⟨sᵢ·sⱼ⟩)` adds the spin–spin contact term −(8π/3) zᵢzⱼ/(mᵢmⱼ) ⟨sᵢ·sⱼ⟩⟨δ(r_ij)⟩
(g = 2) for those pairs. Spin–orbit and tensor terms are not included (fine structure for L ≥ 1 is separate). zᵢzⱼ are read from
the Coulomb terms; p⁴ is the direct ⟨(∇²ψ)²⟩; δ with `delta = :closed` or `:drachman`. Use `coords = PairAxis(i, j)` for molecules
(ridge in the internuclear distance) or systems with separated scales.
"""
function breit_pauli(sys::ThreeBody, wf; spin_half=(true, true, true), spin_dot=Dict{NTuple{2,Int},Float64}(), coords=:perimetric,
                     N::Int=50, T::Union{Type,Nothing}=nothing, delta=:closed, regularize::Bool=false, logvar::LogVariable=LogVariable())
    regularize && !(sys.masses[1] == "1" && sys.masses[2] == "1" && isinf_mass(sys.masses[3])) &&
        throw(ArgumentError("regularize=true requires a (1, 1, ∞) system (Σp⁴ = 4⟨(E−V)²⟩ − 2⟨∇₁²ψ|∇₂²ψ⟩)"))
    zz = Dict{NTuple{2,Int},BigFloat}()
    for t in sys.terms
        (t isa PowerTerm && count(==(-1), t.pow) == 1 && count(==(0), t.pow) == 2) ||
            throw(ArgumentError("breit_pauli requires a Coulomb system (term $(t))"))
        zz[_PAIR_PARTICLES[findfirst(==(-1), t.pow)]] = exact(BigFloat, t.c)
    end
    m = ntuple(i -> isinf_mass(sys.masses[i]) ? BigFloat(Inf) : exact(BigFloat, sys.masses[i]), 3)
    fin = ntuple(i -> isfinite(m[i]), 3)
    pairs = [p for p in _PAIR_PARTICLES if get(zz, p, big(0)) != 0]
    qops = Any[P4(i) for i in 1:3 if fin[i]]
    rpairs = [p for p in pairs if fin[p[1]] && fin[p[2]]]
    append!(qops, [Retardation(p...) for p in rpairs])
    # δ form per pair (`delta` is a symbol or Dict(pair => symbol); missing pairs use :closed)
    dmode(p) = delta isa Symbol ? delta : get(delta, p, :closed)
    all(p -> dmode(p) in (:closed, :drachman), pairs) || throw(ArgumentError("delta must be :closed or :drachman (a Dict per pair is allowed)"))
    dpairs = [p for p in pairs if dmode(p) === :drachman]; cpairs = [p for p in pairs if dmode(p) === :closed]
    append!(qops, [DrachmanDelta(p...) for p in dpairs])
    regularize && append!(qops, [EnergyMinusVSquared(), LaplacianProduct()])
    q = expectation_quad(sys, wf, qops; N=N, T=T, logvar=logvar, coords=coords)
    k = 0
    p4 = ntuple(i -> fin[i] ? (k += 1; q[k]) : big(0), 3)
    R = Dict(p => (k += 1; q[k]) for p in rpairs)
    δ = Dict{NTuple{2,Int},BigFloat}(p => (k += 1; q[k]) for p in dpairs)
    isempty(cpairs) || merge!(δ, Dict(zip(cpairs, expectation(sys, wf, [Delta(p...) for p in cpairs]; logvar=logvar))))
    # regularization
    p4sum = regularize ? 4q[k+1] - 2q[k+2] : p4[1] + p4[2]
    piB = BigFloat(Base.MathConstants.pi)
    inv2(i) = fin[i] ? 1 / m[i]^2 : big(0)
    Ekin = regularize ? -p4sum / 8 : -sum(fin[i] ? p4[i] / (8m[i]^3) : big(0) for i in 1:3)
    Edar = -piB / 2 * sum((zz[p] * ((spin_half[p[1]] ? inv2(p[1]) : 0) + (spin_half[p[2]] ? inv2(p[2]) : 0)) * δ[p] for p in pairs); init=big(0))
    Eoo = -sum((zz[p] / (2m[p[1]] * m[p[2]]) * R[p] for p in rpairs); init=big(0))
    Ess = big(0)
    for (pr, sd) in spin_dot
        p = minmax(pr...)
        (p in pairs && fin[p[1]] && fin[p[2]]) || throw(ArgumentError("spin_dot pair $pr must be a Coulomb pair with both masses finite"))
        Ess += -8piB / 3 * zz[p] / (m[p[1]] * m[p[2]]) * BigFloat(sd) * δ[p]
    end
    (E=Ekin + Edar + Eoo + Ess, kinetic=Ekin, darwin=Edar, orbit_orbit=Eoo, spin_spin=Ess, p4=p4, p4sum=p4sum, retardation=R, delta=δ, zz=zz)
end

"""
    fine_structure(sys, wf; coords=:perimetric, N=50, T=nothing, logvar=LogVariable()) -> NamedTuple

Leading fine structure mα⁴ (+m/M) of triplet ³L states (L ≥ 1) of two-electron atoms (input particles 1, 2 electrons of mass 1 and
charge −1, 3 a nucleus of charge Z and mass M, possibly ∞). Pachucki–Yerokhin arXiv:1011.2467 Eq. (2) (a_e = 0) and Eq. (8) (recoil):
  H_fs = ¼[σ₁·σ₂/r³ − 3σ₁·r σ₂·r/r⁵] + (Z/4)Σᵢ rᵢ×pᵢ·σᵢ/rᵢ³ + (1/4r³)[(σ₂+2σ₁)·r×p₂ − (σ₁+2σ₂)·r×p₁]
         + (Z/2)(m/M)Σᵢ (rᵢ/rᵢ³)×(p₁+p₂)·σᵢ, r = r₁ − r₂.
For triplets σᵢ → S (vector part), sym(σ₁⊗σ₂) → 2 sym(S) (tensor part). Projected within the L multiplet:
  E_J/α² = C·X_J + ½κ·Y_J, X_J = ⟨L·S⟩_J, Y_J = X_J² + X_J/2 − L(L+1)S(S+1)/3,
  C = ⟨L·V⟩/(L(L+1)) (V the spatial operator of the vector part, `LDot`), κ = ⟨L Q L⟩/c_L (`LTensor((1,2))`).
Mass scaling and mass polarization are in the finite-mass wave function (units m_e = 1). E is a dictionary over J = |L−1|..L+1
(atomic units × α²). He 2³P agrees with Table 2 of 1011.2467 (mα⁴(+m/M)).
"""
function fine_structure(sys::ThreeBody, wf; coords=:perimetric, N::Int=50, T::Union{Type,Nothing}=nothing, logvar::LogVariable=LogVariable())
    L = get(wf, :L, 0); L >= 1 || throw(ArgumentError("fine structure requires L ≥ 1"))
    get(wf, :unnatural, false) && throw(ArgumentError("unnatural parity is not supported"))
    (sys.masses[1] == "1" && sys.masses[2] == "1") || throw(ArgumentError("input particles 1, 2 must be electrons (mass 1)"))
    q = Dict(t.pow => exact(BigFloat, t.c) for t in sys.terms if t isa PowerTerm)
    (q[(0, 0, -1)] == 1 && q[(-1, 0, 0)] == q[(0, -1, 0)]) || throw(ArgumentError("must be a Coulomb system of two electrons and one nucleus"))
    Z = -q[(-1, 0, 0)]
    invM = isinf_mass(sys.masses[3]) ? big(0) : 1 / exact(BigFloat, sys.masses[3])
    ops = [LDot((1, 3), 1), LDot((2, 3), 2), LDot((1, 3), 3), LDot((2, 3), 3), LDot((1, 2), 1), LDot((1, 2), 2), LTensor((1, 2)),
           LDot((1, 3), 1; power=0), LDot((2, 3), 2; power=0)]
    v = expectation_quad(sys, wf, ops; N=N, T=T, logvar=logvar, coords=coords)
    LL = big(L * (L + 1))
    so = Z / 4 * (v[1] + v[2]); rec = -Z / 2 * invM * (v[3] + v[4]); soo = 3 * (v[6] - v[5]) / 4
    C = (so + rec + soo) / LL
    κ = v[7] / (2LL^2 / 3 - LL / 2)
    E = Dict{Int,BigFloat}()
    for J in abs(L - 1):L+1
        X = big(J * (J + 1) - L * (L + 1) - 2) / 2
        Y = X^2 + X / 2 - LL * 2 / 3
        E[J] = C * X + κ / 2 * Y
    end
    (E=E, C=C, κ=κ, spin_orbit=so / LL, recoil=rec / LL, spin_other_orbit=soo / LL, L2=v[8] + v[9], Z=Z)
end

"""
    spin_hamiltonian(sys, wf; spin=(true, true, false), g=(2, 2, 2), coords=:perimetric, N=50, T=nothing,
                     logvar=LogVariable(), subspace=nothing) -> NamedTuple

Effective Hamiltonian of the spin-dependent Breit–Pauli terms (atomic units × α²) of a three-body Coulomb system with spin-1/2 particles
(`spin[i]`, g factors `g[i]`), projected on the L multiplet and diagonalized in (2L+1) × spin space. Terms (r_ij = R_i − R_j, zᵢ charges):
  spin–orbit with own orbit: −(gᵢ−1) zᵢzⱼ/(2mᵢ²) (r_ij × pᵢ)·sᵢ/r_ij³
  spin–other-orbit: + zᵢzⱼ gᵢ/(2mᵢmⱼ) (r_ij × pⱼ)·sᵢ/r_ij³
  spin–spin: zᵢzⱼ gᵢgⱼ/(4mᵢmⱼ) [Σ_ab s_ia s_jb Q_ab − (8π/3) sᵢ·sⱼ δ(r_ij)], Q_ab = (δ_ab r² − 3r_a r_b)/r⁵
(general form of Pachucki–Yerokhin arXiv:1011.2467 Eqs. 2 and 8; agrees with `fine_structure` for He). Projection: vector parts
⟨L·V⟩/(L(L+1)) L, tensors κ(½{L_a,L_b} − δ_ab L²/3); only the spin–spin contact term for L = 0. `subspace` (a projector of
size (2L+1)·2^nspin) restricts the diagonalization (e.g. spins matching the spatial symmetry of identical fermions).
Returns levels `levels` ((E, F) with F the total angular momentum quantum number of L + Σs), coefficients `A` (L·sᵢ per particle), `B`, `C` (tensor and contact per pair).
"""
function spin_hamiltonian(sys::ThreeBody, wf; spin=(true, true, false), g=(2, 2, 2), coords=:perimetric, N::Int=50, T::Union{Type,Nothing}=nothing,
                          logvar::LogVariable=LogVariable(), subspace=nothing, qvals=nothing, δvals=nothing)
    L = get(wf, :L, 0)
    get(wf, :unnatural, false) && throw(ArgumentError("unnatural parity is not supported"))
    zz = Dict{NTuple{2,Int},BigFloat}()
    for t in sys.terms
        (t isa PowerTerm && count(==(-1), t.pow) == 1 && count(==(0), t.pow) == 2) || throw(ArgumentError("Coulomb systems only"))
        zz[_PAIR_PARTICLES[findfirst(==(-1), t.pow)]] = exact(BigFloat, t.c)
    end
    z(i, j) = get(zz, minmax(i, j), big(0))
    m = ntuple(i -> isinf_mass(sys.masses[i]) ? BigFloat(Inf) : exact(BigFloat, sys.masses[i]), 3)
    gb = ntuple(i -> BigFloat(g[i]), 3)
    sp = [i for i in 1:3 if spin[i]]
    any(i -> !isfinite(m[i]), sp) && throw(ArgumentError("spins of infinitely heavy particles need explicit magnetic moments (not implemented)"))
    # spatial parts (scalars summed over M) by quadrature
    ops = Any[]; keyof = Dict{Any,Int}()
    need(o) = haskey(keyof, o) ? keyof[o] : (push!(ops, o); keyof[o] = length(ops))
    sgn(i, j) = i < j ? 1 : -1                               # r of LDot is oriented (min, max)
    pairs = [(i, j) for i in sp for j in 1:3 if j != i && z(i, j) != 0]
    if L >= 1
        for (i, j) in pairs
            need(LDot(minmax(i, j), i)); isfinite(m[j]) && need(LDot(minmax(i, j), j))
        end
        for a in sp, b in sp
            a < b && z(a, b) != 0 && need(LTensor((a, b)))
        end
    end
    # qvals, δvals: previous results for the same system and spins (reused when only g changes)
    q = qvals !== nothing ? qvals : isempty(ops) ? BigFloat[] : expectation_quad(sys, wf, ops; N=N, T=T, logvar=logvar, coords=coords)
    val(o) = q[keyof[o]]
    LL = big(L * (L + 1))
    A = Dict{Int,BigFloat}()
    for i in sp
        a = big(0)
        if L >= 1
            for (ii, j) in pairs
                ii == i || continue
                s = sgn(i, j)
                a += -(gb[i] - 1) * z(i, j) / (2m[i]^2) * s * val(LDot(minmax(i, j), i))
                isfinite(m[j]) && (a += z(i, j) * gb[i] / (2m[i] * m[j]) * s * val(LDot(minmax(i, j), j)))
            end
            a /= LL
        end
        A[i] = a
    end
    cL = 2LL^2 / 3 - LL / 2
    B = Dict{NTuple{2,Int},BigFloat}(); C = Dict{NTuple{2,Int},BigFloat}()
    sspairs = [(a, b) for a in sp for b in sp if a < b && z(a, b) != 0]
    δ = δvals !== nothing ? δvals : isempty(sspairs) ? BigFloat[] : expectation(sys, wf, [Delta(p...) for p in sspairs]; logvar=logvar)
    for (k, (a, b)) in enumerate(sspairs)
        γγ = z(a, b) * gb[a] * gb[b] / (4m[a] * m[b])
        B[(a, b)] = L >= 1 ? γγ * val(LTensor((a, b))) / cL : big(0)
        C[(a, b)] = -γγ * 8 * BigFloat(Base.MathConstants.pi) / 3 * δ[k]
    end
    # build and diagonalize in (2L+1) ⊗ (2)^{nspin} (complex Float64; level spacings are set by coefficients to relative 1e-16)
    jm(j) = (mm = j:-1:-j; n = length(mm); Jp = zeros(n, n);
             for k in 2:n; Jp[k-1, k] = sqrt(j * (j + 1) - mm[k] * (mm[k] + 1)); end;
             (ComplexF64.((Jp + Jp') / 2), ComplexF64.((Jp - Jp') / 2im), ComplexF64.(Diagonal(collect(float.(mm))))))
    dims = [2L + 1; fill(2, length(sp))]
    emb(op, pos) = (mats = [Matrix{ComplexF64}(I, d, d) for d in dims]; mats[pos] = op; reduce(kron, mats))
    Lm = [emb(x, 1) for x in jm(L)]
    Sm = Dict(i => [emb(x, 1 + k) for x in jm(1 // 2)] for (k, i) in enumerate(sp))
    D = prod(dims); H = zeros(ComplexF64, D, D)
    L2 = sum(x -> x * x, Lm)
    TL(a, b) = (Lm[a] * Lm[b] + Lm[b] * Lm[a]) / 2 - (a == b ? L2 / 3 : 0 * L2)
    for i in sp, a in 1:3; H .+= Float64(A[i]) .* (Lm[a] * Sm[i][a]); end
    for (a, b) in sspairs
        for x in 1:3, y in 1:3; H .+= Float64(B[(a, b)]) .* (TL(x, y) * Sm[a][x] * Sm[b][y]); end
        for x in 1:3; H .+= Float64(C[(a, b)]) .* (Sm[a][x] * Sm[b][x]); end
    end
    Fv = [Lm[a] + sum(Sm[i][a] for i in sp; init=zeros(ComplexF64, D, D)) for a in 1:3]; F2 = sum(x -> x * x, Fv)
    # H commutes with F² (rotational invariance): diagonalize F² first, then H in each F block (within subspace if given)
    P = subspace === nothing ? Matrix{ComplexF64}(I, D, D) : ComplexF64.(subspace)
    ep = eigen(Hermitian(P * P')); Q = ep.vectors[:, ep.values .> 0.5]   # orthonormal basis of the range of P (unpivoted QR does not span the range)
    f2 = eigen(Hermitian(Q' * F2 * Q))
    Fq = round.(Int, 2 .* (sqrt.(1 .+ 4 .* f2.values) .- 1) ./ 2) .// 2
    levels = Tuple{Float64,Rational{Int}}[]
    for F in sort(unique(Fq))
        V = Q * f2.vectors[:, Fq .== F]
        for e in eigvals(Hermitian(V' * H * V)); push!(levels, (e, F)); end
    end
    sort!(levels)
    (levels=levels, A=A, B=B, C=C, invariants=Dict(string(describe_str(o)) => val(o) for o in ops), H=H, qvals=q, δvals=δ)
end
describe_str(o) = sprint(describe, o)

"""
    araki_sucher(sys, wf; N=50, logvar=LogVariable()) -> BigFloat

Araki–Sucher ⟨1/r₁₂³⟩ (lim_{a→0} ⟨r₁₂⁻³ Θ(r₁₂ − a) + 4π(γ + ln a) δ³(r₁₂)⟩, arXiv:2301.05442 Eq. 7) in the regularized form
(Eq. 13): quadrature part (`ArakiSucherPart`) + 4π(1+γ)⟨δ³(r₁₂)⟩ (closed form). Pair of particles 1, 2 with internal particle 3 of infinite mass.
"""
function araki_sucher(sys::ThreeBody, wf; N::Int=50, logvar::LogVariable=LogVariable(), T::Union{Type,Nothing}=nothing, delta12::Symbol=:drachman)
    pt_order = hasproperty(wf, :order) ? wf.order : (1, 2, 3)
    (pt_order[3] == 3 && isinf_mass(sys.masses[3])) || throw(ArgumentError("araki_sucher requires input particle 3 of infinite mass (no reordering)"))
    q, dr = expectation_quad(sys, wf, [ArakiSucherPart(), DrachmanDelta12()]; N=N, logvar=logvar, T=T)
    fourpi_d12 = _delta12_times_4pi(sys, wf, dr, delta12, logvar)
    q + (1 + BigFloat(Base.MathConstants.eulergamma)) * fourpi_d12
end

# 4π⟨δ³(r₁₂)⟩: :drachman is the global form (quadrature), :closed the closed form on the coalescence line (`Delta`); the FC wave function
# does not satisfy the electron–electron cusp exactly, so Drachman's form is more accurate.
function _delta12_times_4pi(sys, wf, dr, delta12::Symbol, logvar)
    delta12 === :drachman && return dr
    delta12 === :closed && return 4 * BigFloat(Base.MathConstants.pi) * expectation(sys, wf, [Delta(1, 2)]; logvar=logvar)[1]
    throw(ArgumentError("delta12 must be :drachman or :closed"))
end

"""
    qed_correction(sys, wf; lnk0, α=7.2973525693e-3, N=50, logvar=LogVariable()) -> NamedTuple

Leading QED term of singlet S states of two-electron atoms (arXiv:2301.05442 Eq. 6, infinite nuclear mass):
δE_QED/α³ = (4Z/3)(19/30 − 2 ln α − ln k₀)⟨δ(r₁)+δ(r₂)⟩ + (164/15 + (14/3) ln α)⟨δ(r₁₂)⟩ − (7/6π)⟨1/r₁₂³⟩_AS.
The Bethe logarithm ln k₀ is an input. Electron–nucleus δ in closed form; electron–electron δ and AS in Drachman's form (`araki_sucher`).
"""
function qed_correction(sys::ThreeBody, wf; lnk0, α=big"7.2973525693e-3", N::Int=50, logvar::LogVariable=LogVariable(),
                        T::Union{Type,Nothing}=nothing)
    (sys.masses[1] == "1" && sys.masses[2] == "1" && isinf_mass(sys.masses[3])) ||
        throw(ArgumentError("qed_correction requires a two-electron atom (1, 1, ∞) in this input order"))
    q = Dict(t.pow => exact(BigFloat, t.c) for t in sys.terms if t isa PowerTerm)
    # wrapper of the general qed_leading (the same formula for He with infinite mass). −(7/6π)⟨1/r₁₂³⟩_AS = −(14/3)Q (AS = 4πQ)
    r = qed_leading(sys, wf; bethe=lnk0, α=α, N=N, logvar=logvar, T=T)
    (EQED=r.E_over_α3, araki_sucher=4 * BigFloat(Base.MathConstants.pi) * r.Q[(1, 2)], delta12=r.delta_drachman[(1, 2)],
     delta_nuc=(r.delta[1], r.delta[2]), Z=-q[(-1, 0, 0)], lnk0=BigFloat(lnk0), α=BigFloat(α))
end
