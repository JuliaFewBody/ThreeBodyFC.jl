# Nonrelativistic Bethe logarithm (Korobov arXiv:1203.1404).
#
#   β = N/D, J = w₁∇₁ + w₂∇₂ (current in internal coordinates; wᵢ = zᵢ/mᵢ − z₃/m₃ for the current of all particles), D = ⟨[J,[H,J]]⟩/2
#   N = ∫₀^{Eh} k[J(k) + ⟨J²⟩/k] dk + ∫_{Eh}^∞ (dk/k)⟨J (E₀−H)²/(E₀−H−k) J⟩, J(k) = ⟨J (E₀−H−k)⁻¹ J⟩
#
# Steps: (1) diagonalize H in a basis of intermediate states (parity opposite to the initial state, L₀ ± 1 and L₀) and build the
#   spectrum (x_m, A_m) from velocity-form ⟨ψ₀|J|m⟩. Angles are CG-coupled invariants (Angular.coupled_weight, Wigner–Eckart),
#   reducing to scalar element integrals times polynomials in r⃗ᵢ·r⃗ⱼ.
#   (2) D in closed form (⟨δ⟩); J(k) of the basis for k ≤ Λ, the asymptotic form for k > Λ:
#   k²Φ − D ≈ a/√k + (b ln k + c)/k + … (Φ = J + ⟨J²⟩/k; a, b fixed by ⟨δ⟩, c and beyond fitted in a window).

"""
    short_range_sets(base, slot; ζ0=4.0, τ=2.0, n=12) -> Vector{NTuple{3,Float64}}

Short-range sets of intermediate states (for high k, 1203.1404 §III.A): replace entry `slot` (1: r₁, 2: r₂, 3: r₁₂) of `base` = (α, β, γ) by ζ0·τ^k (k = 0..n−1).
"""
short_range_sets(base::NTuple{3,<:Real}, slot::Int; ζ0::Real=4.0, τ::Real=2.0, n::Int=12) =
    [Base.setindex(Float64.(base), Float64(ζ0 * τ^k), slot) for k in 0:n-1]

"""
    bethe_intermediate(sys, es; n, L=1, unnatural=false, symmetry=NoSymmetry(), seed_components=nothing, short_range=[], pow=1, cpow=0, T=BigFloat) -> NamedTuple

Intermediate-state basis of the Bethe logarithm (one channel L and parity): the raw basis of `run_fc(sys, es; L, unnatural, stages=1:n)`
plus, for each short-range set in `short_range` (exponent triples), functions r₁^A r₂^B (A, B ≤ `pow`) of all components. Returns
(basis, α, β, γ, L, unnatural). `cpow` > 0 also adds r₁₂^C (C ≤ cpow) (needed from L₀ ≥ 1 initial states to span r⃗₁·r⃗₂ in the S
channel). From L₀ there are three channels: L = L₀ ± 1 (opposite parity) and L = L₀ (L₀ ≥ 1, unnatural ↔ natural).
"""
function bethe_intermediate(sys::ThreeBody, es::ExponentSets; n::Int, L::Int=1, unnatural::Bool=false, symmetry::Symmetry=NoSymmetry(),
                            seed_components=nothing, short_range=NTuple{3,Float64}[], pow::Int=1, cpow::Int=0, T::Type=BigFloat)
    d = mktempdir()
    run_fc(sys, es; stages=1:n, symmetry=symmetry, L=L, unnatural=unnatural, seed_components=seed_components, T=T, auto_precision=false,
           outdir=d, io=devnull)
    w = load_wavefunction(d, n)
    rm(d; recursive=true, force=true)
    get(w, :order, (1, 2, 3)) == (1, 2, 3) || throw(ArgumentError("particles of the intermediate states were reordered (order=$(w.order)); put the identical pair at input 1, 2"))
    α, β, γ, B = copy(w.α), copy(w.β), copy(w.γ), copy(w.basis)
    for (x, y, z) in short_range
        push!(α, x); push!(β, y); push!(γ, z); k = length(α)
        for A in 0:pow, Bp in 0:pow, C in 0:cpow, a in 0:L-unnatural
            push!(B, (A, Bp, C, 0, 0, k, a))
        end
    end
    (basis=B, α=α, β=β, γ=γ, L=L, unnatural=unnatural)
end

# radial part of J = w₁∇₁ + w₂∇₂ on one term (key) of ψ₀: coefficients of r⃗₁, r⃗₂ in ∇₁f = r⃗₁ f₁/r₁ + r⃗₁₂ f₁₂/r₁₂,
# ∇₂f = r⃗₂ f₂/r₂ − r⃗₁₂ f₁₂/r₁₂, and the coefficients f of w₁∇₁ + w₂∇₂ acting on the angular part 𝒴 (kinds 1..4 = r⃗₁, r⃗₂, ∇₁, ∇₂)
function _bethe_jvec(key, α::T, β::T, γ::T, βσ::T, w1::T, w2::T) where {T}
    A, B, C, d, j = key
    P = Poly{T}((A, B, C, d, j) => one(T))
    D1 = _d1(P, α); D2 = _d2(P, β); D12 = _d12(P, γ, βσ)
    X1 = _shift(D1, -1, 0, 0, 0, w1); _addall!(X1, _shift(D12, 0, 0, -1, 0, w1 - w2))
    X2 = _shift(D2, 0, -1, 0, 0, w2); _addall!(X2, _shift(D12, 0, 0, -1, 0, w2 - w1))
    (X1, X2, _shift(P, 0, 0, 0, 0, w1), _shift(P, 0, 0, 0, 0, w2))
end

"""
    bethe_spectrum(sys, wf0, inter; weights=(1, 1), T=BigFloat, logvar=LogVariable(), cut=60) -> NamedTuple

Diagonalize H in the intermediate states `inter` (result of `bethe_intermediate`, or a vector of them for several channels) and
return x_m = E_m − E₀ and A_m = Σ_{M,M'} |⟨ψ₀ M|J|m M'⟩|²/(2L₀+1) (averaged over the initial M, summed over M' and vector
components), with J = w₁∇₁ + w₂∇₂ (`weights`). Angular parts are CG-coupled invariants (`Angular.coupled_weight`), reducing to
scalar element integrals. Intermediate states below E₀ are kept (x < 0; `bethe_log` takes the principal value). Canonical
orthogonalization of S drops eigenvalues below 10^−cut of the largest; a warning is printed if cut exceeds working digits − 20
(rounding is amplified to 10^(cut−digits)). Stops if exponents of ψ₀ or the intermediate states are not binary (Float64 rounding
of exponent sums). Returns (x, A, D_basis, trk, kept, M, below, channels); `trk` = Σ A/x (oscillator-strength sum, 3 for He with
J = ∇₁+∇₂). ψ₀ in input particle order, without split logarithms or Gaussians.
"""
function bethe_spectrum(sys::ThreeBody, wf0, inters::AbstractVector; kw...)
    sp = [bethe_spectrum(sys, wf0, it; kw...) for it in inters]
    x = reduce(vcat, [s.x for s in sp]); A = reduce(vcat, [s.A for s in sp])
    (x=x, A=A, D_basis=sum(A .* x), trk=sum(A ./ x), kept=sum(s.kept for s in sp), M=sum(s.M for s in sp), below=sum(s.below for s in sp),
     channels=[(L=s.channels[1].L, unnatural=s.channels[1].unnatural, trk=s.trk, M=s.M) for s in sp])
end
function bethe_spectrum(sys::ThreeBody, wf0, inter; weights=(1, 1), T::Type=BigFloat, logvar::LogVariable=LogVariable(), cut::Int=60)
    L0 = get(wf0, :L, 0); u0 = get(wf0, :unnatural, false); Lp = get(inter, :L, 1); up = get(inter, :unnatural, false)
    abs(L0 - Lp) <= 1 <= L0 + Lp || throw(ArgumentError("dipole selection rule: no transition L₀=$(L0) → L=$(Lp)"))
    isodd(L0 + u0 + Lp + up) || throw(ArgumentError("intermediate parity equals the initial one (L₀=$(L0) unnatural=$(u0), L=$(Lp) unnatural=$(up))"))
    get(wf0, :order, (1, 2, 3)) == (1, 2, 3) || throw(ArgumentError("particles of ψ₀ are reordered (order=$(wf0.order))"))
    logvar.split && throw(ArgumentError("split log variables are not supported"))
    isone(logvar.w2) || throw(ArgumentError("w₂ ≠ 1 is not supported"))
    (any(!iszero, get(wf0, :η, [0.0])) ) && throw(ArgumentError("the r₁₂ Gaussian is not supported"))
    _check_binary_sums(wf0, inter)
    # the smaller the kept λ, the more the rounding eps is amplified in XᵀHX (eps/λ); keeping eps/λ ≤ 1e-20 bounds cut by digits − 20
    # (cut changes the subspace, so it is not changed automatically; only a warning)
    nd = floor(Int, T === BigFloat ? precision(BigFloat) * log10(2) : -log10(Float64(eps(T))))
    cut > nd - 20 && @warn "bethe_spectrum: cut=$cut is large for $nd working digits (rounding amplified to 10^(cut−$nd)); use cut ≤ $(nd - 20) or more precision"
    ctx = IntegralContext(T, KernelQuadrature(T), logvar)
    βσ = T(logvar.βσ); w1, w2 = T(weights[1]), T(weights[2])
    X0 = [_bethe_jvec(k, T(wf0.α[k[6]]), T(wf0.β[k[6]]), T(wf0.γ[k[6]]), βσ, w1, w2) for k in wf0.basis]
    CW = Dict((a, c, kd) => [(p, Complex{T}(w)) for (p, w) in Angular.coupled_weight(L0, a, Lp, c, kd; unnatural0=u0, unnaturalp=up)]
              for a in 0:L0-u0, c in 0:Lp-up, kd in 1:4)
    c0 = T.(wf0.c)
    M1 = length(inter.basis)
    # collect the needed element keys without duplicates and compute them with the assembly machinery (kernel prefetch, heaviest
    # first in parallel), then sum (computing per row repeats the same elements and waits on the same kernels)
    eb(x::Float64) = reinterpret(Int64, x)
    function terms(f, i)
        g = inter.basis[i]; c = g[7]; kk = g[6]
        for (n, k0) in enumerate(wf0.basis)
            ea, eb_, eg = wf0.α[k0[6]] + inter.α[kk], wf0.β[k0[6]] + inter.β[kk], wf0.γ[k0[6]] + inter.γ[kk]
            for kd in 1:4, ((A, B, C, d, j), x) in X0[n][kd], ((p, q, r), w) in CW[(k0[7], c, kd)]
                f(n, x, w, (A + g[1] + p + 1, B + g[2] + q + 1, C + g[3] + r + 1, d + g[4], j + g[5], eb(ea), eb(eb_), eb(eg)))
            end
        end
    end
    nt = Threads.nthreads()
    loc = [Set{NTuple{8,Int}}() for _ in 1:nt]
    @sync for (t, rr) in enumerate(Iterators.partition(1:M1, cld(M1, nt)))
        Threads.@spawn (local st = loc[t]; for i in rr; terms((n, x, w, k) -> push!(st, k), i); end)
    end
    ekeys = sort!(collect(reduce(union!, loc)))
    kid = Dict(k => i for (i, k) in enumerate(ekeys))
    tab = ElementTable{T}(Matrix{Int32}(undef, 0, 0), ekeys, Vector{T}(undef, length(ekeys)), Int32[], Vector{Tuple{Int32,T}}[])
    _compute_elements!(tab, ctx, nt, 5_000_000)
    dvec = zeros(Complex{T}, M1)
    Threads.@threads :dynamic for i in 1:M1
        s = [zero(Complex{T}) for _ in eachindex(c0)]
        terms((n, x, w, k) -> (s[n] += x * w * tab.vals[kid[k]]), i)
        dvec[i] = sum(c0[n] * s[n] for n in eachindex(c0))
    end
    _, S0 = assemble(HamiltonianData(sys, logvar, T; L=L0, unnatural=u0), ctx, ExponentSets(wf0.α, wf0.β, wf0.γ), wf0.basis)
    N0 = c0' * S0 * c0
    H1, S1 = assemble(HamiltonianData(sys, logvar, T; L=Lp, unnatural=up), ctx, ExponentSets(inter.α, inter.β, inter.γ), inter.basis)
    Sc = Matrix{T}(S1); Hc = Matrix{T}(H1)
    ev = eigen_mt(Symmetric(Sc))
    keep = ev.values .> maximum(ev.values) * T(10)^(-cut)
    X = ev.vectors[:, keep] ./ sqrt.(ev.values[keep])'
    # A_m = |u_mᵀ d|² (u_m = X e_m) needs only the eigenvalues and eᵀ(Xᵀd): neither the eigenvector matrix nor U = X·E is formed
    eh = eigen_proj(Symmetric(xtHx_mt(X, Hc)), hcat(X' * real.(dvec), X' * imag.(dvec)))
    A = (abs2.(eh.P[:, 1]) .+ abs2.(eh.P[:, 2])) ./ N0
    x = eh.values .- T(wf0.E)
    below = count(<(0), x)
    (x=x, A=A, D_basis=sum(A .* x), trk=sum(A ./ x), kept=count(keep), M=M1, below=below,
     channels=[(L=Lp, unnatural=up, trk=sum(A ./ x), M=M1)])
end

# coefficients of 1/r (charge products q₁₃, q₂₃, q₁₂)
function _coulomb_charges(sys::ThreeBody, ::Type{T}) where {T}
    q = Dict(t.pow => exact(T, t.c) for t in sys.terms if t isa PowerTerm)
    (get(q, (-1, 0, 0), zero(T)), get(q, (0, -1, 0), zero(T)), get(q, (0, 0, -1), zero(T)))
end

"""
    bethe_log(spec, sys, wf0; weights=(1, 1), window=(50, 2000), mode=:closed, Z=1, μ=1, structure=2, T=BigFloat) -> NamedTuple

Bethe logarithm (R∞ units, ln(k₀/R∞)) from the spectrum of `bethe_spectrum`; `weights` as for the spectrum.
- `mode=:closed`: D = −2π[w₁²q₁₃⟨δ₁⟩ + w₂²q₂₃⟨δ₂⟩ + (w₁−w₂)²q₁₂⟨δ₁₂⟩] (closed-form δ) and the asymptotic leading terms
  a = −√2 Σ q²Δw²√μ_ij 4π⟨δ_ij⟩, b = −Σ q³Δw² μ_ij 4π⟨δ_ij⟩ (Δw = wᵢ for pair (i,3), w₁ − w₂ for (1,2); 1203.1404 Eq. 21).
- `mode=:point`: a point pseudo-nucleus (charge `Z`, reduced mass `μ`) for a composite nucleus smaller than the electron Compton
  wavelength (the electron in μHe). D_pt is fitted with leading terms a = −2Z√(2μ)D_pt, b = 2Z²μD_pt; `structure` (0–2) structure
  terms λ₁√k, λ₂k are added to k²Φ in the fit and removed from β of the point pseudo-nucleus.
The fit uses the window `window = (k_min, k_max)` and the integral is split at Λ = k_min. Returns (β, D, coef, residual).
"""
function bethe_log(spec, sys::ThreeBody, wf0; weights=(1, 1), window=(50, 2000), mode::Symbol=:closed, Z=1, μ=1, structure::Int=2,
                   T::Type=BigFloat, logvar::LogVariable=LogVariable())
    A, x = spec.A, spec.x
    kPhi(k) = sum(A .* x ./ (x .+ k))
    int_kPhi(Λ) = sum(A .* x .* log.(abs.((x .+ Λ) ./ x)))          # principal value for x < 0 (below E₀)
    ks = exp.(range(log(T(window[1])), log(T(window[2])); length=40))
    y = [k * kPhi(k) for k in ks]                         # k²Φ
    base = hcat(1 ./ ks, ks .^ -1.5, ks .^ -2, log.(ks) ./ ks .^ 2, ks .^ -2.5)
    if mode === :closed
        δ = expectation(sys, wf0, [Delta(1, 3), Delta(2, 3), Delta(1, 2)]; logvar=logvar)
        q = _coulomb_charges(sys, T); w1, w2 = T(weights[1]), T(weights[2])
        i1, i2, i3 = inverse_masses(sys, T); m1, m2 = exact(T, sys.masses[1]), exact(T, sys.masses[2])
        μs = (1 / i1, 1 / i2, 1 / (1 / m1 + 1 / m2))
        Δw = (w1, w2, w1 - w2)
        D = -2T(π) * sum(Δw[p]^2 * q[p] * T(δ[p]) for p in 1:3)
        a0 = -sqrt(T(2)) * sum(q[p]^2 * Δw[p]^2 * sqrt(μs[p]) * 4T(π) * T(δ[p]) for p in 1:3)
        b0 = -sum(q[p]^3 * Δw[p]^2 * μs[p] * 4T(π) * T(δ[p]) for p in 1:3)
        co = base \ (y .- D .- a0 ./ sqrt.(ks) .- b0 .* log.(ks) ./ ks)
        res = maximum(abs.(base * co .- (y .- D .- a0 ./ sqrt.(ks) .- b0 .* log.(ks) ./ ks))) / maximum(abs.(y))
        λ1 = λ2 = zero(T)
    elseif mode === :point
        Zt, μt = T(Z), T(μ)
        g0 = [1 - 2Zt * sqrt(2μt) / sqrt(k) + 2Zt^2 * μt * log(k) / k for k in ks]
        Mx = hcat(g0, base)
        structure >= 1 && (Mx = hcat(Mx, sqrt.(ks))); structure >= 2 && (Mx = hcat(Mx, ks))
        cc = Mx \ y; res = maximum(abs.(Mx * cc .- y)) / maximum(abs.(y))
        D = cc[1]; co = cc[2:6]
        a0 = -2Zt * sqrt(2μt) * D; b0 = 2Zt^2 * μt * D
        λ1 = structure >= 1 ? cc[7] : zero(T); λ2 = structure >= 2 ? cc[8] : zero(T)
    else
        throw(ArgumentError("mode must be :closed or :point"))
    end
    c, d, e, g, h = co
    Λ = T(window[1])
    tail = 2a0 / sqrt(Λ) + b0 * (log(Λ) + 1) / Λ + c / Λ + 2d / (3Λ^T(1.5)) + e / (2Λ^2) + g * (2log(Λ) + 1) / (4Λ^2) + 2h / (5Λ^T(2.5))
    N = int_kPhi(Λ) - 2λ1 * sqrt(Λ) - λ2 * Λ - D * log(Λ) + tail
    # N_R∞ = Dβ (numerator in R∞ units). For initial states with l ≥ 1, D (∝ ⟨δ⟩) is nearly 0 and β = N/D is meaningless; then use
    # N_R∞ in the energy (−(2/3π)Dβ = −(2/3π)N_R∞ in qed_leading) or ln k₀ normalized by the hydrogenic 2Z⁴μ³/n³
    (β=-N / D + log(T(2)), D=D, N=-N + D * log(T(2)), coef=(a=a0, b=b0, c=c, λ1=λ1, λ2=λ2), residual=res)
end

# charges z₁, z₂, z₃ from the Coulomb coefficients q_ij = zᵢzⱼ (the overall sign is the sign of J and does not affect β)
function _charges_from_coulomb(sys::ThreeBody, ::Type{T}) where {T}
    q13, q23, q12 = _coulomb_charges(sys, T)
    (iszero(q13) || iszero(q23) || iszero(q12)) && throw(ArgumentError("a Coulomb pair is missing, so charges are undetermined; pass weights"))
    z1 = sqrt(q13 * q12 / q23)
    (z1, q12 / z1, q13 / z1)
end

# internal-coordinate weights of the current of all particles, wᵢ = zᵢ/mᵢ − z₃/m₃
function _full_current_weights(sys::ThreeBody, ::Type{T}) where {T}
    z = _charges_from_coulomb(sys, T)
    i1, i2, i3 = inverse_masses(sys, T)          # (1/μ₁₃, 1/μ₂₃, 1/m₃)
    (z[1] * (i1 - i3) - z[3] * i3, z[2] * (i2 - i3) - z[3] * i3)
end

# exchange sign of ψ₀ (physical sign, undoing the −1 mirror factor of unnatural parity); nothing if particles 1, 2 are not identical
function _exchange_sign(sys::ThreeBody, wf0)
    sys.masses[1] == sys.masses[2] || return nothing
    es = ExponentSets(wf0.α, wf0.β, wf0.γ)
    L = get(wf0, :L, 0); un = get(wf0, :unnatural, false)
    idx = Dict(k => i for (i, k) in enumerate(wf0.basis))
    i = argmax(abs.(wf0.c)); k = wf0.basis[i]
    j = get(idx, _mirror_key(k, es, L - un), 0)
    j == 0 && return nothing
    r = wf0.c[j] / wf0.c[i]
    abs(abs(r) - 1) < 1e-6 || return nothing
    Int(sign(r)) * (un ? -1 : 1)
end

# channels (L, unnatural) reachable by dipole transitions
function _bethe_channels(L0::Int, un0::Bool)
    ch = Tuple{Int,Bool}[]
    if !un0
        L0 >= 1 && push!(ch, (L0 - 1, false)); L0 >= 1 && push!(ch, (L0, true)); push!(ch, (L0 + 1, false))
    else
        L0 >= 2 && push!(ch, (L0 - 1, true)); push!(ch, (L0, false)); push!(ch, (L0 + 1, true))
    end
    ch
end

# Default short-range sets: for each particle s with nonzero current weight, slot s takes ζ = 2ζ_max·τᵏ (ζ_max the largest exponent of
# ψ₀ in slot s), and a spectator particle o takes its asymptotic decay κ_o = √(2μ_o(E_th(o) − E₀)) (E_th(o) the threshold for o to
# leave, μ_o = 1/(1/m_o + 1/m₃)). The pair (1,2) (w₁ ≠ w₂) uses the r₁₂ slot with ζ = 2·min(κ₁, κ₂)·τᵏ on the base (κ₁, κ₂, 0).
# κ is put on the 2^-20 grid (exact Float64 sums with the exponents of ψ₀, `_check_binary_sums`).
function _default_short_range(sys::ThreeBody, wf0, w; nsr::Int, τ::Real)
    sets = unique(zip(Float64.(wf0.α), Float64.(wf0.β), Float64.(wf0.γ)))
    th = thresholds(sys)
    i = inverse_masses(sys, Float64)
    function κ(o)
        c = [ch for ch in th.channels if ch.leaves == o]
        isempty(c) && return minimum(t[o] for t in sets)
        dE = minimum(ch.threshold for ch in c) - Float64(wf0.E)
        dE > 0 ? _dyadic(sqrt(2dE / i[o])) : minimum(t[o] for t in sets)
    end
    sr = NTuple{3,Float64}[]
    for s in 1:3
        (s <= 2 ? !iszero(w[s]) : !iszero(w[1] - w[2])) || continue
        smax = maximum(t[s] for t in sets)
        base = s == 1 ? (smax, κ(2), 0.0) : s == 2 ? (κ(1), smax, 0.0) : (κ(1), κ(2), 0.0)
        ζ0 = s <= 2 ? 2smax : _dyadic(2min(κ(1), κ(2)))
        append!(sr, short_range_sets(base, s; ζ0=ζ0, τ=τ, n=nsr))
    end
    sr
end

# Keys of ψ₀ × intermediate (d) and intermediate × intermediate (S₁, H₁) elements are Float64 sums of exponents. Non-binary exponents
# round those sums by 1e-16, which near-dependent directions amplify by 1/√λ into spurious states and a broken TRK sum independent of
# the working precision. Rounding the exponents changes the function space, so this stops instead.
function _check_binary_sums(wf0, inter)
    es = ExponentSets([Float64.(wf0.α); Float64.(inter.α)], [Float64.(wf0.β); Float64.(inter.β)], [Float64.(wf0.γ); Float64.(inter.γ)])
    msg = binary_warning(es)
    msg === nothing && return nothing
    throw(ArgumentError("Bethe logarithm: " * msg * " in ψ₀ or the intermediate states; rebuild ψ₀ with `binary_exponents` sets"))
end

"""
    bethe_logarithm(sys, wf0; weights=:full, n1=3, es1=nothing, symmetry=:auto, nsr=12, τ=2, pow=2, cpow=:auto, window=:auto, cut=60, T=BigFloat, io=devnull)

Bethe logarithm with defaults (channels, intermediate bases, short-range sets and window chosen automatically); the individual steps are
`bethe_intermediate`, `bethe_spectrum`, `bethe_log`.
- `weights = :full`: the current of all particles wᵢ = zᵢ/mᵢ − z₃/m₃ (charges from the Coulomb coefficients); e.g. (1, 0) per particle.
- Intermediate states: FC with the exponent sets of ψ₀ for n1 stages per channel (same exchange sign as ψ₀) plus short-range sets
  (ζ = 2ζ_max·τᵏ; spectators with the threshold decay κ).
- `cpow = :auto`: 2 for all channels if L₀ ≥ 1, 0 if L₀ = 0 (³P needs r₁₂², ¹P needs r₁², r₂² with pow=2).
- `window = :auto`: (12.5, 500)·s with s = max_ij q_ij² μ_ij over pairs carrying current; β is also computed for the window ×½ and ×2 and the spread returned.
`mode`, `Z`, `μ`, `structure` go to `bethe_log` (default window (12.5, 500)·Z²μ for `:point`).
Returns (β, spread, betas, N, Nspread, Ns, D, trk, below, channels, spec); N = Dβ (R∞ units) is used for l ≥ 1 when D ≈ 0.
"""
function bethe_logarithm(sys::ThreeBody, wf0; weights=:full, n1::Int=3, es1=nothing, symmetry=:auto, nsr::Int=12, τ::Real=2.0, pow::Int=2,
                         cpow=:auto, window=:auto, cut::Int=60, mode::Symbol=:closed, Z=1, μ=1, structure::Int=2, T::Type=BigFloat, io::IO=devnull)
    w = weights === :full ? _full_current_weights(sys, T) : (T(weights[1]), T(weights[2]))
    L0 = get(wf0, :L, 0); un0 = get(wf0, :unnatural, false)
    sym = if symmetry === :auto
        s = iszero(w[1] - w[2]) ? _exchange_sign(sys, wf0) : nothing
        s === nothing ? NoSymmetry() : Exchange(s)
    else
        symmetry
    end
    es = es1
    if es1 === nothing
        sets = unique(zip(wf0.α, wf0.β, wf0.γ))
        es = ExponentSets([Float64(t[1]) for t in sets], [Float64(t[2]) for t in sets], [Float64(t[3]) for t in sets])
    end
    sr = _default_short_range(sys, wf0, w; nsr=nsr, τ=τ)
    specs = []
    for (L, un) in _bethe_channels(L0, un0)
        t = @elapsed begin
            cp = cpow === :auto ? (L0 >= 1 ? 2 : 0) : cpow          # r₁₂² also matters in the D and Pᵉ channels for triplet P states
            it = bethe_intermediate(sys, es; n=n1, L=L, unnatural=un, symmetry=sym, short_range=sr, pow=pow, cpow=cp, T=T)
            sp = bethe_spectrum(sys, wf0, it; weights=w, T=T, cut=cut)
        end
        push!(specs, sp)
        println(io, "channel L=$L$(un ? " (unnatural)" : "") M=$(sp.M) kept $(sp.kept) below E₀ $(sp.below) TRK $(fmt(sp.trk, 12)) ($(round(t)) s)"); flush(io)
    end
    spec = (x=reduce(vcat, [s.x for s in specs]), A=reduce(vcat, [s.A for s in specs]))
    win = if window === :auto && mode === :point
        s = T(Z)^2 * T(μ)                         # scale of the point pseudo-nucleus (Z=1, μ = μ_e for the electron in μHe)
        (Float64(12.5s), Float64(500s))
    elseif window === :auto
        q = _coulomb_charges(sys, T); i1, i2, i3 = inverse_masses(sys, T)
        μs = (1 / i1, 1 / i2, 1 / (i1 + i2 - 2i3)); Δw = (w[1], w[2], w[1] - w[2])
        s = maximum(iszero(Δw[p]) || iszero(q[p]) ? zero(T) : q[p]^2 * μs[p] for p in 1:3)
        (Float64(12.5s), Float64(500s))
    else
        window
    end
    wins = [(win[1] / 2, win[2] / 2), win, (2win[1], 2win[2])]
    rs = [bethe_log(spec, sys, wf0; weights=w, window=v, T=T, mode=mode, Z=Z, μ=μ, structure=structure) for v in wins]
    βs = [r.β for r in rs]; Ns = [r.N for r in rs]
    (β=βs[2], spread=maximum(βs) - minimum(βs), betas=βs, N=Ns[2], Nspread=maximum(Ns) - minimum(Ns), Ns=Ns, D=rs[2].D, trk=sum(spec.A ./ spec.x), below=sum(s.below for s in specs),
     channels=[(L=s.channels[1].L, unnatural=s.channels[1].unnatural, trk=s.trk, M=s.M) for s in specs], spec=spec, window=win)
end
