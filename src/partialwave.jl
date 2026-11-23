# Exchange-mirror element pairs for L ≥ 1 by partial waves.
#
# In coordinates (r₁, r₂, x = cos θ₁₂), with W_ac = Σ_l w_l(r₁, r₂) P_l(x) (`Angular.legendre_components`) and Sack's r₁₂^C = Σ_l R_{C,l}(r_<, r_>) P_l(x) (odd C ≥ −1),
#   ∫ r₁^{A₀} r₂^{B₀} r₁₂^{C} σ^D (ln σ)^J W e^{−a r₁ − b r₂} r₁²r₂² dr₁ dr₂ dx = Σ_l 2/(2l+1) ∫∫ r₁^{A₀+2} r₂^{B₀+2} w_l R_{C,l} σ^D (ln σ)^J e^{…} dr₁ dr₂.
# Valid when σ = r₁ + w₂ r₂ does not depend on x (βσ = 0, not split) and the r₁₂ exponents sum to 0. With r₁ = t r₂ for r₁ < r₂
# (σ = r₂ (t + w₂)) and r₂ = t r₁ for r₁ > r₂ (σ = r₁ (1 + w₂ t)), the outer radial integral is closed form,
#   ∫₀^∞ r^M (ln r + λ)^J e^{−κ r} dr = Σ_j C(J,j) λ^{J−j} ∂_M^j [M! κ^{−M−1}]   (J ≤ 2),
# and t ∈ [0, 1] uses Gauss–Legendre in t or in the Möbius variable u = α t/(α t + β), per side. The monomial path (expanding W in
# powers of r₁₂²) adds (L+1)(L+2)/2 cancelling terms for mirror pairs and loses digits at large L.

const _SACK = Dict{NTuple{2,Int},Vector{Tuple{Int,Rational{BigInt}}}}()
const _PWNODES = Dict{Tuple{DataType,Int,Int},Tuple{Vector,Vector}}()
const _PWLOCK = ReentrantLock()
const _PW_ENABLED = Ref(true)          # false: monomial path (for checks)

_poch(x, n) = prod((x + i for i in 0:n-1); init=one(x))
_tq(::Type{T}, r::Rational{BigInt}) where {T} = T(_big(() -> BigFloat(numerator(r)) / BigFloat(denominator(r))))

"""Legendre components R_{C,l} = Σ co r_<^e r_>^{C−e} of r₁₂^C as (e, co). ₂F₁(l − C/2, −(C+1)/2; l + 3/2; t²) terminates by its second
argument for odd C ≥ −1 and by its first for even C ≥ 0 (even C gives 0 for l > C/2; r₁₂^C is a polynomial of degree C/2 in x)."""
function _sack(C::Int, l::Int)
    (C >= -1 && (isodd(C) || C >= 0)) || throw(ArgumentError("Sack's finite sum needs C ≥ −1 (C = $C)"))
    get!(_tls(:pwsack, () -> Dict{NTuple{2,Int},Vector{Tuple{Int,Rational{BigInt}}}}()), (C, l)) do      # per-task copy
        lock(_PWLOCK) do
            get!(_SACK, (C, l)) do
                m = isodd(C) ? (C + 1) ÷ 2 : C ÷ 2 - l
                pre = _poch(-big(C) // 2, l) // _poch(big(1) // 2, l)
                [(l + 2k, pre * _poch(big(l) - big(C) // 2, k) * _poch(-(big(C) + 1) // 2, k) // (_poch(big(l) + 3 // 2, k) * factorial(big(k)))) for k in 0:m]
            end
        end
    end
end

# radius of the Bernstein ellipse ([0,1] → [−1,1]) through a pole x₀ < 0 of the integrand
_bernstein(x0) = (z = 1 - 2x0; z + sqrt(z^2 - 1))

# Tables are read from per-task copies (task_local_storage); a per-table lock is taken only when building one
# (a single global lock stalled the threads during the first assembly at large L).
_tls(k::Symbol, f) = get!(f, task_local_storage(), k)
const _PWNLOCK = ReentrantLock()
# node counts are rounded to a ladder of multiples of 8 in steps of 2^{1/4} (fewer table variants, at most 19% more nodes)
function _pw_table(::Type{T}, nmin::Int) where {T}
    n = 8
    while n < nmin
        n = 8cld(ceil(Int, n * 2^(1 / 4)), 8)
    end
    key = (T, T === BigFloat ? precision(BigFloat) : 0, n)
    get!(_tls(:pwnodes, () -> Dict{Any,Any}()), key) do
        lock(_PWNLOCK) do
            get!(() -> Integrals.gauss_legendre01(T, n), _PWNODES, key)
        end
    end
end

# Nodes per term: the integrand t^p κ(t)^{−(M+1)} s(t)^D (κ = a t + b or a + b t, s = t + w₂ or 1 + w₂ t) has a sharp peak
# in [0, 1] for p, M ~ L, so nodes chosen from the pole position alone are not enough. The Gauss–Legendre error bound
# (64/15) F ρ'^{−2n}/(ρ'²−1) (F = max |f| on the ellipse E_ρ') divided by max |f| on [0, 1] is made ≤ 1e-10 eps of the
# working precision, minimizing n over ρ' inside the pole (no fitted constants).
function _pw_need(lne::Float64, p::Int, M::Int, D::Int, side::Int, a::Float64, b::Float64, w2::Float64)
    key = (p, M, D, side, lne, a, b, w2)
    get!(_tls(:pwneed, () -> Dict{Tuple{Int,Int,Int,Int,Float64,Float64,Float64,Float64},Int}()), key) do
        κ(t) = side == 1 ? a * t + b : a + b * t
        σ(t) = side == 1 ? t + w2 : 1 + w2 * t
        φ(t) = (p > 0 ? p * log(abs(t)) : 0.0) - (M + 1) * log(abs(κ(t))) + (D != 0 ? D * log(abs(σ(t))) : 0.0)
        φmax = maximum(φ(i / 64) for i in 1:64)
        ρp = _bernstein(side == 1 ? -b / a : -a / b)
        D < 0 && (ρp = min(ρp, _bernstein(side == 1 ? -w2 : -1 / w2)))
        best = Inf
        for j in 1:12
            ρ = 1 + (ρp - 1) * j / 13
            F = maximum(φ((1 + (ρ * cis(θ) + cis(-θ) / ρ) / 2) / 2) for θ in range(0, π; length=17))
            best = min(best, (F - φmax + log(64 / (15 * (ρ^2 - 1))) + lne) / (2log(ρ)))
        end
        ceil(Int, best)
    end
end

# per node: d × ∫₀^∞ r^M (ln r + λ)^J e^{−κ r} dr (J ≤ 2) = d M! κ^{−M−1} × {1, v, v² + ψ'(M+1)}, v = λ − ln κ + ψ(M+1).
# ik = κ^{−M−1}, dl = λ − ln κ (geometric tables). BigFloat uses in-place MPFR (no precision lookups or allocations)
function _pw_radvec(d::Vector{T}, ik::Vector{T}, dl::Vector{T}, fM::T, ψM::T, ψ1::T, J::Int, n::Int) where {T}
    J == 0 && return [d[i] * ik[i] * fM for i in 1:n]
    J == 1 && return [d[i] * ik[i] * fM * (dl[i] + ψM) for i in 1:n]
    [d[i] * ik[i] * fM * ((dl[i] + ψM)^2 + ψ1) for i in 1:n]
end
function _pw_radvec(d::Vector{BigFloat}, ik::Vector{BigFloat}, dl::Vector{BigFloat}, fM::BigFloat, ψM::BigFloat, ψ1::BigFloat, J::Int, n::Int)
    prec = precision(d[1]); out = Vector{BigFloat}(undef, n); v = BigFloat(; precision=prec)
    @inbounds for i in 1:n
        z = BigFloat(; precision=prec)
        HighPrecisionLinearAlgebra.mpfr_mul!(z, d[i], ik[i]); HighPrecisionLinearAlgebra.mpfr_mul!(z, z, fM)
        if J >= 1
            HighPrecisionLinearAlgebra.mpfr_add!(v, dl[i], ψM)
            J == 2 && (HighPrecisionLinearAlgebra.mpfr_fma!(v, v, v, ψ1))
            HighPrecisionLinearAlgebra.mpfr_mul!(z, z, v)
        end
        out[i] = z
    end
    out
end
_vmul(a::Vector{T}, b::Vector{T}) where {T} = a .* b
function _vmul(a::Vector{BigFloat}, b::Vector{BigFloat})
    prec = precision(a[1])
    [(z = BigFloat(; precision=prec); HighPrecisionLinearAlgebra.mpfr_mul!(z, a[i], b[i]); z) for i in eachindex(a)]
end

# Nodes per side: the integrand of a side is t^p (α t + β)^{−(M+1)} (γ t + δ)^D R_J (side 1: (α, β, γ, δ) = (a, b, 1, w₂),
# side 2: (b, a, w₂, 1)). Gauss–Legendre in t (:t) needs nodes ∝ L when the pole −β/α of κ is near 0 (α ≫ β). In the Möbius
# variable u = α t/(α t + β) (:u), t^p κ^{−(M+1)} dt becomes a polynomial weight in u and the node count does not grow with L.
# Each side uses the cheaper estimate. u nodes: t = β x/(β + α(1−x)), κ = β(α+β)/(β + α(1−x)), weight × dt/dx = w κ/(β + α(1−x))
# (x Gauss–Legendre on [0, 1], 1 − x from the symmetric nodes).
struct _PWSide{T}
    t::Vector{T}; t2::Vector{T}; κ::Vector{T}; σ::Vector{T}; w::Vector{T}
    dl::Vector{T}                                     # ln σ − ln κ (the log term of the closed radial form is (dl + ψ(M+1))^j)
    tpow::Vector{Vector{T}}                           # t^p (extended as needed)
    ik::Vector{Vector{T}}                             # κ^{−m} (extended by binary products; log₂ m roundings)
    dD::Dict{Int,Vector{T}}                           # D → weight × σ^D
end
function _pw_side(::Type{T}, kind::Symbol, nmin::Int, α::Float64, β::Float64, γ::Float64, δ::Float64) where {T}
    xs, ws = _pw_table(T, nmin); n = length(xs)
    key = (T, T === BigFloat ? precision(BigFloat) : 0, kind, n, α, β, γ, δ)
    get!(_tls(:pwside, () -> Dict{Any,Any}()), key) do              # per task (no lock)
        aT, bT, gT, dT = T(α), T(β), T(γ), T(δ)
        if kind === :t
            t = copy(xs); κ = aT .* t .+ bT; w = copy(ws)
        else
            den = [bT + aT * xs[n+1-i] for i in 1:n]              # β + α(1 − x) (1 − x_i = x_{n+1−i}; Gauss–Legendre nodes are symmetric)
            t = bT .* xs ./ den; κ = (bT * (aT + bT)) ./ den; w = ws .* κ ./ den
        end
        σ = gT .* t .+ dT
        _PWSide{T}(t, t .* t, κ, σ, w, log.(σ) .- log.(κ), [ones(T, n)], [ones(T, n), 1 ./ κ], Dict{Int,Vector{T}}())
    end::_PWSide{T}
end
function _pw_tpow(g::_PWSide{T}, p::Int) where {T}
    while length(g.tpow) <= p
        push!(g.tpow, _vmul(g.tpow[end], g.t))
    end
    g.tpow
end
function _pw_ikpow(g::_PWSide{T}, m::Int) where {T}
    while length(g.ik) <= m
        k = length(g.ik); h = k ÷ 2
        push!(g.ik, _vmul(g.ik[h+1], g.ik[k-h+1]))
    end
    g
end

# :t node count: the larger of _pw_need (peak and pole of κ) and the ellipse through this side's poles (−β/α, and −δ/γ if D ≠ 0 or J > 0)
function _pw_need_t(lne::Float64, p::Int, M::Int, D::Int, J::Int, α::Float64, β::Float64, γ::Float64, δ::Float64)
    ρ = min(_bernstein(-β / α), (D != 0 || J > 0) ? _bernstein(-δ / γ) : Inf)
    max(_pw_need(lne, p, M, D, 1, α, β, δ / γ), ceil(Int, lne / (2log(ρ))) + 8)
end
# :u node count: the same bound as _pw_need for F(u) = f(t(u)) t'(u) on u ∈ [0, X] (X = α/(α+β)). ρ is limited by u_s (σ = 0, if
# D ≠ 0 or J > 0) and u = 1 (pole for Q = M − 1 − p − D < 0), which are ignored when the integrand at the end X is below e^{−lne−20}
# of its maximum (the weight (1−u)^Q suppresses it); u_s < 0 always counts.
function _pw_need_u(lne::Float64, p::Int, M::Int, D::Int, J::Int, α::Float64, β::Float64, γ::Float64, δ::Float64)
    key = (p, M, D, J, lne, α, β, γ, δ)
    get!(_tls(:pwneedu, () -> Dict{NTuple{9,Any},Int}()), key) do
        X = α / (α + β)
        φ(u) = (t = (β / α) * u / (1 - u);
                (p > 0 ? p * log(abs(t)) : 0.0) - (M + 1) * log(abs(α * t + β)) + (D != 0 ? D * log(abs(γ * t + δ)) : 0.0) + log(abs((β / α) / (1 - u)^2)))
        φs = [φ(X * i / 256) for i in 1:255]; φmax = maximum(φs)
        edge = φs[end] > φmax - lne - 20
        us = (δ / γ) * α / (δ / γ * α - β)
        sing = Float64[]
        (D != 0 || J > 0) && isfinite(us) && (us < 0 || edge) && push!(sing, us)
        M - 1 - p - D < 0 && edge && push!(sing, 1.0)
        ρp = isempty(sing) ? 8.0 : min(8.0, minimum(s -> (z = abs(2s / X - 1); z + sqrt(z^2 - 1)), sing))
        best = Inf
        for j in 1:24
            ρ = 1 + (ρp - 1) * j / 25
            F = maximum(φ(X * (1 + (ρ * cis(θ) + cis(-θ) / ρ) / 2) / 2) for θ in range(0, π; length=33))
            best = min(best, (F - φmax + log(64 / (15 * (ρ^2 - 1))) + lne) / (2log(ρ)))
        end
        ceil(Int, best)
    end
end

"""
    pw_element(ctx, comps, A, B, C, D, J, a, b) -> T

Mirror-pair element Σ_l 2/(2l+1) ∫∫ r₁^A r₂^B w_l R_{C,l} σ^D (ln σ)^J e^{−a r₁ − b r₂} dr₁ dr₂ (A, B include the r₁², r₂² of the
volume element; C is the basis power of r₁₂). `comps` = `Angular.legendre_components(L, a, c)`. The t integral uses Gauss–Legendre in t
or u per side, whichever estimate is cheaper (`_pw_side`).
"""
function pw_element(ctx::IntegralContext{T}, comps, A::Int, B::Int, C::Int, D::Int, J::Int, a::Float64, b::Float64) where {T}
    J <= 2 || throw(ArgumentError("the partial-wave path needs powers of ln σ ≤ 2 (J = $J)"))
    w2 = Float64(ctx.cu)
    lne = -log(Float64(eps(T))) + 10log(10)
    geo = ((a, b, 1.0, w2), (b, a, w2, 1.0))           # side 1 (r₁ < r₂, κ = a t + b, σ/r₂ = t + w₂), side 2 (r₂ < r₁, κ = a + b t, σ/r₁ = 1 + w₂ t)
    nt = [8, 8]; nu = [8, 8]
    for (l, terms) in comps, (P, Q, _) in terms
        isempty(_sack(C, l)) && continue
        M = A + B + P + Q + C + D + 1; ex = 2length(_sack(C, l)) - 2
        for (s, p) in ((1, A + P + l + ex), (2, B + Q + l + ex))
            nt[s] = max(nt[s], _pw_need_t(lne, p, M, D, J, geo[s]...)); nu[s] = max(nu[s], _pw_need_u(lne, p, M, D, J, geo[s]...))
        end
    end
    g1, g2 = [nu[s] < nt[s] ? _pw_side(T, :u, nu[s], geo[s]...) : _pw_side(T, :t, nt[s], geo[s]...) for s in 1:2]
    n1, n2 = length(g1.t), length(g2.t)
    d1 = get!(() -> [g1.w[i] * g1.σ[i]^D for i in 1:n1], g1.dD, D)
    d2 = get!(() -> [g2.w[i] * g2.σ[i]^D for i in 1:n2], g2.dD, D)
    rad = Dict{Int,NTuple{2,Vector{T}}}()                     # M → per node d × closed radial form (side 1, side 2)
    radial(M) = get!(rad, M) do
        M >= 0 || throw(ArgumentError("negative radial power (M = $M)"))
        fM = M < length(ctx.fact) ? ctx.fact[M+1] : Integrals.factT(T, M)
        _pw_ikpow(g1, M + 1); _pw_ikpow(g2, M + 1)
        ψM = J >= 1 ? (M + 1 <= length(ctx.ψ) ? ctx.ψ[M+1] : Integrals.digamma_int(T, M + 1)) : zero(T)
        ψ1 = J >= 2 ? (M + 1 <= length(ctx.ψ1) ? ctx.ψ1[M+1] : Integrals.trigamma_int(T, M + 1)) : zero(T)
        (_pw_radvec(d1, g1.ik[M+2], g1.dl, fM, ψM, ψ1, J, n1), _pw_radvec(d2, g2.ik[M+2], g2.dl, fM, ψM, ψ1, J, n2))
    end
    tp1 = _pw_tpow(g1, maximum(A + P + l for (l, terms) in comps for (P, Q, _) in terms))
    tp2 = _pw_tpow(g2, maximum(B + Q + l for (l, terms) in comps for (P, Q, _) in terms))
    s = zero(T)
    h1 = Vector{T}(undef, n1); h2 = Vector{T}(undef, n2)
    for (l, terms) in comps
        R = _sack(C, l)                                        # e = l + 2k; Horner in t² over Σ_k co_k
        isempty(R) && continue                                 # even C with l > C/2 (R_{C,l} = 0)
        cs = [_tq(T, co) for (_, co) in R]
        _pw_horner!(h1, cs, g1.t2); _pw_horner!(h2, cs, g2.t2)
        acc = zero(T)
        for (P, Q, v) in terms
            r1, r2 = radial(A + B + P + Q + C + D + 1)
            part = _pw_dot2(h1, tp1[A+P+l+1], r1, n1) + _pw_dot2(h2, tp2[B+Q+l+1], r2, n2)
            acc += _tq(T, v) * part
        end
        s += acc * 2 / (2l + 1)
    end
    s
end
function _pw_horner!(h::Vector{T}, cs::Vector{T}, t2::Vector{T}) where {T}
    @inbounds for i in eachindex(h)
        v = cs[end]
        for k in length(cs)-1:-1:1
            v = v * t2[i] + cs[k]
        end
        h[i] = v
    end
    h
end

# Σ_i h_i a_i b_i. BigFloat uses in-place MPFR (per-operation allocations and precision lookups dominated the assembly at large L)
function _pw_dot2(h::Vector{T}, a::Vector{T}, b::Vector{T}, n::Int) where {T}
    s = zero(T)
    @inbounds for i in 1:n
        s += h[i] * (a[i] * b[i])
    end
    s
end
function _pw_dot2(h::Vector{BigFloat}, a::Vector{BigFloat}, b::Vector{BigFloat}, n::Int)
    prec = precision(h[1])
    s = BigFloat(0; precision=prec); u = BigFloat(; precision=prec)
    @inbounds for i in 1:n
        HighPrecisionLinearAlgebra.mpfr_mul!(u, a[i], b[i]); HighPrecisionLinearAlgebra.mpfr_fma!(s, h[i], u, s)
    end
    s
end
