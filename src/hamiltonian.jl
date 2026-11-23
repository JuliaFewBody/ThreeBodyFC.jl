# Operator: expansion of H applied to a basis key.
#
#   H = −(1/2μ₁)∇₁² − (1/2μ₂)∇₂² − (1/m₃)∇₁·∇₂ + V,   1/μᵢ = 1/mᵢ + 1/m₃
#
# For an L=0 function f(r₁, r₂, r₁₂), ∇₁², ∇₂², ∇₁·∇₂ take the standard Hylleraas-coordinate form.
# The expansion is a dictionary keyed by monomials (A,B,C,d,j); the exponential part stays the set k of the basis.

const Poly{T} = Dict{Mono,T}
# Split (`LogVariable(split=true)`) monomials (A,B,C,d,R,j) = r₁^A r₂^B r₁₂^C s^d σ^{−R} (ln σ)^j.
# The basis has R = 0; H produces negative powers R = 1, 2 of σ.
const Mono6 = NTuple{6,Int}
const Poly6{T} = Dict{Mono6,T}

"The Hamiltonian in the working precision T (independent of the system object; built once per precision)."
struct HamiltonianData{T}
    invμ1::T
    invμ2::T
    invm3::T
    power::Vector{Tuple{NTuple{3,Int},T}}
    expo::Vector{Tuple{NTuple{3,Int},NTuple{3,Float64},T}}
    βσ::T
    split::Bool
    L::Int                    # total angular momentum (components a = 0..L; 0..L−1 for unnatural parity)
    unnatural::Bool           # unnatural parity (l₁ + l₂ = L + 1, 𝒴_a = (z₁u₂ − z₂u₁) u₁^a u₂^{L−1−a})
    w2::T                     # weight of r₂ in σ = r₁ + w₂ r₂ + βσ r₁₂ (∂σ/∂r₂ = w₂)
end
HamiltonianData{T}(i1, i2, i3, pw, ex, βσ, split) where {T} = HamiltonianData{T}(i1, i2, i3, pw, ex, βσ, split, 0, false, one(T))
HamiltonianData{T}(i1, i2, i3, pw, ex, βσ, split, L) where {T} = HamiltonianData{T}(i1, i2, i3, pw, ex, βσ, split, L, false, one(T))
HamiltonianData{T}(i1, i2, i3, pw, ex, βσ, split, L, unnatural) where {T} = HamiltonianData{T}(i1, i2, i3, pw, ex, βσ, split, L, unnatural, one(T))
# Largest component index (L for natural, L − 1 for unnatural parity); the exchange mirror is a → this − a.
_maxcomp(hd) = hd.L - hd.unnatural
# Sign of the components under exchange (−1 for unnatural parity, where v = z₁u₂ − z₂u₁ changes sign).
_xfactor(hd) = hd.unnatural ? -1 : 1

function HamiltonianData(sys::ThreeBody, lv::LogVariable, ::Type{T}; L::Int=0, unnatural::Bool=false) where {T}
    L >= 0 || throw(ArgumentError("L ≥ 0 required (L=$L)"))
    unnatural && L < 1 && throw(ArgumentError("unnatural parity requires L ≥ 1"))
    i1, i2, i3 = inverse_masses(sys, T)
    pw = Tuple{NTuple{3,Int},T}[]; ex = Tuple{NTuple{3,Int},NTuple{3,Float64},T}[]
    for t in sys.terms
        t isa PowerTerm ? push!(pw, (t.pow, exact(T, t.c))) : push!(ex, (t.pow, t.mu, exact(T, t.c)))
    end
    HamiltonianData{T}(i1, i2, i3, pw, ex, T(lv.βσ), lv.split, L, unnatural, T(lv.w2))
end

# Keys whose coefficients cancel to zero are dropped, or kept with coefficient 0 when the task-local flag is set
# (for `FCRule(keep_cancelled=true)`).
const _KEEP_CANCELLED = :tbfc_keep_cancelled
_keep() = get(task_local_storage(), _KEEP_CANCELLED, false)::Bool

@inline function _add!(w::Dict{K,T}, k::K, c::T) where {K,T}
    iszero(c) && return w
    v = get(w, k, zero(T)) + c
    iszero(v) && !_keep() ? delete!(w, k) : (w[k] = v)
    w
end
_addall!(dest::Dict{K,T}, src::Dict{K,T}, f::T=one(T)) where {K,T} = (for (k, c) in src; _add!(dest, k, c * f); end; dest)

function _shift(w::Poly{T}, dA, dB, dC, dd, f::T) where {T}
    r = Poly{T}()
    for ((A, B, C, d, j), c) in w
        _add!(r, (A + dA, B + dB, C + dC, d + dd, j), c * f)
    end
    r
end

# ∂/∂r₁ (including the exponential). ∂σ/∂r₁ = 1, ∂(ln σ)^j = j(ln σ)^{j−1}/σ.
function _d1(w::Poly{T}, α::T) where {T}
    r = Poly{T}()
    for ((A, B, C, d, j), c) in w
        A != 0 && _add!(r, (A - 1, B, C, d, j), c * A)
        d != 0 && _add!(r, (A, B, C, d - 1, j), c * d)
        j != 0 && _add!(r, (A, B, C, d - 1, j - 1), c * j)
        _add!(r, (A, B, C, d, j), -α * c)
    end
    r
end
# ∂/∂r₂. ∂σ/∂r₂ = w₂ (default 1).
function _d2(w::Poly{T}, β::T, w2::T=one(T)) where {T}
    r = Poly{T}()
    u = isone(w2)
    for ((A, B, C, d, j), c) in w
        B != 0 && _add!(r, (A, B - 1, C, d, j), c * B)
        d != 0 && _add!(r, (A, B, C, d - 1, j), u ? c * d : c * d * w2)
        j != 0 && _add!(r, (A, B, C, d - 1, j - 1), u ? c * j : c * j * w2)
        _add!(r, (A, B, C, d, j), -β * c)
    end
    r
end
# ∂/∂r₁₂. ∂σ/∂r₁₂ = βσ; the Gaussian e^{−η r₁₂²} gives −2η r₁₂.
function _d12(w::Poly{T}, γ::T, βσ::T, η::T=zero(T)) where {T}
    r = Poly{T}()
    for ((A, B, C, d, j), c) in w
        C != 0 && _add!(r, (A, B, C - 1, d, j), c * C)
        if !iszero(βσ)
            d != 0 && _add!(r, (A, B, C, d - 1, j), c * d * βσ)
            j != 0 && _add!(r, (A, B, C, d - 1, j - 1), c * j * βσ)
        end
        _add!(r, (A, B, C, d, j), -γ * c)
        iszero(η) || _add!(r, (A, B, C + 1, d, j), -2η * c)
    end
    r
end

# Split derivatives. s^d depends on r₁, r₂ only (∂s/∂r₁₂ = 0), and for σ = s + βσ r₁₂
# ∂(σ^{−R}(ln σ)^j) = (−R σ^{−R−1}(ln σ)^j + j σ^{−R−1}(ln σ)^{j−1}) ∂σ.
function _shift(w::Poly6{T}, dA, dB, dC, dd, f::T) where {T}
    r = Poly6{T}()
    for ((A, B, C, d, R, j), c) in w
        _add!(r, (A + dA, B + dB, C + dC, d + dd, R, j), c * f)
    end
    r
end
@inline function _dsigma!(r::Poly6{T}, A, B, C, d, R, j, c::T) where {T}
    R != 0 && _add!(r, (A, B, C, d, R + 1, j), -c * R)
    j != 0 && _add!(r, (A, B, C, d, R + 1, j - 1), c * j)
    r
end
function _d1(w::Poly6{T}, α::T) where {T}
    r = Poly6{T}()
    for ((A, B, C, d, R, j), c) in w
        A != 0 && _add!(r, (A - 1, B, C, d, R, j), c * A)
        d != 0 && _add!(r, (A, B, C, d - 1, R, j), c * d)
        _dsigma!(r, A, B, C, d, R, j, c)
        _add!(r, (A, B, C, d, R, j), -α * c)
    end
    r
end
_d2(w::Poly6{T}, β::T, w2::T) where {T} = (isone(w2) || throw(ArgumentError("split requires w₂ = 1")); _d2(w, β))
function _d2(w::Poly6{T}, β::T) where {T}
    r = Poly6{T}()
    for ((A, B, C, d, R, j), c) in w
        B != 0 && _add!(r, (A, B - 1, C, d, R, j), c * B)
        d != 0 && _add!(r, (A, B, C, d - 1, R, j), c * d)
        _dsigma!(r, A, B, C, d, R, j, c)
        _add!(r, (A, B, C, d, R, j), -β * c)
    end
    r
end
function _d12(w::Poly6{T}, γ::T, βσ::T, η::T=zero(T)) where {T}
    r = Poly6{T}()
    for ((A, B, C, d, R, j), c) in w
        C != 0 && _add!(r, (A, B, C - 1, d, R, j), c * C)
        iszero(βσ) || _dsigma!(r, A, B, C, d, R, j, c * βσ)
        _add!(r, (A, B, C, d, R, j), -γ * c)
        iszero(η) || _add!(r, (A, B, C + 1, d, R, j), -2η * c)
    end
    r
end

"""
    kinetic_terms(m, α, β, γ, βσ, η=0) -> (∇₁²f, ∇₂²f, ∇₁·∇₂f)

Expansions of the three differential operators acting on the monomial m times exp(−αr₁−βr₂−γr₁₂−ηr₁₂²).
"""
function kinetic_terms(m::Union{Mono,Mono6}, α::T, β::T, γ::T, βσ::T, η::T=zero(T), w2::T=one(T)) where {T}
    o, h, two = one(T), T(1) / 2, T(2)
    f = Dict{typeof(m),T}(m => o)
    D1 = _d1(f, α); D2 = _d2(f, β, w2); D12 = _d12(f, γ, βσ, η)
    t11 = _d1(D1, α); t12 = _shift(D1, -1, 0, 0, 0, two)
    t13 = _d12(D12, γ, βσ, η); t14 = _shift(D12, 0, 0, -1, 0, two)
    x1 = _d12(D1, γ, βσ, η)
    t15 = empty(f)                         # (r₁² + r₁₂² − r₂²)/(r₁r₁₂) ∂₁∂₁₂ f
    _addall!(t15, _shift(x1, 1, 0, -1, 0, o)); _addall!(t15, _shift(x1, -1, 0, 1, 0, o))
    _addall!(t15, _shift(x1, -1, 2, -1, 0, -o))
    lap1 = empty(f); for w in (t11, t12, t13, t14, t15); _addall!(lap1, w); end

    t21 = _d2(D2, β, w2); t22 = _shift(D2, 0, -1, 0, 0, two)
    x2 = _d12(D2, γ, βσ, η)
    t25 = empty(f)                         # (r₂² + r₁₂² − r₁²)/(r₂r₁₂) ∂₂∂₁₂ f
    _addall!(t25, _shift(x2, 0, 1, -1, 0, o)); _addall!(t25, _shift(x2, 0, -1, 1, 0, o))
    _addall!(t25, _shift(x2, 2, -1, -1, 0, -o))
    lap2 = empty(f); for w in (t21, t22, t13, t14, t25); _addall!(lap2, w); end

    D1D2 = _d2(D1, β, w2)
    mp = empty(f)                          # ∇₁·∇₂ f
    _addall!(mp, _shift(D1D2, 1, -1, 0, 0, h)); _addall!(mp, _shift(D1D2, -1, 1, 0, 0, h))
    _addall!(mp, _shift(D1D2, -1, -1, 2, 0, -h))
    _addall!(mp, t15, -h); _addall!(mp, t25, -h); _addall!(mp, t13, -o); _addall!(mp, t14, -o)
    (lap1, lap2, mp)
end

"""
    _angular_terms!(out, f, hd, α, β, γ, a) -> [(a−1, expansion), (a+1, expansion)]

The extra kinetic terms for L ≥ 1 due to the component 𝒴_a = u₁^a u₂^{L−a}:
  same component: −(1/μ₁) d₁ (f₁/r₁ + f₁₂/r₁₂) − (1/μ₂) d₂ (f₂/r₂ + f₁₂/r₁₂) + (1/m₃) (d₁ + d₂) f₁₂/r₁₂
  𝒴_{a−1}:  (1/μ₁) a f₁₂/r₁₂ − (1/m₃) a (f₂/r₂ + f₁₂/r₁₂)
  𝒴_{a+1}:  (1/μ₂) b f₁₂/r₁₂ − (1/m₃) b (f₁/r₁ + f₁₂/r₁₂)
Natural parity: b = L − a, d₁ = a, d₂ = b. Unnatural parity (𝒴_a = v u₁^a u₂^{L−1−a}, v = z₁u₂ − z₂u₁): b = L − 1 − a,
d₁ = a + 1, d₂ = b + 1 (degrees of 𝒴 in r₁, r₂; r₂·∇₁v = r₁·∇₂v = 0, so the neighbor coefficients stay a, b).
The same-component part is added to `out`; the neighbor expansions are returned.
"""
function _angular_terms!(out::Dict{K,T}, f::Dict{K,T}, hd::HamiltonianData{T}, α::T, β::T, γ::T, a::Int, η::T=zero(T)) where {K,T}
    p = Int(hd.unnatural); b = hd.L - p - a; d1 = a + p; d2 = b + p; o = one(T)
    f1r = _shift(_d1(f, α), -1, 0, 0, 0, o); f2r = _shift(_d2(f, β, hd.w2), 0, -1, 0, 0, o); f12r = _shift(_d12(f, γ, hd.βσ, η), 0, 0, -1, 0, o)
    _addall!(out, f1r, -hd.invμ1 * d1); _addall!(out, f12r, -hd.invμ1 * d1)
    _addall!(out, f2r, -hd.invμ2 * d2); _addall!(out, f12r, -hd.invμ2 * d2)
    iszero(hd.invm3) || _addall!(out, f12r, hd.invm3 * (d1 + d2))
    nb = Tuple{Int,Dict{K,T}}[]
    if a >= 1
        w = empty(f); _addall!(w, f12r, hd.invμ1 * a)
        iszero(hd.invm3) || (_addall!(w, f2r, -hd.invm3 * a); _addall!(w, f12r, -hd.invm3 * a))
        push!(nb, (a - 1, w))
    end
    if b >= 1
        w = empty(f); _addall!(w, f12r, hd.invμ2 * b)
        iszero(hd.invm3) || (_addall!(w, f1r, -hd.invm3 * b); _addall!(w, f12r, -hd.invm3 * b))
        push!(nb, (a + 1, w))
    end
    nb
end

"""
    apply_H(hd, es, key; grow=false) -> Vector{Tuple{Mono,Int,Int,T}}

Expansion of H·φ_key as (monomial, exponent set index, angular component, coefficient). `ExpTerm` shifts the set.
With `grow=true` a missing target set is added to `es` (generation); with `false` it is an error.
"""
function apply_H(hd::HamiltonianData{T}, es::ExponentSets, key::Key; grow::Bool=false,
                 potential_only::Bool=false) where {T}
    hd.split && return _apply_H_split(hd, es, key; grow=grow, potential_only=potential_only)
    A, B, C, d, j, k, a = key
    m = (A, B, C, d, j)
    out = Poly{T}()
    nb = nothing
    if !potential_only
        lap1, lap2, mp = kinetic_terms(m, T(es.α[k]), T(es.β[k]), T(es.γ[k]), hd.βσ, T(es.η[k]), hd.w2)
        _addall!(out, lap1, -hd.invμ1 / 2)
        _addall!(out, lap2, -hd.invμ2 / 2)
        iszero(hd.invm3) || _addall!(out, mp, -hd.invm3)
        hd.L > 0 && (nb = _angular_terms!(out, Poly{T}(m => one(T)), hd, T(es.α[k]), T(es.β[k]), T(es.γ[k]), a, T(es.η[k])))
    end
    for ((p, q, s), c) in hd.power
        _add!(out, (A + p, B + q, C + s, d, j), c)
    end
    res = Tuple{Mono,Int,Int,T}[(mm, k, a, c) for (mm, c) in out]
    nb === nothing || for (aa, w) in nb, (mm, c) in w; push!(res, (mm, k, aa, c)); end
    for ((p, q, s), mu, c) in hd.expo
        kk = grow ? shifted_set!(es, k, mu) : find_set(es, es.α[k] + mu[1], es.β[k] + mu[2], es.γ[k] + mu[3], es.η[k])
        kk == 0 && error("target exponent set missing (create it with grow=true during generation)")
        push!(res, ((A + p, B + q, C + s, d, j), kk, a, c))
    end
    res
end

"H·φ for split log variables; monomials are Mono6 (A,B,C,d,R,j)."
function _apply_H_split(hd::HamiltonianData{T}, es::ExponentSets, key::Key; grow::Bool=false,
                        potential_only::Bool=false) where {T}
    A, B, C, d, j, k, a = key
    m = (A, B, C, d, 0, j)
    out = Poly6{T}()
    nb = nothing
    if !potential_only
        lap1, lap2, mp = kinetic_terms(m, T(es.α[k]), T(es.β[k]), T(es.γ[k]), hd.βσ, T(es.η[k]))
        _addall!(out, lap1, -hd.invμ1 / 2)
        _addall!(out, lap2, -hd.invμ2 / 2)
        iszero(hd.invm3) || _addall!(out, mp, -hd.invm3)
        hd.L > 0 && (nb = _angular_terms!(out, Poly6{T}(m => one(T)), hd, T(es.α[k]), T(es.β[k]), T(es.γ[k]), a, T(es.η[k])))
    end
    for ((p, q, s), c) in hd.power
        _add!(out, (A + p, B + q, C + s, d, 0, j), c)
    end
    res = Tuple{Mono6,Int,Int,T}[(mm, k, a, c) for (mm, c) in out]
    nb === nothing || for (aa, w) in nb, (mm, c) in w; push!(res, (mm, k, aa, c)); end
    for ((p, q, s), mu, c) in hd.expo
        kk = grow ? shifted_set!(es, k, mu) : find_set(es, es.α[k] + mu[1], es.β[k] + mu[2], es.γ[k] + mu[3], es.η[k])
        kk == 0 && error("target exponent set missing (create it with grow=true during generation)")
        push!(res, ((A + p, B + q, C + s, d, 0, j), kk, a, c))
    end
    res
end
