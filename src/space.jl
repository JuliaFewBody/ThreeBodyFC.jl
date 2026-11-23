# Function space: basis keys and exponent sets.
#
# A key (A,B,C,d,j,k,a) stands for
#     r₁^A r₂^B r₁₂^C σ^d (ln σ)^j exp(−α_k r₁ − β_k r₂ − γ_k r₁₂ − η_k r₁₂²) · 𝒴_a,   σ = r₁ + r₂ + β_σ r₁₂.
# Exponents are positive for decay. η_k > 0 adds a Gaussian in r₁₂ (e^{−η(r₁₂−Rₑ)²} has γ = −2ηRₑ; the constant is dropped).
# a is the angular component for L ≥ 1 (𝒴_a = u₁^a u₂^{L−a}, u = x + iy); for L = 0 only a = 0 (𝒴₀ = 1).

const Key = NTuple{7,Int}       # (A,B,C,d,j,k,a)
const Mono = NTuple{5,Int}      # (A,B,C,d,j)

"""
    ExponentSets(α, β, γ; η=zeros)

List of exponent sets (mutable: `ExpTerm` potentials add sets). η is the coefficient of the Gaussian e^{−η r₁₂²}
(default 0, none). Sets with η > 0 may have negative γ (`gaussian_r12`).
"""
struct ExponentSets
    α::Vector{Float64}
    β::Vector{Float64}
    γ::Vector{Float64}
    η::Vector{Float64}
    function ExponentSets(α, β, γ; η=zeros(length(α)))
        length(α) == length(β) == length(γ) == length(η) || throw(ArgumentError("α, β, γ, η must have the same length"))
        all(>=(0), η) || throw(ArgumentError("η must be ≥ 0 (η=$η)"))
        new(Float64.(collect(α)), Float64.(collect(β)), Float64.(collect(γ)), Float64.(collect(η)))
    end
end
Base.length(e::ExponentSets) = length(e.α)
Base.copy(e::ExponentSets) = ExponentSets(copy(e.α), copy(e.β), copy(e.γ); η=copy(e.η))
_gaussian(es::ExponentSets) = any(!iszero, es.η)

"""
    gaussian_r12(α, η, Re) -> ExponentSets

Sets for e^{−α(r₁+r₂)} e^{−η(r₁₂−Rₑ)²} (the H₂⁺ initial function of Hijikata–Nakashima–Nakatsuji, JCP 130, 024102 (2009)).
Arguments may be vectors of equal length. γ = −2ηRₑ (the constant e^{−ηRₑ²} is dropped).
"""
gaussian_r12(α, η, Re) = (a = collect(Float64, vcat(α)); e = collect(Float64, vcat(η)); r = collect(Float64, vcat(Re));
                          ExponentSets(a, copy(a), -2 .* e .* r; η=e))

"Index of the set (α_k+μ₁, β_k+μ₂, γ_k+μ₁₂), added if absent."
function shifted_set!(es::ExponentSets, k::Int, mu::NTuple{3,Float64})
    a = es.α[k] + mu[1]; b = es.β[k] + mu[2]; g = es.γ[k] + mu[3]
    i = find_set(es, a, b, g, es.η[k])
    i == 0 || return i
    push!(es.α, a); push!(es.β, b); push!(es.γ, g); push!(es.η, es.η[k])
    length(es)
end

const _SET_RTOL = 1e-12    # finer than the 2⁻²⁰ binary grid, coarser than Float64 rounding of sums
function find_set(es::ExponentSets, a, b, g, η=0.0)
    tol(x) = _SET_RTOL * max(1.0, abs(x))
    for i in 1:length(es)
        abs(es.α[i] - a) <= tol(a) && abs(es.β[i] - b) <= tol(b) && abs(es.γ[i] - g) <= tol(g) && abs(es.η[i] - η) <= tol(η) && return i
    end
    0
end

"""
    LogVariable(βσ; split=false, w2=1)

Argument of the logarithm σ = r₁ + w₂ r₂ + βσ·r₁₂ (σ = r₁ + r₂ for βσ = 0, w₂ = 1). Requires βσ > −1, w₂ > 0, w₂ + βσ > 0.
w₂ ≠ 1 is incompatible with exchange symmetry and with `split`.

- `split=false` (default): d in the key is a power of σ; basis r₁^A r₂^B r₁₂^C σ^d (ln σ)^j e^{…}.
- `split=true`: d is a power of s = r₁ + r₂ and only the logarithm has argument σ; basis
  r₁^A r₂^B r₁₂^C s^d [ln(s + βσ r₁₂)]^j e^{…}, the initial function ψ₀ = [1 + ln(s + βu)] e^{−αs} of
  Nakashima–Nakatsuji, JCP 127, 224104 (2007). The function set is generated in the βσ = 0 ring.
  Sets with α ≠ β work, with more expensive (two-dimensional) element integrals.
"""
struct LogVariable
    βσ::Float64
    split::Bool
    w2::Float64
    function LogVariable(b::Real = 0.0; split::Bool = false, w2::Real = 1.0)
        1 + b > 0 || throw(ArgumentError("βσ > −1 required (βσ=$b)"))
        w2 > 0 && w2 + b > 0 || throw(ArgumentError("w₂ > 0 and w₂ + βσ > 0 required (w₂=$w2, βσ=$b)"))
        split && w2 != 1 && throw(ArgumentError("split requires w₂ = 1"))
        new(Float64(b), split, Float64(w2))
    end
end
lambda(lv::LogVariable) = 1.0 + lv.βσ
Integrals.IntegralContext(::Type{T}, engine::KernelEngine, lv::LogVariable; kwargs...) where {T} =
    IntegralContext(T, engine; βσ=lv.βσ, split=lv.split, w2=lv.w2, kwargs...)
describe(io::IO, lv::LogVariable) =
    print(io, "logvar βσ=", lv.βσ, lv.split ? " split (powers of s × ln(s+βσ r₁₂))" : "", isone(lv.w2) ? "" : " w₂=$(lv.w2) (σ = r₁ + w₂ r₂ + βσ r₁₂)")

"""
    check_exponents(es)

Check that every set decays (perimetric exponents (β+γ, α+γ, α+β) all positive; r₁ = v+w, r₂ = u+w, r₁₂ = u+v),
so that errors appear here rather than deep inside the parallel assembly. With a Gaussian (η > 0) only α + β > 0 is
needed, but the Gaussian element integrals require α = β for all sets.
"""
function check_exponents(es::ExponentSets)
    if _gaussian(es)
        for k in 1:length(es)
            es.α[k] == es.β[k] && es.α[k] > 0 ||
                throw(ArgumentError("with a Gaussian (η > 0), all sets need α = β > 0 (set $k: α=$(es.α[k]), β=$(es.β[k]))"))
        end
        return nothing
    end
    for k in 1:length(es)
        a, b, g = es.α[k], es.β[k], es.γ[k]
        (b + g > 0 && a + g > 0 && a + b > 0) ||
            throw(ArgumentError("exponent set $k = (α, β, γ) = ($a, $b, $g) does not decay (β+γ, α+γ, α+β must be positive)"))
    end
    nothing
end

"""
    binary_warning(es; shifts) -> Union{Nothing,String}

A warning if exponents are not on a binary grid, so that the Float64 sums forming element exponents may be rounded.
Rounded sums shift cross-set elements of H by 1e-16 independently of the working precision, which can produce
spurious eigenvalues when S is nearly dependent. Harmless when all sets share the same rounding (same α, γ).
"""
function binary_warning(es::ExponentSets; shifts=NTuple{3,Float64}[])
    vals = [es.α; es.β; es.γ; es.η; [x for m in shifts for x in m]]
    q = maximum(_fracbits, vals; init=0)
    16maximum(abs, vals; init=0.0) < 2.0^(53 - q) && return nothing
    bad = first(x for x in vals if _fracbits(x) == q)
    "exponent $bad (and others) is not on a binary grid (step 2^-$q); sums of exponents across sets are rounded in Float64, " *
    "which can produce spurious eigenvalues if S is nearly dependent. `binary_exponents(es)` (2^-20 grid) is recommended"
end

"q of the binary grid step 2^{-q} that represents x (0 for integers)."
function _fracbits(x::Float64)
    (iszero(x) || !isfinite(x)) && return 0
    m = Int64(significand(abs(x)) * 2.0^52)
    max(0, 52 - exponent(x) - trailing_zeros(m))
end

"""
    binary_exponents(es; bits=20) -> ExponentSets

The sets with every exponent rounded to the 2^{-bits} grid.
"""
binary_exponents(es::ExponentSets; bits::Int=20) =
    ExponentSets([round(x * 2.0^bits) / 2.0^bits for x in es.α], [round(x * 2.0^bits) / 2.0^bits for x in es.β],
                 [round(x * 2.0^bits) / 2.0^bits for x in es.γ]; η=[round(x * 2.0^bits) / 2.0^bits for x in es.η])

"""
    NewLog(β = 1.0)

The initial function ψ₀ = [1 + ln(s + βu)] e^{…}; same as `LogVariable(β; split=true)`.
"""
NewLog(β::Real=1.0) = LogVariable(β; split=true)
