# The system: masses of the three particles and the potential.
#
# Coordinates r₁ = |x₁ − x₃|, r₂ = |x₂ − x₃|, r₁₂ = |x₁ − x₂| (particle 3 is the reference).

"""
    PowerTerm(c, (p, q, s))

Potential term c · r₁^p r₂^q r₁₂^s (Coulomb terms have negative powers such as (−1, 0, 0)).
"""
struct PowerTerm
    c::String              # the coefficient is kept as a string and read in the working precision
    pow::NTuple{3,Int}
end
PowerTerm(c::AbstractString, pow) = PowerTerm(String(c), Tuple(pow))
PowerTerm(c::Integer, pow) = PowerTerm(string(c), Tuple(pow))
PowerTerm(c::AbstractFloat, pow) = PowerTerm(_exact_string(c), Tuple(pow))

"""
    ExpTerm(c, (p, q, s), (μ₁, μ₂, μ₁₂))

Potential term c · r₁^p r₂^q r₁₂^s · exp(−μ₁r₁ − μ₂r₂ − μ₁₂r₁₂) (Yukawa and similar). Acting on exponent set k it
produces the set (α_k+μ₁, β_k+μ₂, γ_k+μ₁₂).
"""
struct ExpTerm
    c::String
    pow::NTuple{3,Int}
    mu::NTuple{3,Float64}
end
ExpTerm(c::AbstractString, pow, mu) = ExpTerm(String(c), Tuple(pow), Tuple(Float64.(mu)))
ExpTerm(c::Integer, pow, mu) = ExpTerm(string(c), Tuple(pow), Tuple(Float64.(mu)))
ExpTerm(c::AbstractFloat, pow, mu) = ExpTerm(_exact_string(c), Tuple(pow), Tuple(Float64.(mu)))

"""
A floating-point coefficient becomes the exact fraction of its binary value. The decimal `repr` round-trips but is a
different number, and a relative 1e-16 difference can be amplified to 1e-11 in the elements by cancellations.
"""
_exact_string(c::AbstractFloat) = (r = Rational{BigInt}(c); denominator(r) == 1 ? string(numerator(r)) : string(numerator(r), "/", denominator(r)))

"""
    ThreeBody(masses, terms)

`masses = (m₁, m₂, m₃)` as strings (`"Inf"` for an infinite mass); `terms` is a list of `PowerTerm`/`ExpTerm`.
Coefficients are strings (exact decimal values) or floats (exact binary values); the two define different operators.
"""
struct ThreeBody
    masses::NTuple{3,String}
    terms::Vector{Union{PowerTerm,ExpTerm}}
    ThreeBody(masses, terms) = new(Tuple(String.(string.(masses))), Union{PowerTerm,ExpTerm}[terms...])
end

"""
    coulomb(q₁, q₂, q₃) -> Vector{PowerTerm}

Coulomb interaction q₁q₃/r₁ + q₂q₃/r₂ + q₁q₂/r₁₂ for charges given as strings or integers.
"""
function coulomb(q1, q2, q3)
    p(a, b) = _strmul(string(a), string(b))
    [PowerTerm(p(q1, q3), (-1, 0, 0)), PowerTerm(p(q2, q3), (0, -1, 0)), PowerTerm(p(q1, q2), (0, 0, -1))]
end
_strmul(a::String, b::String) = (r = _parse_rat(a) * _parse_rat(b); denominator(r) == 1 ? string(numerator(r)) : string(numerator(r), "/", denominator(r)))
_parse_rat(s::String) = occursin('/', s) ? (x = split(s, '/'); parse(BigInt, x[1]) // parse(BigInt, x[2])) : parse(BigInt, s) // 1

"""
    exact(T, s) -> T

A constant of the Hamiltonian from its string in the working precision `T` (accepts fractions "a/b" and "Inf").
"""
function exact(::Type{T}, s::AbstractString) where {T}
    s == "Inf" && return T(Inf)
    setprecision(BigFloat, max(precision(BigFloat), 512)) do
        if occursin('/', s)
            a, b = split(s, '/')
            return T(BigFloat(a) / BigFloat(b))
        end
        T(BigFloat(s))
    end
end

"Inverse reduced masses 1/μ₁, 1/μ₂ (particle i with particle 3) and 1/m₃ in type T."
function inverse_masses(sys::ThreeBody, ::Type{T}) where {T}
    m1, m2, m3 = (exact(T, m) for m in sys.masses)
    inv3 = isinf(m3) ? zero(T) : inv(m3)
    (inv(m1) + inv3, inv(m2) + inv3, inv3)
end

function describe(io::IO, sys::ThreeBody)
    println(io, "system masses=(", join(sys.masses, ", "), ")")
    for t in sys.terms
        if t isa PowerTerm
            println(io, "  term ", t.c, " · r1^", t.pow[1], " r2^", t.pow[2], " r12^", t.pow[3])
        else
            println(io, "  term ", t.c, " · r1^", t.pow[1], " r2^", t.pow[2], " r12^", t.pow[3],
                    " · exp(-", t.mu[1], " r1 - ", t.mu[2], " r2 - ", t.mu[3], " r12)")
        end
    end
end
