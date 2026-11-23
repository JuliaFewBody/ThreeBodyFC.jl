# Removal of exact linear dependencies before assembly (no integrals).
#
# The function of key (A,B,C,d,j,k) is r₁^A r₂^B r₁₂^C s^d L_j e^{−α_k r₁ − β_k r₂ − γ_k r₁₂} (s = r₁ + r₂, L_j the j-th
# power of the logarithm). With βσ ≠ 0 (not split), d is a power of σ = r₁ + r₂ + βσ r₁₂, so σ is expanded instead of s
# (βσ is a binary rational, made an integer with an inverse modulo p).
# Multiplying by the common s^{−dmin} gives polynomials r₁^A r₂^B r₁₂^C (r₁ + r₂)^{d − dmin}; functions with different (j, k)
# are independent, so exact dependence is dependence of the polynomial coefficient vectors per (j, k). In a symmetrized
# basis the vectors of Φ = φ + sign·Pφ are used. The coefficients are integers, so the rank is computed modulo the prime
# p = 2^61 − 1 (a false dependence has probability about 1/p per row). This catches dependencies through chains of
# relations that the generation rules miss. Only the basis that is solved shrinks; the generators of the next stage
# (the FC function space and M_n) are unchanged.

const _DEP_P = Int64(2)^61 - 1
_mulmod(a::Int64, b::Int64) = Int64(mod(widemul(a, b), _DEP_P))

# Different components a (times u₁^a u₂^{L−a}) are independent too, so the vectors are split by (k, j, a).
function _depvec(t::Key, dmin::Int, binom::Matrix{Int64})
    A, B, C, d, j, k, a = t
    e = d - dmin
    Dict{NTuple{6,Int},Int64}((k, j, a, A + i, B + e - i, C) => binom[e+1, i+1] for i in 0:e)
end
"Value modulo p of a rational x (Float64 values are binary, hence exact)."
_modp(x::Real) = (r = Rational{BigInt}(x); Int64(mod(numerator(r) * invmod(denominator(r), big(_DEP_P)), _DEP_P)))
# Expansion of σ^e with σ = r₁ + c₂ r₂ + c₁₂ r₁₂ (p2, p12 hold the powers 0..e of c₂, c₁₂ modulo p).
function _depvec(t::Key, dmin::Int, binom::Matrix{Int64}, p2::Vector{Int64}, p12::Vector{Int64})
    A, B, C, d, j, k, a = t
    e = d - dmin
    v = Dict{NTuple{6,Int},Int64}()
    for i in 0:e, l in 0:e-i
        m = e - i - l
        c = _mulmod(_mulmod(binom[e+1, i+1], binom[e-i+1, l+1]), _mulmod(p2[l+1], p12[m+1]))
        c == 0 || (v[(k, j, a, A + i, B + l, C + m)] = c)
    end
    v
end

"""
    exact_independent(basis, es, symmetry, L=0; c2=1, c12=0) -> (basis, ndep)

Pass `c2`, `c12` when d is a power of σ = r₁ + c₂ r₂ + c₁₂ r₁₂ (c₁₂ = βσ of a non-split log variable).

Drop the symmetrized functions (pairs of a representative φ and its mirror Pφ) that lie exactly in the span of the
earlier ones. The returned basis stays closed under mirroring and spans the same space; `ndep` is the number of dropped
symmetrized functions.
"""
function exact_independent(basis::Vector{Key}, es::ExponentSets, sym::Symmetry, L::Int=0; c2::Real=1, c12::Real=0)
    isempty(basis) && return (basis, 0)
    dmin = minimum(t[4] for t in basis)
    emax = maximum(t[4] for t in basis) - dmin
    binom = [k <= n ? Int64(mod(binomial(big(n), big(k)), _DEP_P)) : Int64(0) for n in 0:emax, k in 0:emax]
    plain = isone(c2) && iszero(c12)                 # σ = r₁ + r₂ (default)
    pw(x) = (q = _modp(x); v = ones(Int64, emax + 1); for i in 2:emax+1; v[i] = _mulmod(v[i-1], q); end; v)
    p2, p12 = plain ? (Int64[], Int64[]) : (pw(c2), pw(c12))
    dv(t) = plain ? _depvec(t, dmin, binom) : _depvec(t, dmin, binom, p2, p12)
    sign = sym isa Exchange ? sym.sign : 0
    idx = Dict(b => i for (i, b) in enumerate(basis))
    piv = Dict{NTuple{6,Int},Dict{NTuple{6,Int},Int64}}()
    drop = Set{Int}()
    for (i, t) in enumerate(basis)
        mi = sign == 0 ? i : idx[_mirror_key(t, es, L)]
        i <= mi || continue
        sign < 0 && mi == i && continue
        v = dv(t)
        if sign != 0
            for (key, c) in dv(basis[mi])
                x = mod(get(v, key, Int64(0)) + sign * c, _DEP_P)
                x == 0 ? delete!(v, key) : (v[key] = x)
            end
        end
        filter!(kv -> kv[2] != 0, v)
        while true
            if isempty(v)
                push!(drop, i); push!(drop, mi)
                break
            end
            col = maximum(keys(v))
            pr = get(piv, col, nothing)
            if pr === nothing
                inv = invmod(v[col], _DEP_P)
                piv[col] = Dict(key => _mulmod(c, inv) for (key, c) in v)
                break
            end
            f = v[col]
            for (key, c) in pr
                x = mod(get(v, key, Int64(0)) - _mulmod(f, c), _DEP_P)
                x == 0 ? delete!(v, key) : (v[key] = x)
            end
        end
    end
    ndep = sign == 0 ? length(drop) : count(i -> i <= idx[_mirror_key(basis[i], es, L)], drop)
    ([t for (i, t) in enumerate(basis) if !(i in drop)], ndep)
end
