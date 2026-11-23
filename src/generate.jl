# Generation: the FC rule.
#
#   functions of ψ_{n+1} = {g·φ, g·(each term of Hφ) : φ ∈ ψ_n, g ∈ gs} that satisfy the admission rule
#
# g is written as shifts of monomial powers (dA, dB, dC, dd); e.g. g = r₁r₂/σ + r₁₂ is [(1,1,0,−1), (0,0,1,0)].

"""
    FCRule(; seeds, g, dmin, dedup, potential_only)

- `seeds`: monomials (A,B,C,d,j) of ψ₀, the same for every exponent set. Default {1, ln σ}.
- `g`: list of monomial shifts.
- `dmin`: lower bound on the power d of σ: an integer, `:paper` (−2n at stage n), or a rational q (−⌈q n⌉; `:paper` is q = 2).
- `dedup`: drop keys that are exactly dependent through σ = r₁ + r₂, (A,B,C,d) = (A+1,B,C,d−1) + (A,B+1,C,d−1), at every
  stage. Not valid for βσ ≠ 0.
- `potential_only`: generate only from the potential terms of Hφ.
- `keep_cancelled`: also use monomials of Hφ whose coefficients cancel exactly. The default (false) collects only the
  functions that appear on the right-hand side; the two differ for exponents that satisfy the cusp condition.
- `exp_generate`: also generate from terms whose exponent set was shifted by an `ExpTerm`. If false the exponent sets are
  fixed and `ExpTerm` only enters the matrix elements.
- `gcoef`: coefficients (strings) of the terms of g, all 1 if empty. E.g. g = −1/V_Ne + 1/V_ee = r₁r₂/(Zσ) + r₁₂ for He
  is `g=[(1,1,0,-1), (0,0,1,0)], gcoef=["1/Z", "1"]`. Only used for the cancellation test of `collect=:st`.
- `collect`: in which representation the functions "appearing" on the right-hand side are counted.
  - `:monomial` (default): each term of g times each monomial r₁^A r₂^B r₁₂^C σ^d of Hφ.
  - `:st`: add g·Hφ and g·φ with coefficients, expand in s = r₁+r₂, t = r₁−r₂ monomials s^l t^m r₁₂^C (ln s)^j and keep
    those with nonzero coefficient (above `rtol` times the largest). Terms with negative powers of r₁, r₂, r₁₂ are
    dropped before expanding, and ω = l+m+C ≥ 0. For exchange-symmetric sets, odd or even powers of t are dropped by
    `parity`. This is the rule of Nakashima–Nakatsuji, JCP 127, 224104 (2007) and reproduces the He M_n (2, 10, 34, 77,
    146, …). Needs βσ = 0 (σ = s). For each collected (l+m, C, j, k) the largest m = D is taken and the space is spanned
    by the top keys r₁^a r₂^{D−a} σ^{l+m−D} (a = 0..D).
"""
Base.@kwdef struct FCRule
    seeds::Vector{Mono} = Mono[(0, 0, 0, 0, 0), (0, 0, 0, 0, 1)]
    g::Vector{NTuple{4,Int}} = [(1, 1, 0, -1), (0, 0, 1, 0)]
    dmin::Union{Int,Symbol,Rational{Int}} = :paper
    dedup::Bool = true
    potential_only::Bool = false
    keep_cancelled::Bool = false
    jmax::Int = 1
    exp_generate::Bool = true
    gcoef::Vector{String} = String[]
    collect::Symbol = :monomial
    rtol::Float64 = 1e-10
end

describe(io::IO, r::FCRule) =
    println(io, "rule seeds=", r.seeds, " g=", r.g, " dmin=", r.dmin, " dedup=", r.dedup,
            " potential_only=", r.potential_only, " keep_cancelled=", r.keep_cancelled, " jmax=", r.jmax,
            " exp_generate=", r.exp_generate, " gcoef=", isempty(r.gcoef) ? "1" : r.gcoef,
            " collect=", r.collect, r.collect === :st ? " rtol=$(r.rtol)" : "")

_dmin(r::FCRule, n::Int) = r.dmin === :paper ? -2n : r.dmin isa Rational ? -ceil(Int, r.dmin * n) : Int(r.dmin)

"Admission rule: A,B,C ≥ 0, d ≥ dmin, 0 ≤ j ≤ jmax, A+B+C+d ≥ 0 (integrability of the elements)."
_admissible(t::Key, dmin::Int, jmax::Int) =
    t[1] >= 0 && t[2] >= 0 && t[3] >= 0 && t[4] >= dmin && 0 <= t[5] <= jmax && t[1] + t[2] + t[3] + t[4] >= 0

# Components start at a = 0 ((l₁,l₂) = (0,L) for L ≥ 1). Without symmetrization the mirror a = L is seeded too
# (`_seed_comps` in the driver); with a = 0 alone the space is not closed under 1 ↔ 2.
seed_basis(r::FCRule, es::ExponentSets, comps=0:0) = sort!([(m..., k, a) for k in 1:length(es) for m in r.seeds for a in comps])

"""
    drop_dependent(basis) -> basis

For σ = r₁ + r₂, (A,B,C,d,j,k) is the sum of (A+1,B,C,d−1,j,k) and (A,B+1,C,d−1,j,k), so it adds nothing to the span
if both are present. Drop such keys (the span is unchanged).
"""
function drop_dependent(v::Vector{Key})
    S = Set(v)
    filter(t -> !(((t[1] + 1, t[2], t[3], t[4] - 1, t[5], t[6], t[7]) in S) &&
                  ((t[1], t[2] + 1, t[3], t[4] - 1, t[5], t[6], t[7]) in S)), v)
end

"""
    next_stage(rule, hd, es, basis, n) -> basis

Function set of stage n. `es` may grow through `ExpTerm`.
"""
function next_stage(rule::FCRule, hd::HamiltonianData{T}, es::ExponentSets, basis::Vector{Key}, n::Int;
                    parity::Int=0) where {T}
    rule.collect === :st && return next_stage_st(rule, hd, es, basis, n; parity=parity)
    rule.collect === :monomial || throw(ArgumentError("collect must be :monomial or :st ($(rule.collect))"))
    rule.dedup && !(iszero(hd.βσ) && isone(hd.w2)) && throw(ArgumentError("dedup requires βσ = 0 and w₂ = 1"))
    dmin = _dmin(rule, n)
    s = Set{Key}(basis)
    for key in basis
        A, B, C, d, j, k, a = key
        for (gA, gB, gC, gd) in rule.g
            t = (A + gA, B + gB, C + gC, d + gd, j, k, a)
            _admissible(t, dmin, rule.jmax) && push!(s, t)
        end
        Hφ = task_local_storage(_KEEP_CANCELLED, rule.keep_cancelled) do
            apply_H(hd, es, key; grow=true, potential_only=rule.potential_only)
        end
        for ((x, y, z, dd, jl), kk, aa, _) in Hφ
            rule.exp_generate || kk == k || continue
            for (gA, gB, gC, gd) in rule.g
                t = (x + gA, y + gB, z + gC, dd + gd, jl, kk, aa)
                _admissible(t, dmin, rule.jmax) && push!(s, t)
            end
        end
    end
    out = sort!(collect(s))
    rule.dedup ? drop_dependent(out) : out
end

# =============================================================================
# collect = :st (collect the functions appearing on the right-hand side in the s, t representation)
# =============================================================================

"Terms of g with their coefficients in T."
function _gterms(rule::FCRule, ::Type{T}) where {T}
    isempty(rule.gcoef) && return [(g, one(T)) for g in rule.g]
    length(rule.gcoef) == length(rule.g) || throw(ArgumentError("gcoef and g have different lengths"))
    [(g, exact(T, c)) for (g, c) in zip(rule.g, rule.gcoef)]
end

"r₁^A r₂^B = 2^{−(A+B)} Σ c_i s^i t^{A+B−i} as (i, c_i) (integer coefficients; the caller applies 2^{−(A+B)})."
function _r1r2_to_st(A::Int, B::Int)
    out = Dict{Int,BigInt}()
    for a in 0:A, b in 0:B          # (s+t)^A (s−t)^B
        c = binomial(BigInt(A), a) * binomial(BigInt(B), b) * (isodd(B - b) ? -1 : 1)
        i = a + b
        out[i] = get(out, i, BigInt(0)) + c
    end
    out
end

"Group (l+m, C, j, k) of a key and the largest power of t it spans."
_st_group(t::Key) = ((t[1] + t[2] + t[4], t[3], t[5], t[6]), t[1] + t[2])   # collect=:st is L = 0 only (a = 0)

"""
    next_stage_st(rule, hd, es, basis, n; parity) -> basis

One stage of `collect = :st`. `parity` is the exchange sign (+1: even powers of t only, −1: odd only, 0: both) and
applies only to sets with α_k = β_k.
"""
function next_stage_st(rule::FCRule, hd::HamiltonianData{T}, es::ExponentSets, basis::Vector{Key}, n::Int;
                       parity::Int=0) where {T}
    iszero(hd.βσ) && isone(hd.w2) || throw(ArgumentError("collect=:st requires βσ = 0 and w₂ = 1 (σ = r₁+r₂)"))
    hd.L == 0 && all(t -> t[7] == 0, basis) || throw(ArgumentError("collect=:st requires L = 0"))
    gt = _gterms(rule, T)
    selfmirror(k) = es.α[k] == es.β[k]
    mrange(D, k) = parity != 0 && selfmirror(k) ? ((parity < 0 ? 1 : 0):2:D) : (0:D)
    ms = Dict{NTuple{4,Int},Set{Int}}()            # (l+m, C, j, k) → powers m that appear
    for t in basis
        grp, D = _st_group(t)
        s = get!(ms, grp, Set{Int}())
        for m in mrange(D, t[6]); push!(s, m); end
    end
    st_cache = Dict{NTuple{2,Int},Dict{Int,BigInt}}()
    for key in basis
        A0, B0, C0, d0, j0, k0, _ = key
        Hφ = apply_H(hd, es, key; grow=true, potential_only=rule.potential_only)
        srcs = (Hφ, Tuple{Mono,Int,Int,T}[((A0, B0, C0, d0, j0), k0, 0, one(T))])
        for src in srcs
            acc = Dict{NTuple{6,Int},T}()
            for ((x, y, z, dd, jl), kk, _, c) in src
                rule.exp_generate || kk == k0 || continue
                for ((gA, gB, gC, gd), gc) in gt
                    u = (x + gA, y + gB, z + gC, dd + gd, jl, kk)
                    acc[u] = get(acc, u, zero(T)) + c * gc
                end
            end
            st = Dict{NTuple{6,Int},T}()            # (l, m, C, j, k, 0)
            for ((a, b, cc, dd, jl, kk), c) in acc
                (a < 0 || b < 0 || cc < 0 || iszero(c)) && continue
                ex = get!(() -> _r1r2_to_st(a, b), st_cache, (a, b))
                f = c / T(2)^(a + b)
                for (i, e) in ex
                    iszero(e) && continue
                    u = (dd + i, a + b - i, cc, jl, kk, 0)
                    st[u] = get(st, u, zero(T)) + f * T(e)
                end
            end
            isempty(st) && continue
            mx = maximum(abs, values(st))
            for ((l, m, cc, jl, kk, _), c) in st
                abs(c) > rule.rtol * mx || continue
                l + m + cc >= 0 && 0 <= jl <= rule.jmax || continue
                parity != 0 && selfmirror(kk) && isodd(m) != (parity < 0) && continue
                push!(get!(ms, (l + m, cc, jl, kk), Set{Int}()), m)
            end
        end
    end
    out = Key[]
    for ((p, cc, jl, kk), s) in ms
        D = maximum(s)
        s == Set(mrange(D, kk)) || @warn "collect=:st: powers of t are not closed downward (the top keys span a larger space; possible for α ≠ β)" group = (p, cc, jl, kk) ms = sort(collect(s)) maxlog = 1
        for a in 0:D
            push!(out, (a, D - a, cc, p - D, jl, kk, 0))
        end
    end
    sort!(out)
end
