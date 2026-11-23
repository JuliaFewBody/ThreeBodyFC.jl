# Particle reordering (input numbering → internal numbering).
#
# Internally the identical pair used for exchange symmetry is particles 1, 2 (`Exchange` swaps 1 ↔ 2). If the pair is
# (1,3) or (2,3) in the input, the particles are reordered internally; printing and results use the input numbering.
# π means "internal particle a = input particle π[a]". Distance indices as in subsystems.jl (1 = 1–3, 2 = 2–3, 3 = 1–2).
#
# Reordering only renames coordinates (H is the same operator), but the FC generation is centered on particle 3, so the
# function space depends on the order. Apart from the reordering required by exchange symmetry, the order is not changed.

const _PERMS = ((1, 2, 3), (2, 1, 3), (1, 3, 2), (3, 1, 2), (2, 3, 1), (3, 2, 1))

"Which input distance is internal distance d under order π (a 3-tuple)."
_distance_map(π::NTuple{3,Int}) = ntuple(3) do d
    i, j = _PAIR_PARTICLES[d]
    findfirst(p -> minmax(p...) == minmax(π[i], π[j]), _PAIR_PARTICLES)
end

"""
    permute(sys, π) / permute(es, π)

The system with particles reordered (internal a = input π[a]); powers, exponents and exponent sets follow the distances.
"""
function permute(sys::ThreeBody, π::NTuple{3,Int})
    dm = _distance_map(π)
    pw(t) = ntuple(d -> t.pow[dm[d]], 3)
    terms = map(sys.terms) do t
        t isa PowerTerm ? PowerTerm(t.c, pw(t)) : ExpTerm(t.c, pw(t), ntuple(d -> t.mu[dm[d]], 3))
    end
    ThreeBody(ntuple(a -> sys.masses[π[a]], 3), terms)
end
function permute(es::ExponentSets, π::NTuple{3,Int})
    dm = _distance_map(π)
    v = (es.α, es.β, es.γ)
    _gaussian(es) && dm[3] != 3 &&
        throw(ArgumentError("the r₁₂ Gaussian (η > 0) belongs to the internal r₁₂: put the identical pair at input particles 1, 2"))
    ExponentSets(copy(v[dm[1]]), copy(v[dm[2]]), copy(v[dm[3]]); η=copy(es.η))
end
_inverse(π::NTuple{3,Int}) = Tuple(invperm(collect(π)))

# Decimal, fraction or "Inf" strings as rationals (so "1" and "1.0" are the same mass); nothing for Inf
function _rat(s::AbstractString)
    s = strip(s)
    s == "Inf" && return nothing
    occursin('/', s) && return (x = split(s, '/'); _rat(x[1]) // _rat(x[2]))
    m = match(r"^([+-]?)(\d*)(?:\.(\d*))?(?:[eE]([+-]?\d+))?$", s)
    m === nothing && throw(ArgumentError("not a number: $s"))
    frac = something(m[3], "")
    v = parse(BigInt, string(m[2], frac, isempty(m[2]) && isempty(frac) ? "0" : "")) // big(10)^length(frac)
    m[4] === nothing || (e = parse(Int, m[4]); v *= e >= 0 ? big(10)^e : 1 // big(10)^(-e))
    m[1] == "-" ? -v : v
end

"Whether H is invariant under 1 ↔ 2 (masses and terms compared by value, not by spelling)."
function _exchange_invariant(sys::ThreeBody)
    _rat(sys.masses[1]) == _rat(sys.masses[2]) || return false
    key(t, sw) = (t isa ExpTerm, _rat(t.c), sw ? (t.pow[2], t.pow[1], t.pow[3]) : t.pow,
                  t isa ExpTerm ? (sw ? (t.mu[2], t.mu[1], t.mu[3]) : t.mu) : (0.0, 0.0, 0.0))
    sort(map(t -> key(t, false), sys.terms)) == sort(map(t -> key(t, true), sys.terms))
end

"""
    exchange_order(sys) -> π or nothing

Order that puts an exchange-symmetric identical pair at particles 1, 2: (1,2,3) if the input already is, `nothing` if there is no such pair.
"""
function exchange_order(sys::ThreeBody)
    for π in _PERMS
        _exchange_invariant(permute(sys, π)) && return π
    end
    nothing
end

"""
    orient(sys, es, symmetry; io) -> (sys′, es′, π)

With exchange symmetry, reorder (and print) if the identical pair is not at particles 1, 2; otherwise π = (1,2,3).
"""
function orient(sys::ThreeBody, es, symmetry::Symmetry; io::IO=stdout)
    symmetry isa Exchange || return sys, es, (1, 2, 3)
    _exchange_invariant(sys) && return sys, es, (1, 2, 3)
    π = exchange_order(sys)
    π === nothing && throw(ArgumentError("Exchange: no pair of particles with equal masses and symmetric potential"))
    println(io, "reordering particles: identical pair (", π[1], ", ", π[2], ") becomes internal 1, 2 (internal 1, 2, 3 = input ",
            join(π, ", "), "); exponent sets are permuted accordingly; printed thresholds use the input numbering")
    sys′ = permute(sys, π)
    sys′, es === nothing ? nothing : permute(es, π), π
end
