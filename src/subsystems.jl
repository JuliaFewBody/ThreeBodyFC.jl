# Two-body subsystems: dissociation thresholds and exponent scales (centers of the auto_exponents grid).
#
# Distance index d: 1 = r₁ (particles 1–3), 2 = r₂ (2–3), 3 = r₁₂ (1–2).
# When particle k leaves, the two distances containing k go to ∞ and the remaining pair is a bound two-body state (threshold).

const _PAIR_PARTICLES = ((1, 3), (2, 3), (1, 2))
const _OTHER_PARTICLE = (2, 1, 3)          # particle that leaves when pair d remains

"""
    PairPotential

Two-body potential Σ c r^p e^{−m r} of pair d and its reduced mass μ. `asym` is the value at r → ∞ (0, a constant, or +Inf for confinement).
"""
struct PairPotential
    d::Int
    μ::BigFloat
    terms::Vector{Tuple{BigFloat,Int,Float64}}
    asym::BigFloat
end

"""
    pair_potentials(sys) -> (pots, constant, nonpair)

Split the terms of H by pair: terms depending on one distance belong to that pair, terms with zero powers and
exponents are constants, and terms depending on two or more distances are three-body terms (ignored for thresholds and
counted in `nonpair`).
"""
function pair_potentials(sys::ThreeBody)
    setprecision(BigFloat, 256) do
        m = [exact(BigFloat, s) for s in sys.masses]
        rmass(i, j) = isinf(m[i]) ? m[j] : isinf(m[j]) ? m[i] : m[i] * m[j] / (m[i] + m[j])
        terms = [Tuple{BigFloat,Int,Float64}[] for _ in 1:3]
        constant = BigFloat(0); nonpair = 0
        for t in sys.terms
            c = exact(BigFloat, t.c)
            mu = t isa ExpTerm ? t.mu : (0.0, 0.0, 0.0)
            on = [t.pow[d] != 0 || mu[d] != 0 for d in 1:3]
            if !any(on)
                constant += c
            elseif count(on) == 1
                d = findfirst(on); push!(terms[d], (c, t.pow[d], mu[d]))
            else
                nonpair += 1
            end
        end
        pots = map(1:3) do d
            i, j = _PAIR_PARTICLES[d]
            asym = BigFloat(0)
            for (c, p, mm) in terms[d]
                mm > 0 && continue
                p < 0 && continue
                p == 0 && (asym += c; continue)
                c > 0 ? (asym = BigFloat(Inf)) : throw(ArgumentError("potential of pair $d goes to −∞ as r → ∞ (term $c r^$p)"))
            end
            PairPotential(d, rmass(i, j), terms[d], asym)
        end
        (pots, constant, nonpair)
    end
end

# matrix elements between single e^{−ζr} (without 4π): S, kinetic, potential
_pair_S(a) = 2 / a^3
_pair_T(pp::PairPotential, ζi, ζj) = ζi * ζj * (2 / (ζi + ζj)^3) / (2 * pp.μ)
function _pair_V(pp::PairPotential, a)
    v = zero(a)
    for (c, p, m) in pp.terms
        p > -3 || throw(ArgumentError("r^$p in pair $(pp.d) makes the two-body integral diverge"))
        v += c * factorial(big(2 + p)) / (a + m)^(3 + p)
    end
    v
end

"Length scale of a pair: geometric mean over terms of ζ ≈ (2μ|c|)^{1/(2+p)} balancing the kinetic energy (with m for exponential terms)."
function _pair_scale(pp::PairPotential)
    s = Float64[]
    for (c, p, m) in pp.terms
        (iszero(c) || p <= -2) && continue
        push!(s, max(Float64(2 * pp.μ * abs(c))^(1 / (2 + p)), m))
    end
    isempty(s) ? 1.0 : exp(sum(log, s) / length(s))
end

"""
    single_exponential(pp) -> (ζ, E)

Optimal ζ of a single e^{−ζr} for pair d (variational, golden section in ln ζ); ζ = μZ for −Z/r. Used as the center of
the exponent grid (also defined for confining potentials). For unbound pairs ζ sticks to the small end.
"""
function single_exponential(pp::PairPotential)
    setprecision(BigFloat, 256) do
        s = _pair_scale(pp)
        f(x) = (ζ = BigFloat(exp(x)); a = 2ζ; Float64((_pair_T(pp, ζ, ζ) + _pair_V(pp, a)) / _pair_S(a)))
        lo, hi = log(s) - 12, log(s) + 12
        g = (sqrt(5) - 1) / 2
        x1 = hi - g * (hi - lo); x2 = lo + g * (hi - lo); f1 = f(x1); f2 = f(x2)
        while hi - lo > 1e-10
            if f1 < f2
                hi, x2, f2 = x2, x1, f1; x1 = hi - g * (hi - lo); f1 = f(x1)
            else
                lo, x1, f1 = x1, x2, f2; x2 = lo + g * (hi - lo); f2 = f(x2)
            end
        end
        x = (x1 + x2) / 2
        (ζ=exp(x), E=f(x))
    end
end

"""
    pair_levels(pp; k=3, nb=90) -> (E, dE)

The lowest k S-wave levels of pair d (variational, geometric ζ of e^{−ζr}, BigFloat). `dE` is the difference between nb
and 2nb/3 functions (a convergence estimate). E is an upper bound; unbound levels come out at asym or above.
"""
function pair_levels(pp::PairPotential; k::Int=3, nb::Int=90)
    setprecision(BigFloat, 320) do
        s = _pair_scale(pp)
        function lev(N)
            ζ = [BigFloat(s) * BigFloat(10)^(-4 + 7 * (i - 1) / (N - 1)) for i in 1:N]
            S = [_pair_S(ζ[i] + ζ[j]) for i in 1:N, j in 1:N]
            H = [_pair_T(pp, ζ[i], ζ[j]) + _pair_V(pp, ζ[i] + ζ[j]) for i in 1:N, j in 1:N]
            [Float64(x.E) for x in canonical_roots(CanonicalCut(rel=1e-60), H, S, k)]
        end
        a = lev(nb); b = lev(2nb ÷ 3)
        (E=a, dE=abs.(a .- b))
    end
end

"""
    Thresholds

Dissociation thresholds. `E` is the lowest one (+Inf if no pair can dissociate); `channels` lists (leaving particle, pair level E, threshold).
"""
struct Thresholds
    E::Float64
    dE::Float64
    channels::Vector{NamedTuple{(:leaves, :pair, :Epair, :threshold),Tuple{Int,Int,Float64,Float64}}}
    nonpair::Int
end

"""
    thresholds(sys) -> Thresholds

Three-body dissociation thresholds: particle k leaves and the remaining pair is in its two-body ground state (S-wave
variational upper bound). No such channel if a pair containing the leaving particle is confining; full breakup is included.
Three-body terms (depending on two or more distances) are assumed to vanish at large distance (counted in `nonpair`).
Pure Coulomb attraction gives the exact −μ(q_iq_j)²/2; other pairs use the `pair_levels` variational value (BigFloat,
converged to 16 digits).
"""
function thresholds(sys::ThreeBody)
    key = sprint(describe, sys)          # once per system (auto_exponents calls run_fc many times for the same system)
    lock(_THRESHOLDS_LOCK) do
        get!(() -> _thresholds(sys), _THRESHOLDS, key)
    end
end

const _THRESHOLDS = Dict{String,Thresholds}()
const _THRESHOLDS_LOCK = ReentrantLock()

function _thresholds(sys::ThreeBody)
    pots, C, nonpair = pair_potentials(sys)
    ch = NamedTuple{(:leaves, :pair, :Epair, :threshold),Tuple{Int,Int,Float64,Float64}}[]
    dEmax = 0.0
    for d in 1:3
        others = setdiff(1:3, d)
        any(o -> isinf(pots[o].asym), others) && continue
        pp = pots[d]
        isinf(pp.μ) && continue
        h = _hydrogenic_pair(pp)
        E, dE = isempty(pp.terms) ? ([0.0], [0.0]) :
                h !== nothing ? ([Float64(-pp.μ * pp.terms[1][1]^2 / 2)], [0.0]) : pair_levels(pp; k=1, nb=90)
        Ep = min(E[1], Float64(pp.asym))
        push!(ch, (leaves=_OTHER_PARTICLE[d], pair=d, Epair=Ep,
                   threshold=Ep + Float64(sum(pots[o].asym for o in others) + C)))
        E[1] < Float64(pp.asym) && (dEmax = max(dEmax, dE[1]))      # pseudo-levels of unbound pairs are not counted
    end
    all(p -> isfinite(p.asym), pots) && push!(ch, (leaves=0, pair=0, Epair=0.0, threshold=Float64(sum(p.asym for p in pots) + C)))
    E = isempty(ch) ? Inf : minimum(c.threshold for c in ch)
    Thresholds(E, dEmax, ch, nonpair)
end

function describe(io::IO, th::Thresholds)
    name(d) = ("1–3", "2–3", "1–2")[d]
    if isinf(th.E)
        println(io, "thresholds: none (all particles confined, no dissociation)")
        return
    end
    @printf(io, "threshold: %.12f (", th.E)
    print(io, join([c.leaves == 0 ? @sprintf("full breakup %.6g", c.threshold) :
                    @sprintf("particle %d leaves, pair %s at %.10g", c.leaves, name(c.pair), c.Epair) for c in th.channels], ", "))
    th.dE > 1e-12 && @printf(io, ", two-body convergence %.1e", th.dE)
    th.nonpair > 0 && print(io, ", $(th.nonpair) three-body terms assumed to vanish at large distance")
    println(io, ")")
end

"""
    pair_scales(sys) -> NTuple{3,Float64}

Optimal single exponent ζ per pair (`single_exponential`); NaN for unbound pairs (ζ at the small end) or pairs without terms.
Centers of the auto_exponents grid for α (pair 1–3), β (pair 2–3), γ (pair 1–2); exactly μ|q₁q₂| for pure Coulomb attraction.
"""
function pair_scales(sys::ThreeBody)
    pots, _, _ = pair_potentials(sys)
    Tuple(map(pots) do pp
        isempty(pp.terms) && return NaN
        isinf(pp.μ) && return NaN
        h = _hydrogenic_pair(pp)
        h === nothing || return h
        z = single_exponential(pp)
        (z.E < Float64(pp.asym) && z.ζ > 1e-4 * _pair_scale(pp)) ? z.ζ : NaN
    end)
end

# pure Coulomb attraction c/r (c < 0): ζ = μ|c| exactly
function _hydrogenic_pair(pp::PairPotential)
    (length(pp.terms) == 1 && pp.terms[1][2] == -1 && pp.terms[1][3] == 0 && pp.terms[1][1] < 0) || return nothing
    Float64(pp.μ * -pp.terms[1][1])
end

"""
    tail_exponents(sys, E; centers, usegamma=true) -> Vector{NTuple{3,Float64}}

If E (a three-body energy, e.g. an extrapolated E∞) is below the dissociation threshold, exponent sets (α, β, γ) matching the
asymptotic decay κ = √(2μ_p (E_th − E)) of each lowest-threshold channel (particle p leaving the remaining pair; μ_p is the
reduced mass of p and the pair). The two distances containing p both approach the distance R of p from the pair, so their
exponents sum to κ (half each, or 0 for r₁₂ and κ for the other). The exponent of the remaining pair is `centers` (its grid
center; 0 if γ is unused). Used as auto_exponents candidates for weakly bound states near threshold and Rydberg-like states.
"""
function tail_exponents(sys::ThreeBody, E::Real; centers::NTuple{3,Float64}, usegamma::Bool=true)
    th = thresholds(sys)
    out = NTuple{3,Float64}[]
    (isfinite(th.E) && E < th.E) || return out
    m = setprecision(() -> [exact(BigFloat, s) for s in sys.masses], BigFloat, 256)
    for c in th.channels
        (c.leaves == 0 || c.threshold > th.E + 1e-12 * max(1, abs(th.E))) && continue
        p = c.leaves; i, j = _PAIR_PARTICLES[c.pair]
        mp, mpair = m[p], m[i] + m[j]
        μ = isinf(mp) ? Float64(mpair) : isinf(mpair) ? Float64(mp) : Float64(mp * mpair / (mp + mpair))
        κ = sqrt(2μ * (c.threshold - E))
        x = zeros(3)
        x[c.pair] = (c.pair == 3 && !usegamma) ? 0.0 : centers[c.pair]
        ds = [d for d in 1:3 if p in _PAIR_PARTICLES[d]]
        if 3 in ds                                     # p is particle 1 or 2: r₁₂ and the distance of p from particle 3
            d = ds[1] == 3 ? ds[2] : ds[1]
            y = copy(x); y[d] = κ; push!(out, Tuple(y))                # exponent 0 on r₁₂, κ on the p–3 distance
            usegamma && (y = copy(x); y[d] = κ / 2; y[3] = κ / 2; push!(out, Tuple(y)))
        else                                           # p is particle 3: half each on r₁ and r₂
            y = copy(x); y[1] = κ / 2; y[2] = κ / 2; push!(out, Tuple(y))
        end
    end
    unique!([_dyadic.(t) for t in out])
end
