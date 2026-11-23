# Automatic choice of exponent sets: add sets one at a time by their variational gain at a small stage (greedy).

"""
    exponent_grid(sys, d; nstates=1) -> (center, values, kind)

Candidate grid of exponents for distance d (1 = r₁, 2 = r₂, 3 = r₁₂), centered from the two-body subsystem (`subsystems.jl`):
- pairs bound by a single e^{−ζr} (Coulomb-like): the optimal ζ (μ|q_iq_j| for pure Coulomb attraction);
- weakly bound pairs not captured by one exponent: geometric mean of the asymptotic decay κ = √(2μ(V∞ − E₀)) and the potential scale;
- unbound pairs: the potential scale (`_pair_scale`), extended far downward;
- purely repulsive pairs: negative values −c·2^{−k/2} (k = 0..8, `kind = :repulsive`), pulling the pair apart; c is the smaller
  center of the other two (bound) pairs, or the pair's own scale.
The grid is center × 2^{−k/2} (k = −2..6; −2..12 for unbound pairs). `nothing` for pairs without terms.
"""
function exponent_grid(sys::ThreeBody, d::Int; nstates::Int=1)
    pp = pair_potentials(sys)[1][d]
    (isempty(pp.terms) || isinf(pp.μ)) && return nothing
    scale = _pair_scale(pp)
    h = _hydrogenic_pair(pp)
    z = h === nothing ? single_exponential(pp) : (ζ=h, E=-Inf)
    single_ok = h !== nothing || (z.E < Float64(pp.asym) && z.ζ > 1e-4 * scale)
    if single_ok
        c, kind, ks = z.ζ, :single, -2:6
    else
        E0 = isfinite(pp.asym) ? pair_levels(pp; k=1).E[1] : -Inf
        if isfinite(pp.asym) && E0 < Float64(pp.asym) - 1e-12
            κ = sqrt(2 * Float64(pp.μ) * (Float64(pp.asym) - E0))
            c, kind, ks = sqrt(κ * scale), :shallow, -2:8
        elseif all(t -> t[1] > 0, pp.terms)
            # purely repulsive (μ⁻ and e⁻, p and p in H₂⁺): negative exponents (e^{+|γ|r}). Normalizability needs the other two
            # decays to compensate (α + γ > 0, β + γ > 0), so the scale is the smaller center of the other (bound) pairs
            others = [exponent_grid(sys, e; nstates=nstates) for e in 1:3 if e != d]
            cs = [g.center for g in others if g !== nothing && g.kind in (:single, :shallow)]
            c = isempty(cs) ? scale : minimum(cs)
            return (center=c, values=[_dyadic(-c * 2.0^(-k / 2)) for k in 0:8], kind=:repulsive)
        else
            c, kind, ks = scale, :unbound, -2:12
        end
    end
    (center=c, values=[_dyadic(c * 2.0^(-k / 2)) for k in ks], kind=kind)
end

# Candidate classes for pruning in `auto_exponents`: single sets by the sign of γ, tail sets, and families by direction × up/down.
# Grid index of a single set k_d = −2 log₂(|x_d| / |center_d|) (grid = center × 2^{−k/2}). Coarse representatives of single
# sets are the checkerboard with even Σk (every point is one step from a coarse one); of families, lengths 2, 4, 8 and the longest.
function _cand_meta(c::Vector{NTuple{3,Float64}}, extra, cent, exch)
    kof(x, d) = (x == 0 || cent[d] == 0) ? 0 : round(Int, -2 * log2(abs(x) / abs(cent[d])))
    if length(c) == 1
        t = c[1]
        tail = any(e -> e == t || (exch && (e[2], e[1], e[3]) == t), extra)
        cls = tail ? :tail : t[3] == 0 ? :gamma0 : t[3] > 0 ? :gamma_pos : :gamma_neg
        k = (kof(t[1], 1), kof(t[2], 2), kof(t[3], 3))
        return (cls=cls, k=k, fam=nothing, len=1, coarse=tail || iseven(sum(k)))
    end
    d = findfirst(i -> c[1][i] != c[2][i], 1:3)
    up = abs(c[2][d]) > abs(c[1][d])
    root = Tuple(i == d ? 0.0 : c[1][i] for i in 1:3)
    len = length(c)
    (cls=Symbol("family_", ("a", "b", "g")[d], up ? "_up" : "_down"), k=nothing, fam=(root, d, up, c[1][d]), len=len,
     coarse=len in (2, 4, 8))
end
# neighbors: single sets within one grid step in every coordinate; families with the same root, direction and start, length ±1
_cand_near(a, b) = a.cls == b.cls && (a.fam === nothing ? all(abs.(a.k .- b.k) .<= 1) : a.fam == b.fam && abs(a.len - b.len) <= 1)

"""
    exponent_window(sys) -> ((lo₁, hi₁), (lo₂, hi₂), (lo₃, hi₃))

Exponent window (absolute values) per coordinate; families (`families` of `auto_exponents`) extend to its ends.
- upper end: twice the largest grid center of the bound pairs (`:single`/`:shallow` of `exponent_grid`), the same for all coordinates;
- lower end: the coordinate's smallest grid value ÷ 4 (far decay; tail sets cover smaller κ of excited or weakly bound states).
(0, 0) for coordinates without terms.
"""
function exponent_window(sys::ThreeBody)
    gs = [exponent_grid(sys, d) for d in 1:3]
    cs = [g.center for g in gs if g !== nothing && g.kind in (:single, :shallow)]
    hi = 2 * (isempty(cs) ? maximum(g.center for g in gs if g !== nothing) : maximum(cs))
    Tuple(g === nothing ? (0.0, 0.0) : (minimum(abs, g.values) / 4, hi) for g in gs)
end

"""
    auto_exponents(sys; n=3, nstates=1, symmetry=nothing, maxsets=8, gamma=:auto, objective=:logerr,
                   start=nothing, io=stdout, kwargs...) -> (es, history)

Choose exponent sets (α, β, γ) one at a time at stage `n` (greedy).

- Candidates: the grids of α, β, γ (`exponent_grid`). `space = :full` (default) uses their product (γ whenever the 1–2 pair has
  terms); `:v3` uses all (α, β, 0) plus γ added to the chosen (α, β), with γ only for bound or purely repulsive 1–2 pairs
  (`gamma = :auto`; `true`/`false` force it). With exchange symmetry (a, b) and (b, a) span the same space, so only a ≥ b.
  A chosen value at the grid edge extends the grid by two steps. Tail sets (`tail_exponents`) match the asymptotic decay
  κ = √(2μ(E_th − E∞)) of each state whose extrapolated E∞ is below the dissociation threshold.
- Objective (`objective = :logerr`, default): sum over states of the log of the error relative to E∞ extrapolated geometrically
  from stages n, n+1, n+2 of the current sets; states above the threshold are excluded. `:energy` uses the sum of the lowest
  `nstates` energies.
- Gain of a set = decrease of the objective ÷ increase of the symmetrized dimension M. Stop when it falls below the gain of raising
  the stage n → n+1 with the current sets, or at `maxsets` sets.
- Candidates that do not take the Cholesky path in the working precision (Float64x4 by default) are re-solved in BigFloat
  `bigbits` bits (fallback energies are upper bounds but understate the gain).
- `budget = M_b`: instead of stopping, raise the stage and compare candidates again, until M at stage n+1 reaches M_b or `maxsets`.
  The decision is always the local gain at the current M (no extrapolation).
- `exhaustive = false` (default): evaluate the coarse representatives of each candidate class, then refine around the top three
  of the classes whose best gain is at least `keep_ratio` times the overall best, until the best stops moving. `exhaustive = true`
  evaluates all candidates.
- `lazy = true`: use each candidate's gain from the previous round as an upper bound and stop when it falls below the best gain
  (gains are not quite additive, so off by default).
- `families = true`: also offer families, one exponent of a chosen set (α, β, nonzero γ) in geometric progression with ratio
  `famratio` to the window end (`exponent_window`), lengths 2..`famlen`, added in one move. `chosen` in `history` lists the sets added.
- `lookahead = K` (default 0, off): before stopping, try pairs of the top K and top 2K candidates added together; if a pair beats
  raising the stage, add it and continue (gains are not additive: a set useless alone can help together with another).
- `start` (`ExponentSets`): starting sets; default one set (α, β, 0) at the grid centers. If that gives fewer than `nstates`
  roots at stage `n`, sets with β × 1/4, 1/8, … are added until it does (printed). The integral memory is shared by candidates.

Identical pairs not at input particles 1, 2 are reordered internally (`orient`); the returned sets and `history` use the input
numbering. This changes the function space, so it is not a `run_fc` default; compare the chosen sets with others at the same M
at large stages before using them. Other keywords go to `run_fc`. `history` has one row per round; `ranking` lists the top
`nrank` candidates of the round (`cand`, `gain`, `M`, `Es`), for guidance only (the top 5 are printed).
"""
function auto_exponents(sys::ThreeBody; n::Int=3, nstates::Int=1, symmetry=nothing, maxsets::Int=8, gamma=:auto,
                        objective::Symbol=:logerr, start=nothing, io::IO=stdout, space::Symbol=:full, lazy::Bool=false,
                        bigbits::Int=256, budget=nothing, families::Bool=true, famratio::Float64=2.0, famlen::Int=8,
                        exhaustive::Bool=false, keep_ratio::Float64=0.1, nrank::Int=10, lookahead::Int=0,
                        cache_cap::Int=20_000_000, memdiag::Int=0, kwargs...)
    space in (:full, :v3) || throw(ArgumentError("space must be :full or :v3"))
    budget === nothing || objective === :logerr || throw(ArgumentError("budget requires objective = :logerr"))
    sym = symmetry === nothing ? default_symmetry(sys) : symmetry
    exch = sym isa Exchange
    sys0 = sys
    sys, start, π = orient(sys, start, sym; io=io)      # reorder internally if the identical pair is not at 1, 2 (returned sets use the input numbering)
    th = thresholds(sys)
    g1 = exponent_grid(sys, 1; nstates=nstates); g2 = exch ? g1 : exponent_grid(sys, 2; nstates=nstates)
    g3 = exponent_grid(sys, 3; nstates=nstates)
    (g1 === nothing || g2 === nothing) && throw(ArgumentError("no exponent grid: pair 1–3 or 2–3 has no terms"))
    # γ: v3 only for bound or purely repulsive 1–2 pairs; full whenever the pair has terms
    usegamma = gamma === :auto ? (g3 !== nothing && (space === :full || g3.kind !== :unbound)) : Bool(gamma)
    usegamma && g3 === nothing && throw(ArgumentError("no γ grid: pair 1–2 has no terms"))
    G = [copy(g1.values), copy(g2.values), usegamma ? copy(g3.values) : Float64[]]
    sets = start === nothing ? [(_dyadic(g1.center), _dyadic(g2.center), 0.0)] : collect(zip(start.α, start.β, start.γ))
    mk(ps) = ExponentSets(getindex.(ps, 1), getindex.(ps, 2), getindex.(ps, 3))
    kw = Dict{Symbol,Any}(kwargs)
    ctx = get(kw, :auto_precision, false) ? nothing :
          IntegralContext(get(kw, :T, Float64x4), get(kw, :engine, KernelQuadrature()), get(kw, :logvar, LogVariable());
                          kernel_cap=cache_cap, elem_cap=cache_cap)
    # Candidates whose S is intractable in the working precision are re-solved in BigFloat bigbits bits instead of the dense
    # fallback (much faster; fallback energies are subspace upper bounds and understate the gain of diffuse candidates)
    ctxbig = Ref{Any}(Dict{Int,Any}()); nbig = Ref(0)          # BigFloat contexts per bit count
    # candidate evaluation raises the precision by its own rule (to share the integral memory between candidates); run_fc's
    # automatic raise is used only when auto_precision = true is passed (ctx = nothing)
    kwr = (; (k => v for (k, v) in pairs(kwargs) if k != :auto_precision)...)
    clean(r) = r.solution.method === :cholesky && all(x -> x.method in (:cholesky, :cholesky_lift), r.states)
    # memdiag = k > 0: print integral memory entries and GC live size every k evaluations (diagnostic only)
    nevall = Ref(0)
    cnt(c) = c === nothing ? "—" : string(c.store.count[], "/", c.elems.count[], "/", c.angys.count[], "/", c.dpows.count[], "/",
                                          c.epsgs.count[], "/", c.monos.count[], "/", length(c.abgrids))
    function memline()
        rss = try parse(Int, split(first(filter(l -> startswith(l, "RssAnon"), readlines("/proc/self/status"))))[2]) / 2^20 catch; NaN end
        l0 = Base.gc_live_bytes() / 2^30
        GC.gc(); l1 = Base.gc_live_bytes() / 2^30
        # threads keep their last task (and its closure's storage) alive; an empty @threads :static drops it
        Threads.@threads :static for _ in 1:Threads.nthreads(); end
        GC.gc(); l2 = Base.gc_live_bytes() / 2^30
        sz(c) = c === nothing ? 0.0 : (Base.summarysize(c.store.shards) + Base.summarysize(c.elems.shards) + Base.summarysize(c.angys.shards) +
                                       Base.summarysize(c.dpows.shards) + Base.summarysize(c.epsgs.shards)) / 2^30
        @printf(io, "    [memory] eval %d: kernels/elements/angys/dpows/epsgs/monos/abgrids = %s (BigFloat %s), memory size %.2f + %.2f GB, live %.2f → after GC %.2f → after dropping thread refs %.2f GB, RssAnon %.2f GB\n",
                nevall[], cnt(ctx), join([cnt(c) for c in values(ctxbig[])], ","), sz(ctx), sum(sz, values(ctxbig[]); init=0.0), l0, l1, l2, rss); flush(io)
    end
    # (set collection, stage) that did not take the Cholesky path in the working precision. A superset at a stage ≥ that one contains
    # the same principal submatrix (λmin only decreases), so it will not either: skip the working-precision attempt and use BigFloat
    # directly (the working-precision result was discarded anyway).
    x4fail = Tuple{Set{NTuple{3,Float64}},Int}[]
    nskip = Ref(0)
    function evaluate(ps, m)
        nevall[] += 1
        memdiag > 0 && nevall[] % memdiag == 0 && memline()
        # with memdiag > 0, also print the sets and stage of each candidate (the memory line, with two GCs, only every k evaluations)
        memdiag > 0 && (println(io, "    [candidate] eval ", nevall[], " stage ", m, " sets ", ps); flush(io))
        bigT = get(kw, :T, Float64x4) === BigFloat
        skipx4 = ctx !== nothing && !bigT && any(f -> f[2] <= m && issubset(f[1], ps), x4fail)
        skipx4 && (nskip[] += 1)
        r = try
            r1 = nothing
            if !skipx4
                r1v = run_fc(sys, mk(ps); stages=m:m, nstates=nstates, symmetry=sym, io=devnull, context=ctx, threshold=th,
                             auto_precision=ctx === nothing, skip_fallback=(ctx !== nothing && !bigT), kwr...)
                # also re-solve when run_fc returned nothing because a stage failed in the working precision
                r1 = isempty(r1v) ? nothing : r1v[end]
                r1 === nothing && (ctx === nothing || bigT) && return nothing
                ctx !== nothing && !bigT && (r1 === nothing || !clean(r1)) && push!(x4fail, (Set(ps), m))
            end
            if ctx !== nothing && !bigT && (r1 === nothing || !clean(r1))
                nbig[] += 1
                # if the re-solve does not take the Cholesky path either, double the bits (up to 4×); never accept a non-Cholesky
                # result (a pivoted fallback can give energies far below the true E when the precision is insufficient)
                kb = (; (k => v for (k, v) in pairs(kwr) if !(k in (:T, :engine)))...)
                b = bigbits; rb = nothing
                while true
                    rb = setprecision(BigFloat, b) do
                        c = get!(() -> IntegralContext(BigFloat, KernelQuadrature(BigFloat), get(kw, :logvar, LogVariable());
                                                       kernel_cap=cache_cap, elem_cap=cache_cap), ctxbig[], b)
                        v = run_fc(sys, mk(ps); stages=m:m, nstates=nstates, symmetry=sym, io=devnull, context=c, threshold=th,
                                   auto_precision=false, skip_fallback=true, T=BigFloat, engine=KernelQuadrature(BigFloat), kb...)
                        isempty(v) ? nothing : v[end]
                    end
                    (rb !== nothing && clean(rb)) && break
                    b >= 4bigbits && return nothing
                    b *= 2
                end
                rb
            else
                r1
            end
        catch e
            e isa InterruptException && rethrow()
            return nothing
        end
        Es = [Float64(_tobig(x.E)) for x in (nstates == 1 ? [r.solution] : r.states)]
        # differences in the working precision (deep-stage differences are below the Float64 rounding of E)
        Eb = [_tobig(x.E) for x in (nstates == 1 ? [r.solution] : r.states)]
        length(Es) < nstates && return nothing
        (Es=Es, Eb=Eb, M=r.Msolved)
    end
    # Same set: every exponent within a ratio 2^{1/8} (γ with the same sign, or both 0). Neighbors on the grid (ratio √2) remain
    # different sets. Exact equality alone would let a grid point next to a tuned start set in, and near dependence would hit
    # the precision wall.
    near1(x, y) = x == y || (x != 0 && y != 0 && sign(x) == sign(y) && abs(log(abs(x / y))) < log(2) / 8)
    near(c, s) = near1(c[1], s[1]) && near1(c[2], s[2]) && near1(c[3], s[3])
    same(c, s) = near(c, s) || (exch && near((c[2], c[1], c[3]), s))
    # Cost gate: for split log variables, skip sets whose self pair (2α, 2β, 2γ) needs an (x,y) grid of more than 4096 nodes
    # (`_split_nodes`; printed once per set); such grids cost tens of GB per evaluation.
    lvs = get(kw, :logvar, LogVariable()); digs = -log10(Float64(eps(get(kw, :T, Float64x4))))
    heavyseen = Set{NTuple{3,Float64}}()
    function light(c)
        lvs.split || return true
        m = _split_nodes(2c[1], 2c[2], 2c[3], 80, digs)
        m <= 4096 && return true
        c in heavyseen || (push!(heavyseen, c); @printf(io, "  cost gate: set (%.6g, %.6g, %.6g) needs a split grid of %d nodes; not a candidate\n", c..., m); flush(io))
        false
    end
    valid(c) = (c[2] + c[3] > 0 && c[1] + c[3] > 0 && c[1] + c[2] > 0) && light(c)
    cent = (g1.center, g2.center, g3 === nothing ? 0.0 : g3.center)
    W = exponent_window(sys)
    canon(c) = exch && c[1] < c[2] ? (c[2], c[1], c[3]) : c
    function candidates(extra)
        cs = NTuple{3,Float64}[]
        for a in G[1], b in G[2]
            exch && a < b && continue
            push!(cs, (a, b, 0.0))
            if usegamma && space === :full              # product of the three grids
                for g in G[3]; push!(cs, (a, b, g)); end
            end
        end
        if usegamma && space === :v3
            for s in sets, g in G[3]
                push!(cs, (s[1], s[2], g))
            end
        end
        for c in extra                                   # tail sets (far decay of states below threshold, `tail_exponents`)
            push!(cs, canon(c))
        end
        unique!(cs)
        out = [[c] for c in cs if valid(c) && !any(s -> same(c, s), sets)]
        families && append!(out, family_candidates())
        filter(c -> length(sets) + length(c) <= maxsets, out)
    end
    # Families (even-tempered): one exponent of a chosen set in geometric progression with ratio famratio to the window end
    # (`exponent_window`), added in one move; the one-at-a-time greedy cannot pick sequences that gain nothing at small M.
    # Lengths 2..famlen; γ (r₁₂) only if nonzero in the chosen set (sign kept)
    function family_candidates()
        out = Vector{NTuple{3,Float64}}[]
        for s in sets, d in 1:3, up in (true, false)
            d == 3 && (!usegamma || s[3] == 0) && continue
            lo, hi = W[d]
            sg = sign(s[d]); x = abs(s[d]); fam = NTuple{3,Float64}[]
            for _ in 1:famlen
                x = up ? x * famratio : x / famratio
                (x > hi || x < lo) && break
                y = collect(s); y[d] = _dyadic(sg * x); c = canon(Tuple(y))
                valid(c) || break
                any(t -> same(c, t), sets) || push!(fam, c)
                length(fam) >= 2 && push!(out, copy(fam))
            end
        end
        unique!(out)
    end
    function extend!(cv)
        length(cv) == 1 || return                        # families do not extend the grid (they already reach the window end)
        c = cv[1]
        for (d, x) in ((1, c[1]), (2, c[2]), (3, c[3]))
            v = G[d]; (isempty(v) || x == 0) && continue
            # sort by absolute value and extend at the ends (γ grids of repulsive pairs have negative values only)
            sg = sign(x); a = abs(x); m = sort(abs.(v); rev=true)
            a == m[1] && (m = [_dyadic(2a); _dyadic(sqrt(2.0) * a); m])
            a == m[end] && (m = [m; _dyadic(a / sqrt(2.0)); _dyadic(a / 2)])
            G[d] = sg .* m
            exch && d == 1 && (G[2] = G[1])
        end
    end
    println(io, "auto_exponents: stage n=", n, "  nstates=", nstates, "  objective ", objective, "  symmetry ", exch ? "exchange" : "none")
    @printf(io, "  grid centers α %.6g (%s)  β %.6g (%s)  γ %s\n", g1.center, g1.kind, g2.center, g2.kind,
            usegamma ? @sprintf("%.6g (%s)", g3.center, g3.kind) : "none")
    print(io, "  "); describe(io, thresholds(sys0))
    t0 = time()
    cur = evaluate(sets, n)
    # The default single start set may not give the higher roots at stage n; add sets with the decay of the outer particle 2
    # scaled by 1/4, 1/8, … one at a time until they are solved (up to nstates + 1 sets). A given `start` is not changed.
    if cur === nothing && start === nothing
        a0, b0 = sets[1][1], sets[1][2]
        for j in 1:nstates
            push!(sets, (a0, _dyadic(b0 / 2.0^(j + 1)), 0.0))
            cur = evaluate(sets, n)
            cur === nothing || (println(io, "  stage $n not solved with one start set; added widened sets: ", sets); break)
        end
    end
    cur === nothing && error("stage $n cannot be solved with start sets $(sets)")
    tfirst = time() - t0
    @printf(io, "  start %s  E=%s  M=%d\n", sets, join([@sprintf("%.12f", e) for e in cur.Es], " "), cur.M); flush(io)
    history = Any[]
    lastgain = Dict{Vector{NTuple{3,Float64}},Float64}()   # gain of each candidate at its last evaluation (bound for lazy evaluation)
    while length(sets) < maxsets
        up = evaluate(sets, n + 1)
        up2 = objective === :logerr ? evaluate(sets, n + 2) : nothing
        if up === nothing || (objective === :logerr && up2 === nothing)
            println(io, "  stop: stages n+1, n+2 cannot be solved with the current sets"); break
        end
        Etail = Float64[]                                # energies for tail sets (extrapolated E∞, else the stage n+1 E)
        if objective === :logerr
            active = [e < th.E for e in up2.Es]
            any(active) || (active .= true)
            Einf = map(1:nstates) do k
                Δ1 = up.Eb[k] - up2.Eb[k]; Δ0 = cur.Eb[k] - up.Eb[k]
                t = tail_estimate(Float64(Δ1), Float64(Δ0))
                up2.Eb[k] - (isnan(t) ? max(Δ1, 0) : t)
            end
            floor_ = [max(Float64(up2.Eb[k] - Einf[k]) / 10, 1e-300) for k in 1:nstates]
            Etail = [Float64(Einf[k]) for k in 1:nstates if active[k]]
            F(o) = sum(log(max(Float64(o.Eb[k] - Einf[k]), floor_[k])) for k in 1:nstates if active[k])
            atfloor = o -> any(k -> active[k] && Float64(o.Eb[k] - Einf[k]) < floor_[k], 1:nstates)
        else
            F = o -> Float64(sum(o.Eb))
            atfloor = o -> false
            Etail = up.Es
        end
        extra = unique!(reduce(vcat, [tail_exponents(sys, e; centers=cent, usegamma=usegamma) for e in Etail];
                               init=NTuple{3,Float64}[]))
        Fcur = F(cur)
        # Equal gains (candidates at the floor, same M) are broken by the lower E (variational: lower E with as many functions is better),
        # not by evaluation order.
        beats(g, o, b) = b === nothing || g > b[2] || (g == b[2] && sum(o.Eb) < sum(b[3].Eb))
        nfloor = Ref(0)
        sg = (Fcur - F(up)) / max(up.M - cur.M, 1)
        best = nothing; nev = 0; tround = time()
        seen = Dict{Vector{NTuple{3,Float64}},Tuple{Float64,Int,Vector{Float64}}}()   # candidates evaluated this round (gain, M, E)
        cands = candidates(extra)
        if isempty(history)
            # time per evaluation (the start set at stage n ran with empty integral memory, so this overestimates)
            @printf(io, "  %d candidates, start set at stage n took %.1f s (M=%d) → about %.0f min per round; lower n if too slow\n",
                    length(cands), tfirst, cur.M, length(cands) * tfirst / 60); flush(io)
        end
        # Lazy evaluation: gains shrink as sets are added, so the previous gain bounds the current one. Evaluate by decreasing bound and
        # stop when the next bound is below the best gain so far (new candidates have bound +Inf and are always evaluated)
        nbig0 = nbig[]; nskip0 = nskip[]
        if exhaustive || lazy
            order = lazy ? sort(cands; by=c -> -get(lastgain, c, Inf)) : cands
            for c in order
                lazy && best !== nothing && get(lastgain, c, Inf) <= best[2] && break
                o = evaluate([sets; c], n); nev += 1
                if o === nothing
                    lastgain[c] = -Inf; continue
                end
                g = (Fcur - F(o)) / max(o.M - cur.M, 1)
                lastgain[c] = g; seen[c] = (g, o.M, o.Es); atfloor(o) && (nfloor[] += 1)
                beats(g, o, best) && (best = (c, g, o))
            end
            prunenote = ""
        else
            # coarse representatives per class, then refinement near the best of promising classes (`exhaustive = false`)
            gains = Dict{Vector{NTuple{3,Float64}},Float64}()
            function tryc(c)
                haskey(gains, c) && return gains[c]
                o = evaluate([sets; c], n); nev += 1
                g = o === nothing ? -Inf : (Fcur - F(o)) / max(o.M - cur.M, 1)
                gains[c] = g; o === nothing || (seen[c] = (g, o.M, o.Es); atfloor(o) && (nfloor[] += 1))
                o !== nothing && beats(g, o, best) && (best = (c, g, o))
                g
            end
            Esum(c) = haskey(seen, c) ? sum(seen[c][3]) : Inf
            meta = Dict(c => _cand_meta(c, extra, cent, exch) for c in cands)
            classes = unique(m.cls for m in values(meta))
            for c in cands                                    # 1. coarse representatives (all classes)
                meta[c].coarse && tryc(c)
            end
            ncoarse = nev
            clsbest = Dict(k => maximum((gains[c] for c in cands if meta[c].cls == k && haskey(gains, c)); init=-Inf) for k in classes)
            gall = maximum(values(clsbest); init=-Inf)
            # 2. promising classes (best gain ≥ keep_ratio × overall best; only the best class if the overall best is ≤ 0)
            keep = [k for k in classes if isfinite(clsbest[k]) &&
                    (gall > 0 ? clsbest[k] >= keep_ratio * gall : clsbest[k] == gall)]
            for k in keep
                # from the class's top three, evaluate neighbors (one grid step, family length ±1) until the best stops moving
                seeds = sort([c for c in cands if meta[c].cls == k && haskey(gains, c)]; by=c -> (-gains[c], Esum(c)))[1:min(3, end)]
                for s0 in seeds
                    c0 = s0
                    for _ in 1:4
                        nb = [c for c in cands if meta[c].cls == k && !haskey(gains, c) && _cand_near(meta[c], meta[c0])]
                        isempty(nb) && break
                        for c in nb; tryc(c); end
                        cb = argmin(c -> (-get(gains, c, -Inf), Esum(c)), [c0; nb])
                        cb == c0 && break
                        c0 = cb
                    end
                end
            end
            dropped = setdiff(classes, keep)
            prunenote = @sprintf(", coarse %d, fine %d, dropped classes %s", ncoarse, nev - ncoarse,
                                 isempty(dropped) ? "none" : join(string.(sort(dropped)), ","))
        end
        stop = best === nothing || best[2] <= sg
        # Two-step lookahead (`lookahead = K`): if adding a pair from the top K and top 2K candidates beats raising the stage, add it
        # and continue (set gains are not additive)
        pairnote = ""
        if stop && lookahead > 0 && length(seen) >= 2
            top = [c for (c, _) in sort(collect(seen); by=x -> (-x[2][1], sum(x[2][3])))]
            np = 0; bestp = nothing
            for i in 1:min(lookahead, length(top)), j in i+1:min(2lookahead, length(top))
                cp = unique!(vcat(top[i], top[j]))
                (length(sets) + length(cp) > maxsets || any(c -> any(s -> same(c, s), sets), cp)) && continue
                o = evaluate([sets; cp], n); nev += 1; np += 1
                o === nothing && continue
                g = (Fcur - F(o)) / max(o.M - cur.M, 1)
                beats(g, o, bestp) && (bestp = (cp, g, o, top[i], top[j]))
            end
            if bestp !== nothing
                pairnote = @sprintf(", best of %d pairs %.3e", np, bestp[2])
                if bestp[2] > sg
                    best = bestp[1:3]; stop = false
                    extend!(bestp[4]); extend!(bestp[5])
                end
            end
            println(io, "    lookahead: best single ", @sprintf("%.3e", isempty(top) ? NaN : seen[top[1]][1]), " ≤ gain of raising the stage ",
                    @sprintf("%.3e", sg), pairnote, stop ? " → pairs lose too" : " → adding the pair"); flush(io)
        end
        # ranking of promising candidates (not used for the choice; guidance for manual tuning): top nrank by gain
        ranking = [(cand=c, gain=v[1], M=v[2], Es=v[3]) for (c, v) in sort(collect(seen); by=x -> (-x[2][1], sum(x[2][3])))[1:min(nrank, end)]]
        if !isempty(ranking)
            rl(r) = length(r.cand) == 1 ? @sprintf("(%.4g, %.4g, %.4g) %.2e", r.cand[1]..., r.gain) :
                    @sprintf("family%d(%.4g, %.4g, %.4g…) %.2e", length(r.cand), r.cand[1]..., r.gain)
            println(io, "    promising (by gain): ", join(rl.(ranking[1:min(5, end)]), ", ")); flush(io)
        end
        if stop && budget !== nothing && up.M < budget
            # raising the stage pays more: raise it and compare candidates again at the larger stage (up to the target M)
            n += 1; cur = up
            push!(history, (sets=copy(sets), Es=cur.Es, M=cur.M, chosen=:stage, gain=best === nothing ? NaN : best[2],
                            stage_gain=sg, evals=nev, ranking=ranking))
            @printf(io, "  ↑ stage n=%d  E=%s  M=%d  (best gain %.3e ≤ gain of raising the stage %.3e, %d evals%s, %.0f min)\n", n,
                    join([@sprintf("%.12f", e) for e in cur.Es], " "), cur.M, best === nothing ? NaN : best[2], sg, nev,
                    prunenote, (time() - tround) / 60); flush(io)
            continue
        end
        if stop
            @printf(io, "  stop: best gain %.3e ≤ %s\n", best === nothing ? NaN : best[2],
                    @sprintf("gain of raising the stage %.3e (per function)%s", sg, budget === nothing ? "" : ", M at stage n+1 ≥ budget $(budget)"))
            push!(history, (sets=copy(sets), Es=cur.Es, M=cur.M, chosen=nothing, gain=best === nothing ? NaN : best[2],
                            stage_gain=sg, evals=nev, ranking=ranking))
            break
        end
        c, g, o = best
        append!(sets, c); cur = o; extend!(c)
        push!(history, (sets=copy(sets), Es=cur.Es, M=cur.M, chosen=c, gain=g, stage_gain=sg, evals=nev, ranking=ranking))
        istail = length(c) == 1 && any(t -> same(c[1], t) || c[1] == t, extra)
        lab = length(c) == 1 ? @sprintf("(%.6g, %.6g, %.6g)", c[1]...) :
              @sprintf("family of %d (%.6g, %.6g, %.6g) … (%.6g, %.6g, %.6g)", length(c), c[1]..., c[end]...)
        @printf(io, "  + %s%s  E=%s  M=%d  gain %.3e  gain of raising the stage %.3e  (%d evals / %d candidates%s, %d re-solved in BigFloat (%d skipped the working precision), %d tail candidates, %d at the floor (ties by E), %.0f min)\n",
                lab, istail ? " tail" : "", join([@sprintf("%.12f", e) for e in cur.Es], " "), cur.M, g, sg, nev,
                length(cands), prunenote, nbig[] - nbig0, nskip[] - nskip0, length(extra), nfloor[], (time() - tround) / 60); flush(io)
    end
    # internal → input numbering (unchanged if not reordered)
    πi = _inverse(π)
    back(ps) = (e = permute(mk(ps), πi); collect(zip(e.α, e.β, e.γ)))
    if π != (1, 2, 3)
        history = map(h -> merge(h, (sets=back(h.sets), chosen=h.chosen isa Vector ? back(h.chosen) : h.chosen,
                                     ranking=[merge(r, (cand=back(r.cand),)) for r in h.ranking])), history)
    end
    (es=permute(mk(sets), πi), history=history)
end

"""
    w2_rule(sys; coef=0.85) -> (w2, m, inrange)

Rule of thumb for w₂ in σ = r₁ + w₂ r₂: w₂ ≈ coef · (m₂/m₁)^{1/4} (m₁, m₂ masses of particles 1, 2). Fitted on families of good
bases (α = Z·m_red, β ∈ {1,2,4}·Z/2) for 4 ≤ m₁/m₂ ≤ 207; nearly independent of the nucleus (Z = 2–4) and the stage (4–8).
`inrange` tells whether 4 ≤ m₁/m₂ ≤ 207 (values outside are unreliable). Useful as a start or center for `auto_w2`.
"""
function w2_rule(sys::ThreeBody; coef::Real=0.85)
    m1, m2 = (parse(Float64, x) for x in sys.masses[1:2])
    (isfinite(m1) && isfinite(m2) && m2 > 0) || throw(ArgumentError("w2_rule: masses of particles 1, 2 must be finite and positive"))
    m = m1 / m2
    (w2=coef * m^(-1 / 4), m=m, inrange=4 <= m <= 207)
end

"""
    set_rule(sys; nsets=4) -> (es, c, Z, inrange, family)

Exponent sets without search, from empirical rules in units of the cusp exponent c (= Z·μ_in, the grid center), fitted to the
sets chosen by `auto_exponents`; the criterion is an error ratio ≤ 10 at the same Meff. Exchange-symmetric pure Coulomb systems
(masses (m₁, m₁, m₃), charges (−,−,+Z)); for Z = 1 the rule switches with m = m₃/m₁ (each band has `inrange`):
- m ≤ 0.1 **v7**: x = max(0.58, min(1, 0.59 (1/(18m))^0.6)), (α,β,γ) = (c,c,0), (xc,c,xc), (c/√2,c/√2,c), (c/√2,c/√2,2c) (heavy pair + light particle).
- 0.1 < m ≤ 0.5 **v5**: (c,c,0), (√2c,√2c,−c/(2√2)), (c,c/√2,−c/(4√2)), (√2c,√2c,−c/√2).
- 0.7 ≤ m ≤ 1.0 **v6** (near Ps⁻): (c,c,0), (c/2,c/2,−c/16), (c/2,c/(2√2),0), (c/2,c/2,−c/(16√2)).
- m ≥ 1.5 **v2**: (c,c,0), (c/2,c/2,−c/8), (c/√2,c/4,0), (√2c,c/√2,−c/4).
- 0.5 < m < 0.7 and 1.0 < m < 1.5 are outside every rule (the nearest band is returned with `inrange` false).
Z ≥ 2 with m ≥ 3 (or m = ∞ with Z ≤ 3): **v3** (4 sets), and with `nsets=8` **v3'** (plus (2c,√2c,−c/2), (2√2c,√2c,−c/√2),
(2√2c,c,−c/4), (c,c,−c/(16√2))). `family` is the chosen rule (:v2 :v3 :v3p :v5 :v6 :v7 :A_atomic).
NoSymmetry heavy pairs (+,+,−) (masses (ma, mb, m₃), ma ≤ mb, q = ma/m₃, r = mb/ma, μ_h = ma mb/(ma+mb)/m₃; pdμ, ptμ, dtμ type;
pass `symmetry=NoSymmetry()` for equal masses) use `family=:v7`: with c₁, c₂ the grid centers of pairs 1, 2 and
x = max(0.58, min(1, 0.59 (q/18)^0.6)), (α,β,γ) = (c₁,c₂,0), (x c₁, c₂, x c₁), (c₁/√2, c₂/√2, c₁), (c₁/√2, c₂/√2, 2c₁).
Range q ≥ 8.5, μ_h ≤ 30, 1 ≤ r ≤ 10 (r ≤ 5 for q < 13).
"""
function set_rule(sys::ThreeBody; nsets::Int=4, symmetry=nothing)
    m1, m2, m3 = (parse(Float64, x) for x in sys.masses)
    sy = symmetry === nothing ? default_symmetry(sys) : symmetry
    _heavy_pair_shape(sys, m1, m2, m3) && (m1 != m2 || sy isa NoSymmetry) && return _set_rule_heavy(sys, m1, m2, m3, sy)
    m1 == m2 || return _set_rule_atomic(sys, m1, m2, m3)
    g = exponent_grid(sys, 1)
    g === nothing && throw(ArgumentError("set_rule: no grid for the pair of particle 1"))
    c = _dyadic(g.center)
    μin = isfinite(m3) ? m1 * m3 / (m1 + m3) : m1
    Zr = c / μin; Z = round(Int, Zr)
    sym = default_symmetry(sys) isa Exchange
    pp = pair_potentials(sys)[1][1]
    m = m3 / m1                                   # mass ratio of the third particle (the Z = 1 rules switch on m)
    nsets in (4, 8) || throw(ArgumentError("set_rule: nsets must be 4 or 8"))
    nsets == 8 && Z == 1 && throw(ArgumentError("set_rule: no 8-set rule for Z = 1 (nsets=4)"))
    D = _dyadic
    base = sym && 1 <= Z <= 4 && abs(Zr - Z) < 0.05 && _hydrogenic_pair(pp) !== nothing
    if Z == 1 && m <= 0.1
        inr = 1 / 2000 <= m; fam = :v7          # v7 form
        x = max(0.58, min(1.0, 0.59 * (1 / m / 18)^0.6))
        a = [c, x * c, c / √2, c / √2]; b = [c, c, c / √2, c / √2]; γ = [0.0, x * c, c, 2c]
    elseif Z == 1 && m <= 0.6
        inr = m <= 0.5; fam = :v5
        a = [c, √2 * c, c, √2 * c]; b = [c, √2 * c, c / √2, √2 * c]; γ = [0.0, -c / (2√2), -c / (4√2), -c / √2]
    elseif Z == 1 && m <= 1.25
        inr = 0.7 <= m <= 1.0; fam = :v6
        a = [c, c / 2, c / 2, c / 2]; b = [c, c / 2, c / (2√2), c / 2]; γ = [0.0, -c / 16, 0.0, -c / (16√2)]
    elseif Z == 1
        inr = 1.5 <= m <= 7300 || !isfinite(m); fam = :v2
        a = [c, c / 2, c / √2, √2 * c]; b = [c, c / 2, c / 4, c / √2]; γ = [0.0, -c / 8, 0.0, -c / 4]
    else
        inr = (isfinite(m) && 3 <= m <= 7300) || (!isfinite(m) && Z <= 3); fam = nsets == 8 ? :v3p : :v3
        a = [c, √2 * c, 2c, c / √2]; b = [c, c, c, c / √2]; γ = [0.0, -c / 8, -c / (2√2), -c / 16]
        if nsets == 8
            append!(a, [2c, 2√2 * c, 2√2 * c, c]); append!(b, [√2 * c, √2 * c, c, c]); γ = vcat(γ, [-c / 2, -c / √2, -c / 4, -c / (16√2)])
        end
    end
    (es=ExponentSets(D.(a), D.(b), D.(γ)), c=c, Z=Z, inrange=base && inr, family=fam, w2=nothing)
end

# Z=1 heavy pair (+,+,−) of two heavy like-charged particles and one light opposite one (pdμ, dtμ type): 1–3 and 2–3 attractive, 1–2 repulsive, m₃ lightest
function _heavy_pair_shape(sys::ThreeBody, m1, m2, m3)
    (isfinite(m1) && isfinite(m2) && isfinite(m3) && m3 > 0 && m3 < min(m1, m2)) || return false
    g1 = exponent_grid(sys, 1); g2 = exponent_grid(sys, 2); g3 = exponent_grid(sys, 3)
    (g1 === nothing || g2 === nothing || g3 === nothing || g3.kind !== :repulsive) && return false
    pps = pair_potentials(sys)[1]
    (_hydrogenic_pair(pps[1]) !== nothing && _hydrogenic_pair(pps[2]) !== nothing) || return false
    abs(g1.center / Float64(pps[1].μ) - 1) < 0.05 && abs(g2.center / Float64(pps[2].μ) - 1) < 0.05
end

# NoSymmetry heavy pairs (+,+,−) (pdμ, ptμ, dtμ type): the v7 ladder (γ > 0, lower bound 0.58 on x)
function _set_rule_heavy(sys::ThreeBody, m1::Float64, m2::Float64, m3::Float64, sy)
    c1 = _dyadic(exponent_grid(sys, 1).center); c2 = _dyadic(exponent_grid(sys, 2).center)
    q = min(m1, m2) / m3; r = max(m1, m2) / min(m1, m2); μh = m1 * m2 / (m1 + m2) / m3
    D = _dyadic; ns = sy isa NoSymmetry && m1 <= m2
    x = max(0.58, min(1.0, 0.59 * (q / 18)^0.6))
    a = [c1, x * c1, c1 / √2, c1 / √2]; b = [c2, c2, c2 / √2, c2 / √2]; γ = [0.0, x * c1, c1, 2c1]
    inrange = ns && q >= 8.5 && μh <= 30 && 1 <= r <= 10 && !(q < 13 && r > 5)
    (es=ExponentSets(D.(a), D.(b), D.(γ)), c=c1, Z=1, inrange=inrange, family=:v7, w2=nothing)
end

# NoSymmetry atomic type (m, 1, M) (μHe type, m₁ ≠ m₂): three sets of family A plus the w₂ rule
function _set_rule_atomic(sys::ThreeBody, m1::Float64, m2::Float64, m3::Float64)
    isfinite(m3) && isfinite(m1) && isfinite(m2) && m2 > 0 || throw(ArgumentError("set_rule: masses must be finite and positive"))
    g2 = exponent_grid(sys, 2)
    g2 === nothing && throw(ArgumentError("set_rule: no grid for the pair of particle 2"))
    pp2 = pair_potentials(sys)[1][2]
    Zr = g2.center / Float64(pp2.μ); Z = round(Int, Zr)
    mred = m1 * m3 / (m1 + m3)
    a = round(Z * mred * 2^20) / 2^20                   # α = Z·m_red (on the 2^{-20} grid)
    es = ExponentSets(fill(a, 3), (Z / 2) .* [1.0, 2.0, 4.0], zeros(3))
    wr = w2_rule(sys); w2 = round(wr.w2 * 2^20) / 2^20
    inrange = default_symmetry(sys) isa NoSymmetry && 2 <= Z <= 4 && abs(Zr - Z) < 0.05 && 4 <= m1 / m2 <= 100 && m3 / m1 >= 8 &&
              _hydrogenic_pair(pair_potentials(sys)[1][1]) !== nothing && _hydrogenic_pair(pp2) !== nothing
    (es=es, c=a, Z=Z, inrange=inrange, family=:A_atomic, w2=w2)
end

"""
    auto_w2(sys, es; n=3, kmax=6, refine=true, βσ=0.0, io=stdout, kwargs...) -> (w2, table)

Choose the weight w₂ of σ = r₁ + w₂ r₂ + βσ r₁₂ by the energy of stage `n`: evaluate w₂ = 2^{−k} (k = 0..kmax) and refine around the
best by 2^{±1/2} (`refine`). The number of functions (Meff) does not depend on w₂, so energies at the same stage compare accuracy
at the same Meff (a differing Meff is printed). w₂ ≠ 1 requires NoSymmetry, L = 0 and no Gaussian. This changes the function
space, so it is not a `run_fc` default (the recommendation is printed). `table` lists (w2, E, Meff) in evaluation order. Other
keywords go to `run_fc` (`rule` gets dedup=false for w₂ ≠ 1).
"""
_with_dedup(r::FCRule, d::Bool) = FCRule(; (f => getfield(r, f) for f in fieldnames(FCRule))..., dedup=d)

function auto_w2(sys::ThreeBody, es::ExponentSets; n::Int=3, kmax::Int=6, refine::Bool=true, βσ::Real=0.0,
                 io::IO=stdout, rule::FCRule=FCRule(), kwargs...)
    haskey(kwargs, :symmetry) && !(kwargs[:symmetry] isa NoSymmetry) && throw(ArgumentError("w₂ ≠ 1 requires NoSymmetry"))
    table = NamedTuple{(:w2, :E, :Meff),Tuple{Float64,BigFloat,Int}}[]
    function ev(w)
        i = findfirst(t -> t.w2 == w, table); i === nothing || return table[i]
        rl = isone(w) ? rule : _with_dedup(rule, false)
        r = run_fc(sys, es; stages=n:n, logvar=LogVariable(βσ; w2=w), rule=rl, symmetry=NoSymmetry(), io=devnull, kwargs...)
        t = (w2=w, E=_tobig(r[end].E), Meff=r[end].Msolved - r[end].nexact)
        push!(table, t)
        @printf(io, "  w₂ = %-10.6g  E = %s  Meff = %d\n", w, fmt(t.E, 30), t.Meff); flush(io)
        t
    end
    println(io, "auto_w2: stage n=", n, "  βσ=", βσ)
    let r = try w2_rule(sys) catch; nothing end
        r === nothing || @printf(io, "  rule w₂ ≈ 0.85 (m₂/m₁)^{1/4} = %.4g (m₁/m₂ = %.4g, %s)\n", r.w2, r.m, r.inrange ? "within the rule range 4–207" : "outside the range, unreliable")
    end
    for k in 0:kmax; ev(2.0^(-k)); end
    best = argmin(t -> t.E, table)
    if refine
        for f in (sqrt(2.0), 1 / sqrt(2.0))
            w = best.w2 * f
            w <= 1 && ev(w)
        end
        best = argmin(t -> t.E, table)
    end
    length(unique(t.Meff for t in table)) == 1 || println(io, "  note: Meff differs with w₂ (energies are not at the same Meff)")
    @printf(io, "  recommended w₂ = %.6g (E %.3e below w₂ = 1); pass logvar=LogVariable(%g; w2=%.6g) to run_fc (the default is unchanged)\n",
            best.w2, Float64(ev(1.0).E - best.E), βσ, best.w2)
    (w2=best.w2, table=table)
end

"""
    auto_form(sys, es; n=3, nplan=nothing, forms=nothing, symmetry=default_symmetry(sys), io=stdout, kwargs...) -> (best, table)

First step of the outer loop that chooses the function form (log variable and rule): with the same exponent sets `es`, solve
stages 1..n for each form and compare energies at the same Meff. Default `forms`: ln(r₁+r₂) (default), ln(r₁+r₂+r₁₂), with
exchange symmetry also ln(r₁+r₂+r₁₂) with the rule of the paper (collect=:st), and with NoSymmetry and no Gaussian
σ = r₁ + w₂ r₂ for w₂ = 1/2, 1/4, 1/8 (dedup=false). `forms` may be passed as (name, logvar, rule) (the first is the baseline).
Decision by variational bounds only: a form wins if its lowest E among stages with Meff ≤ that of the baseline's stage n is
below the baseline's E at stage n. The lowest-E winner is recommended (printed); `run_fc` defaults are not changed. Error ratios
relative to a provisional E∞ are printed for guidance only. Choose the form first, then tune the sets (`auto_exponents`) and α
(`optimize_alpha`) for it. `best` is the winning (label, logvar, rule) (the baseline if none), `table` lists
(label, logvar, rule, stages = [(n, Meff, E)]).
With `nplan` (the stage one intends to reach), stages 1..n+2 are solved and the form with the smallest projected error
e_n / r^{nplan−n} is chosen (r = (E_n − E_{n+1})/(E_{n+1} − E_{n+2}), e_n the difference from the geometric E∞ of stages n..n+2,
as the logerr of `auto_exponents`); the energy at stage n alone does not capture the rate. Other keywords go to `run_fc`.
"""
function auto_form(sys::ThreeBody, es::ExponentSets; n::Int=3, nplan::Union{Nothing,Int}=nothing, forms=nothing,
                   symmetry=default_symmetry(sys), io::IO=stdout, kwargs...)
    if forms === nothing
        forms = Any[("ln(r₁+r₂) (default)", LogVariable(), FCRule()), ("ln(r₁+r₂+r₁₂)", NewLog(), FCRule())]
        # the rule of the paper (NN2007): collect with coefficients in the s, t representation (exchange-symmetric systems only)
        symmetry isa Exchange && !_gaussian(es) &&
            push!(forms, ("ln(r₁+r₂+r₁₂), rule of the paper", NewLog(), FCRule(collect=:st, gcoef=["1/2", "1"], dedup=false)))
        if symmetry isa NoSymmetry && !_gaussian(es)
            for w in (0.5, 0.25, 0.125)
                push!(forms, (@sprintf("log1 w₂=%g", w), LogVariable(0.0; w2=w), FCRule(dedup=false)))
            end
        end
    end
    kw = (; (k => v for (k, v) in pairs(kwargs) if !(k in (:logvar, :rule, :stages, :io)))...)
    table = NamedTuple[]
    ntop = nplan === nothing ? n : n + 2
    println(io, "auto_form: stages 1..", ntop, ", ", length(forms), " forms, ", length(es), " sets, symmetry ", symmetry isa Exchange ? "exchange" : "none",
            nplan === nothing ? "" : ", compared by the error projected to stage $nplan from the ratio of stages $n..$(n + 2)")
    function runform(lab, lv, rl)
        t = @elapsed r = try
            run_fc(sys, es; stages=1:ntop, logvar=lv, rule=rl, symmetry=symmetry, io=devnull, kw...)
        catch e
            e isa InterruptException && rethrow()
            sprint(showerror, e)
        end
        st = r isa String ? Tuple{Int,Int,BigFloat}[] : [(s.n, s.Msolved - s.nexact, _tobig(s.E)) for s in r]
        push!(table, (label=lab, logvar=lv, rule=rl, stages=st))
        if isempty(st)
            @printf(io, "  %s: failed %s\n", lab, r isa String ? first(r, 200) : "empty")
        else
            @printf(io, "  %s (%.0f s): %s\n", lab, t, join([@sprintf("n=%d Meff=%d E=%s", s[1], s[2], fmt(s[3], 20)) for s in st], "  "))
        end
        flush(io)
    end
    for (lab, lv, rl) in forms; runform(lab, lv, rl); end
    if nplan !== nothing
        # projected error at stage nplan from the ratio of stages n..n+2 (smaller is better); Inf without three stages or with ratio ≤ 1
        function proj(f)
            st = Dict(x[1] => x[3] for x in f.stages)
            all(k -> haskey(st, k), n:n+2) || return (Inf, NaN, NaN)
            d1 = st[n] - st[n+1]; d2 = st[n+1] - st[n+2]
            (d1 > 0 && d2 > 0 && d1 > d2) || return (Inf, NaN, NaN)
            r = Float64(d1 / d2); e = Float64(st[n] - (st[n+2] - d2^2 / (d1 - d2)))
            (e / r^(nplan - n), e, r)
        end
        isempty(table[1].stages) && error("baseline form $(table[1].label) cannot be solved")
        while true      # if the smallest w₂ on the grid wins, add half of it (as for the stage-n criterion)
            ws = [f for f in table if f.logvar.w2 != 1 && !f.logvar.split && !isempty(f.stages)]
            isempty(ws) && break
            e = argmin(f -> f.logvar.w2, ws)
            fb = argmin(f -> proj(f)[1], table)
            (fb === e && e.logvar.w2 > 1 / 64) || break
            w = e.logvar.w2 / 2
            println(io, "  w₂ = ", e.logvar.w2, " is best at the end of the candidates → adding w₂ = ", w); flush(io)
            runform(@sprintf("log1 w₂=%g", w), LogVariable(e.logvar.βσ; w2=w), e.rule)
        end
        @printf(io, "  decision (error projected to stage %d from the ratio of stages %d..%d, e_%d / r^%d):\n", nplan, n, n + 2, n, nplan - n)
        for f in table
            pj, e, r = proj(f)
            Ms = [x[2] for x in f.stages if n <= x[1] <= n + 2]
            @printf(io, "    %s: e_%d = %.3e, ratio r = %.4g, projection at stage %d %.3e (Meff %s)\n", f.label, n, e, r, nplan, pj, join(Ms, "/"))
        end
        fb = argmin(f -> proj(f)[1], table)
        best = isfinite(proj(fb)[1]) ? (label=fb.label, logvar=fb.logvar, rule=fb.rule) : (label=table[1].label, logvar=table[1].logvar, rule=table[1].rule)
        length(unique(Tuple(x[2] for x in f.stages) for f in table if !isempty(f.stages))) == 1 ||
            println(io, "  note: Meff per stage differs between forms (the projections compare stages, not equal Meff)")
        println(io, "  recommended: ", best.label, best.label == table[1].label ? " (the baseline)" :
                "; use this logvar and rule and tune the sets and α for it (pass logvar and rule to auto_exponents / optimize_alpha)")
        flush(io)
        return (best=best, table=table)
    end
    base = table[1]
    isempty(base.stages) && error("baseline form $(base.label) cannot be solved")
    Mref, Eref = base.stages[end][2], base.stages[end][3]
    # lowest E among the candidate's stages with Meff ≤ the baseline's Meff at stage n (nothing if none)
    bestok(f) = (ok = [s for s in f.stages if s[2] <= Mref]; isempty(ok) ? nothing : argmin(x -> x[3], ok))
    # if the smallest w₂ on the grid wins, add half of it and compare again (until it is not at the end, or 1/64); the rule is unchanged
    while true
        ws = [f for f in table if f.logvar.w2 != 1 && !f.logvar.split && !isempty(f.stages)]
        isempty(ws) && break
        e = argmin(f -> f.logvar.w2, ws)
        cands = [(f, bestok(f)) for f in table[2:end]]
        cands = [(f, s) for (f, s) in cands if s !== nothing && s[3] < Eref]
        isempty(cands) && break
        fb = argmin(x -> x[2][3], cands)[1]
        (fb === e && e.logvar.w2 > 1 / 64) || break
        w = e.logvar.w2 / 2
        println(io, "  w₂ = ", e.logvar.w2, " is best at the end of the candidates → adding w₂ = ", w); flush(io)
        runform(@sprintf("log1 w₂=%g", w), LogVariable(e.logvar.βσ; w2=w), e.rule)
    end
    # provisional E∞: geometric tail (tail_estimate) subtracted from the column with the lowest last E (guidance only; the last increment without three stages or a ratio in (0,1))
    ib = argmin(i -> isempty(table[i].stages) ? big(Inf) : table[i].stages[end][3], eachindex(table))
    sb = table[ib].stages
    Einf = if length(sb) >= 3
        t = tail_estimate(Float64(sb[end-1][3] - sb[end][3]), Float64(sb[end-2][3] - sb[end-1][3]))
        sb[end][3] - (isnan(t) ? max(sb[end-1][3] - sb[end][3], zero(BigFloat)) : t)
    else
        sb[end][3] - (length(sb) >= 2 ? max(sb[end-1][3] - sb[end][3], zero(BigFloat)) : zero(BigFloat))
    end
    best = (label=base.label, logvar=base.logvar, rule=base.rule); bestE = Eref
    @printf(io, "  decision (baseline %s at stage %d, lowest E among stages with Meff ≤ %d):\n", base.label, base.stages[end][1], Mref)
    for f in table[2:end]
        s = bestok(f)
        if s === nothing
            @printf(io, "    %s: no stage with Meff ≤ %d\n", f.label, Mref); continue
        end
        win = s[3] < Eref
        rat = Float64((s[3] - Einf) / (Eref - Einf))
        @printf(io, "    %s: E at stage %d (Meff %d) is %s the baseline (ΔE = %.3e, indicative error ratio %.3g)\n", f.label, s[1], s[2],
                win ? "below → wins" : "above", Float64(s[3] - Eref), rat)
        win && s[3] < bestE && (best = (label=f.label, logvar=f.logvar, rule=f.rule); bestE = s[3])
    end
    best.logvar.w2 != 1 && best.logvar.w2 <= 1 / 64 &&
        println(io, "  note: the best w₂ = ", best.logvar.w2, " is at the end of the extended candidates (1/64); refine with auto_w2(sys, es; n)")
    println(io, "  recommended: ", best.label, best.label == base.label ? " (the baseline)" :
            "; use this logvar and rule and tune the sets and α for it (pass logvar and rule to auto_exponents / optimize_alpha)")
    flush(io)
    (best=best, table=table)
end

"""
    auto_basis(sys; n=3, symmetry=default_symmetry(sys), maxsets=8, late=0, late_rounds=3, gauss=true, nplan=8, io=stdout, kwargs...) -> (es, logvar, rule, form, start, auto, recheck, late, gauss)

Choose the basis in the order form → scale → sets. Everything is a recommendation; `run_fc` defaults are unchanged.
1. Form: `auto_form(sys, default_exponents(sys); n, nplan)` (ln(r₁+r₂), ln(r₁+r₂+r₁₂), w₂), by the error projected to stage nplan.
2. Scale: tune the single set (α, β, 0) at the grid centers by the energy of stage n in that form (`optimize_alpha` for α = β
   with exchange symmetry, else `optimize_exponents` over (α, β)).
3. Sets: `auto_exponents(sys; n, start, logvar, rule)` from the tuned set.
The scale comes before the sets because sets added at small stages can compensate a mistuned α and then lose at large stages.
4b. Gaussian in r₁₂ (`gauss = true`, default; exchange symmetry with the identical pair at input 1, 2, or NoSymmetry with
   particle 3 lighter than 1, 2, `_gauss_pair_ok`): one set from the moments of the step-3 wave function by `auto_gaussr`;
   returned if it beats step 3 at the same Meff and the localization power C* exceeds `nplan` (step 5 is skipped, `_gauss_step`).
5. Late sets (only for `late > 0`, default 0): `late_greedy(sys, es; n=late, rounds=late_rounds)` from the step-3 sets.
`maxsets` and other keywords go to `auto_exponents` (and `run_fc`).
"""
function auto_basis(sys::ThreeBody; n::Int=3, symmetry=default_symmetry(sys), maxsets::Int=8, late::Int=0, late_rounds::Int=3,
                    gauss::Bool=true, nplan::Int=8, io::IO=stdout, kwargs...)
    kw = (; (k => v for (k, v) in pairs(kwargs) if !(k in (:logvar, :rule, :start, :symmetry)))...)
    kwrun = (; (k => v for (k, v) in pairs(kw) if !(k in (:budget, :families, :famratio, :famlen, :exhaustive, :keep_ratio, :nrank, :lookahead,
                                                          :lazy, :gamma, :objective, :space, :bigbits, :nstates, :cache_cap, :memdiag)))...)
    println(io, "auto_basis: stage n=", n, "  1. form → 2. scale → 3. sets"); flush(io)
    f = auto_form(sys, default_exponents(sys); n=n, nplan=nplan, symmetry=symmetry, io=io, kwrun...)
    lv, rl = f.best.logvar, f.best.rule
    exch = symmetry isa Exchange
    _, _, π = orient(sys, nothing, symmetry; io=devnull)
    if π != (1, 2, 3)          # identical pair not at input 1, 2: no scale tuning (avoids renumbering the start set), start from the auto_exponents default
        println(io, "  2. scale: identical pair not at input particles 1, 2; not tuned (auto_exponents default start)"); flush(io)
        r = auto_exponents(sys; n=n, logvar=lv, rule=rl, symmetry=symmetry, maxsets=maxsets, io=io, kw...)
        f4 = _recheck_form(sys, r.es, f, n, symmetry, io, kwrun; nplan=nplan)
        lg = _late_step(sys, r.es, late, late_rounds, lv, rl, symmetry, io, kwrun)
        return (es=lg === nothing ? r.es : lg.es, logvar=lv, rule=rl, form=f, start=nothing, auto=r, recheck=f4, late=lg, gauss=nothing)
    end
    g1 = exponent_grid(sys, 1); g2 = exch ? g1 : exponent_grid(sys, 2)
    a0, b0 = _dyadic(g1.center), _dyadic(g2.center)
    bin(x) = round(x * 2.0^20) / 2.0^20
    println(io, "  2. scale: tuning the start set (", a0, ", ", b0, ", 0) by the energy of stage ", n, " (", f.best.label, ")"); flush(io)
    start = if exch
        o = optimize_alpha(sys, n; bracket=(a0 / 2, 1.5a0), tol=1e-3 * a0, logvar=lv, rule=rl, symmetry=symmetry, io=io, kwrun...)
        a = bin(o.α); ExponentSets([a], [a], [0.0])
    else
        o = optimize_exponents(sys, n, [a0, b0]; make_es=x -> ExponentSets([bin(x[1])], [bin(x[2])], [0.0]), maxeval=40,
                               logvar=lv, rule=rl, symmetry=symmetry, io=io, kwrun...)
        ExponentSets([bin(o.x[1])], [bin(o.x[2])], [0.0])
    end
    @printf(io, "  → one set (%.6g, %.6g, 0)\n  3. sets: auto_exponents (from the tuned set)\n", start.α[1], start.β[1]); flush(io)
    r = auto_exponents(sys; n=n, start=start, logvar=lv, rule=rl, symmetry=symmetry, maxsets=maxsets, io=io, kw...)
    f4 = _recheck_form(sys, r.es, f, n, symmetry, io, kwrun; nplan=nplan)
    gr = gauss && _gauss_pair_ok(sys, symmetry) ? _gauss_step(sys, r.es, n, lv, rl, symmetry, io, kwrun; nplan=nplan) : nothing
    if gr !== nothing && gr.win
        println(io, "auto_basis recommends: one Gaussian set (α, η, Re) = ", gr.g.x, ", rule=", gr.g.rule,
                " (late_greedy does not support Gaussian sets, so step 5 is skipped; dmin ≈ −⌈n/3⌉ works well at large stages)"); flush(io)
        return (es=gr.g.es, logvar=LogVariable(), rule=gr.g.rule, form=f, start=start, auto=r, recheck=f4, late=nothing, gauss=gr)
    end
    println(io, "auto_basis recommends: logvar=", f.best.label, ", ", length(r.es), " sets α=", r.es.α, " β=", r.es.β, " γ=", r.es.γ,
            " (compare at large stages with the single set ", start.α[1], " at the same Meff before use)")
    # sets chosen at stage n can saturate at large stages; late_greedy adds sets by the error ratio at the same Meff at stage 6
    late > 0 || (println(io, "hint: to add sets at a large stage, late_greedy(sys, es; n=6, symmetry, logvar, rule) or auto_basis(…; late=6)",
                         " (one set at a time by the error ratio at the same Meff at stage 6; hours)"); flush(io))
    lg = _late_step(sys, r.es, late, late_rounds, lv, rl, symmetry, io, kwrun)
    (es=lg === nothing ? r.es : lg.es, logvar=lv, rule=rl, form=f, start=start, auto=r, recheck=f4, late=lg, gauss=gr)
end

"""
4b. Gaussian in r₁₂ (`gauss = true`, exchange symmetry): one set from the step-3 wave function by `auto_gaussr`, compared with
stages 1..nm of the step-3 sets (the moment run of `auto_gaussr`). Two conditions to win:
- E: the lowest Gaussian E among stages with Meff ≤ that of step 3 at stage nm is below step 3's E at stage nm (as `auto_form`).
- Localization: from κ = ⟨r₁₂⟩²/Var of the step-3 wave function, the power C* at which one exponential term has an r₁₂ peak of
  the same width (`_peak_power`, closed form) exceeds `nplan`. Exponential bases gain one power of r₁₂ per stage through g, so
  from stage C* they can form the peak and overtake the Gaussian. The local order p is printed only.
"""
function _gauss_step(sys, es, n, lv, rl, symmetry, io, kwrun; nplan::Int=8)
    println(io, "  4b. Gaussian in r₁₂: (α, η, Re) from the step-3 wave function, tuned and compared at the same Meff"); flush(io)
    g = try
        auto_gaussr(sys, es; logvar=lv, rule=rl, symmetry=symmetry, io=io, kwrun...)
    catch e
        e isa InterruptException && rethrow()
        println(io, "  Gaussian in r₁₂: failed ", first(sprint(showerror, e), 200)); flush(io)
        return nothing
    end
    nb, Mref, Eref = g.base[end]
    ok = [q for q in g.stages if q[2] <= Mref]
    winE = !isempty(ok) && minimum(q -> q[3], ok) < Eref
    pg, pe = _local_order(g.stages), _local_order(g.base)
    κ = g.mom[1]^2 / g.mom[2]
    Cs = _peak_power(κ, sort(es.α)[(length(es) + 1) ÷ 2], sort(es.γ)[(length(es) + 1) ÷ 2])
    winC = Cs === nothing || Cs > nplan
    win = winE && winC
    @printf(io, "  localization κ = ⟨r₁₂⟩²/Var = %.3g → one term r₁₂^C e^{−γr₁₂} forms a peak of the same width for C ≥ %s (from stage C); planned stage nplan = %d\n",
            κ, Cs === nothing ? ">200" : string(Cs), nplan)
    if isempty(ok)
        @printf(io, "  decision: no Gaussian stage with Meff ≤ %d → exponential form\n", Mref)
    else
        q = argmin(x -> x[3], ok)
        @printf(io, "  decision: E at Gaussian stage %d (Meff %d) is %s step 3 at stage %d (Meff %d) (ΔE = %.3e), C* %s nplan → %s (local order p: Gaussian %.3g, step 3 %.3g)\n",
                q[1], q[2], nb, Mref, winE ? "below" : "above", Float64(q[3] - Eref), winC ? ">" : "≤",
                win ? "Gaussian form (the exponential form overtakes beyond stage C*)" : winE ? "exponential form (it overtakes at stage C* ≤ nplan)" : "exponential form", pg, pe)
    end
    flush(io)
    (g=g, win=win, winE=winE, winC=winC, pg=pg, pe=pe, Mref=Mref, Eref=Eref, κ=κ, Cstar=Cs, nplan=nplan)
end

"""
Whether the Gaussian in r₁₂ (distance of particles 1–2, e^{−η(r₁₂−Re)²}) applies: exchange symmetry (identical pair at 1, 2), or
NoSymmetry with particle 3 lighter than 1 and 2 (molecular ions with different nuclei such as HD⁺; r₁₂ is the slow internuclear coordinate).
"""
function _gauss_pair_ok(sys::ThreeBody, symmetry)
    symmetry isa Exchange && return true
    symmetry isa NoSymmetry || return false
    m = [_rat(x) for x in sys.masses]
    m[3] === nothing && return false                                  # particle 3 infinitely heavy
    all(i -> m[i] === nothing || m[i] > m[3], 1:2)
end

"""
    _peak_power(κ, α, γ; Cmax=200) -> C or nothing

Smallest C for which the r₁₂ marginal of one exponential term r₁₂^C e^{−γr₁₂} e^{−α(r₁+r₂)} (volume element r₁r₂r₁₂; after integrating r₁, r₂:
r₁₂^{2C+1} e^{−(2γ+2α)r₁₂}(r₁₂³/3a + r₁₂²/a² + r₁₂/a³)/2, a = 2α) has ⟨r₁₂⟩²/Var ≥ κ. For γ = 0, κ_C ≈ 3.94, 6.21, 8.37, 10.5, …
(≈ 2.1C + 3.9) independent of α. g = r₁₂ adds one power per stage, so an exponential basis can carry the r₁₂ peak from stage C.
"""
function _peak_power(κ, α, γ; Cmax::Int=200)
    a = 2α; b = 2γ + a
    b > 0 || return nothing
    lg(N) = sum(log, 1:N-1; init=0.0)
    for C in 0:Cmax
        M(k) = sum(w * exp(lg(N + k) - (N + k) * log(b)) for (w, N) in ((1 / (3a), 2C + 5), (1 / a^2, 2C + 4), (1 / a^3, 2C + 3)))
        m1 = M(1) / M(0); v = M(2) / M(0) - m1^2
        m1^2 / v >= κ && return C
    end
    nothing
end

"Local order p = ln(ΔE₋₁/ΔE)/ln(M/M₋₁) of the last two increments of [(n, Meff, E)]; NaN for fewer than three stages or non-positive increments."
function _local_order(st)
    length(st) >= 3 || return NaN
    (_, M1, E1), (_, M2, E2), (_, M3, E3) = st[end-2], st[end-1], st[end]
    d1, d2 = E1 - E2, E2 - E3
    (d1 > 0 && d2 > 0 && M3 > M2) || return NaN
    Float64(log(d1 / d2) / log(BigFloat(M3) / M2))
end

"""
    auto_gaussr(sys, es; nm=4, s=4, n=s+4, logvar=LogVariable(), rule=FCRule(), symmetry=default_symmetry(sys), io=stdout, kwargs...)
        -> (es, rule, x0, x, stages, base)

One Gaussian set (e^{−α(r₁+r₂)} e^{−η(r₁₂−Re)²}, `gaussian_r12`) computed from results, without hand-chosen values:
1. From the wave function of stage `nm` of the given form and sets (`es`, `logvar`, `rule`): Re = ⟨r₁₂⟩, η = 1/(4 Var r₁₂)
   (width of ψ²), α = median of `es.α`. The moments use at least three times the working precision (≥ 768 bits), doubled while
   ⟨H⟩ − E exceeds 1e-20 E.
2. Tune (α, η, Re) by the energy of stage `s` (`optimize_exponents`, up to 100 evaluations).
3. Solve stages 1..n with the tuned set and return `stages` = [(n, Meff, E)] (for comparing forms).
`base` is [(n, Meff, E)] of stages 1..nm of the given sets. The rule is `FCRule(seeds=[(0,0,0,0,0)], jmax=0, dmin=-2)` (Gaussian
integrals have no logarithm and α = β).
"""
function auto_gaussr(sys::ThreeBody, es::ExponentSets; nm::Int=4, s::Int=4, n::Int=s + 4, logvar::LogVariable=LogVariable(),
                     rule=FCRule(), symmetry=default_symmetry(sys), io::IO=stdout, kwargs...)
    _gauss_pair_ok(sys, symmetry) || throw(ArgumentError("auto_gaussr: needs exchange symmetry (identical pair at input 1, 2) or particle 3 lighter than 1, 2 (NoSymmetry)"))
    kw = (; (k => v for (k, v) in pairs(kwargs) if !(k in (:logvar, :rule, :stages, :io, :outdir, :T)))...)
    # the run_fc calls below print to devnull, so print the warning about non-binary exponents here
    bw = binary_warning(es); bw === nothing || println(io, "  warning: ", bw)
    d = mktempdir()
    t = @elapsed r0 = run_fc(sys, es; stages=1:nm, logvar=logvar, rule=rule, symmetry=symmetry, T=BigFloat, io=devnull, outdir=d, kw...)
    wf = load_wavefunction(d, nm)
    bits = max(768, 3 * precision(BigFloat)); m = nothing
    while true
        m = setprecision(BigFloat, bits) do
            expectation(sys, wf, [DistancePower(0, 0, 1), DistancePower(0, 0, 2), TotalEnergy()]; logvar=logvar)
        end
        (abs(m[3] - wf.E) <= 1e-20 * abs(wf.E) || bits >= 4096) && break
        bits *= 2
    end
    rm(d; recursive=true, force=true)
    base = [(q.n, q.Msolved - q.nexact, _tobig(q.E)) for q in r0]
    R1, R2 = Float64(m[1]), Float64(m[2]); v = R2 - R1^2
    v > 0 && R1 > 0 || error("auto_gaussr: ⟨r₁₂⟩ = $(R1), Var = $(v) (⟨H⟩−E = $(Float64(m[3] - wf.E)) at $(bits) bits)")
    x0 = [sort(es.α)[(length(es) + 1) ÷ 2], 1 / (4v), R1]
    @printf(io, "  Gaussian: from the wave function of stage %d (Meff %d, %.0f s, moments at %d bits, ⟨H⟩−E %.1e) ⟨r₁₂⟩ %.5g, Var %.4g → (α, η, Re) = (%.4g, %.4g, %.4g)\n",
            nm, r0[end].Msolved - r0[end].nexact, t, bits, Float64(m[3] - wf.E), R1, v, x0...); flush(io)
    grule = FCRule(seeds=[(0, 0, 0, 0, 0)], jmax=0, dmin=-2)
    bin(x) = round(x * 2.0^20) / 2.0^20
    mk(x) = gaussian_r12(bin(x[1]), bin(x[2]), bin(x[3]))
    t = @elapsed o = optimize_exponents(sys, s, x0; make_es=mk, step=0.1, tol=2e-3, maxeval=100, rule=grule, symmetry=symmetry, T=BigFloat, io=devnull, kw...)
    @printf(io, "  tuned by the energy of stage %d (%d evaluations, %.0f s) → (α, η, Re) = (%.4g, %.4g, %.4g)\n", s, length(o.evals), t, o.x...); flush(io)
    ges = mk(o.x)
    t = @elapsed r = run_fc(sys, ges; stages=1:n, rule=grule, symmetry=symmetry, T=BigFloat, io=devnull, kw...)
    st = [(q.n, q.Msolved - q.nexact, _tobig(q.E)) for q in r]
    @printf(io, "  stages 1..%d (%.0f s): %s\n", n, t, join([@sprintf("n=%d Meff=%d E=%s", q[1], q[2], fmt(q[3], 20)) for q in st], "  ")); flush(io)
    (es=ges, rule=grule, x0=x0, x=o.x, stages=st, base=base, mom=(R1, v))
end

"5. Late sets (for `late > 0`): `late_greedy(sys, es; n=late, rounds)` from the step-3 sets; nothing for 0."
function _late_step(sys, es, late, rounds, lv, rl, symmetry, io, kwrun)
    late > 0 || return nothing
    println(io, "  5. late sets: late_greedy (stage ", late, ", rounds ≤ ", rounds, ")"); flush(io)
    late_greedy(sys, es; n=late, rounds=rounds, logvar=lv, rule=rl, symmetry=symmetry, io=io, kwrun...)
end

"""
4. Check of the form: rerun `auto_form` on the chosen sets `es` at the same stage n (nothing for a single set, which is the
space of step 1). Step 1 chooses the form with one set at the grid centers, and the best form can change with more sets; a
difference is printed (not switched).
"""
function _recheck_form(sys, es, f, n, symmetry, io, kwrun; nplan=nothing)
    length(es) <= 1 && return nothing
    m = n
    println(io, "  4. form check: auto_form on the ", length(es), " chosen sets (stages 1..", m, ")"); flush(io)
    f4 = auto_form(sys, es; n=m, nplan=nplan, symmetry=symmetry, io=io, kwrun...)
    f4.best.label == f.best.label ||
        println(io, "  note: on the chosen sets ", f4.best.label, " is better (step 1 chose ", f.best.label,
                "); use it, or retune the sets in it (pass logvar and rule to auto_exponents)")
    flush(io)
    f4
end

"""
    late_greedy(sys, es; n=6, rounds=3, thr=0.8, screen=n-2, topk=6, symmetry=default_symmetry(sys), io=stdout, kwargs...) -> (es, history)

Late greedy: add sets one at a time at a large stage `n`. `auto_exponents` measures sets at stages 2–3 and misses sets whose
gain appears only later; such sets were of the type γ = +c/2 (decaying in r₁₂).
- **Candidates**: single sets (α, β) from the 1–3 grid (`exponent_grid`) between 1/4 and 2 times its center c, times
  γ ∈ {0, c_γ/2, −c_γ/4, γ of the current sets}; c_γ is the smaller center of the 1–3 and 2–3 grids. α ≥ β with exchange
  symmetry; β from the 2–3 grid if it differs. Sets nearly equal to current ones (ratios within 2^{1/8}) are excluded.
- **Measure**: error ratio at the same Meff, without references. E∞ is extrapolated geometrically from stages n+1..n+3 of the
  current sets (`tail_estimate`); (Meff, E − E∞) of the current stages 1..n+3 is interpolated log–log and the candidate's error
  at its stage-n Meff is divided by it (0 if the candidate goes below E∞).
- **Stop**: if the best ratio is below `thr`, add the set and start a new round (E∞ rebuilt); stop at `thr` or after `rounds`.
- **Screening** (`screen`, default n − 2; 0 disables): measure all candidates at stage `screen` and only the top `topk` (default 6)
  at stage n. The threshold applies to the stage-n ratio.
- **Cost** per round: stages 1..n+3 of the current sets once, stages 1..screen per candidate, stages 1..n for the top topk.
- `cands`: candidate sets (α, β, γ) to use instead (duplicates and invalid sets are still excluded).
Ground state only; changes the function space, so not a default. Other keywords go to `run_fc`. `history` has one row per round:
`base` ((stage, Meff, E) of the current sets), `Einf`, `rows` (per candidate at stage n: `cand`, `M`, `err`, `ratio`), `screened`
(the same at stage screen), `chosen`.
"""
function late_greedy(sys::ThreeBody, es::ExponentSets; n::Int=6, rounds::Int=3, thr::Float64=0.8, cands=nothing,
                     screen::Int=n >= 4 ? n - 2 : 0, topk::Int=6, symmetry=default_symmetry(sys), io::IO=stdout, kwargs...)
    exch = symmetry isa Exchange
    g1 = exponent_grid(sys, 1); g2 = exch ? g1 : exponent_grid(sys, 2)
    (g1 === nothing || g2 === nothing) && throw(ArgumentError("no exponent grid: pair 1–3 or 2–3 has no terms"))
    c = g1.center
    win(g) = sort(unique(_dyadic.([x for x in g.values if g.center / 4 - 1e-9 <= x <= 2g.center + 1e-9])); rev=true)
    va = win(g1); vb = exch ? va : win(g2)
    # scale of γ (exponent of r₁₂): the smaller center of the two bound pairs; the γ of the current sets is also a candidate
    cg = min(c, g2.center)
    gs = Tuple(unique([0.0, _dyadic(cg / 2), -_dyadic(cg / 4), es.γ...]))
    near1(x, y) = x == y || (x != 0 && y != 0 && sign(x) == sign(y) && abs(log(abs(x / y))) < log(2) / 8)
    near(a, s) = near1(a[1], s[1]) && near1(a[2], s[2]) && near1(a[3], s[3])
    same(a, s) = near(a, s) || (exch && near((a[2], a[1], a[3]), s))
    valid(a) = a[2] + a[3] > 0 && a[1] + a[3] > 0 && a[1] + a[2] > 0
    mk(ps) = ExponentSets(getindex.(ps, 1), getindex.(ps, 2), getindex.(ps, 3))
    series(ps, m) = [(s.n, s.Msolved - s.nexact, _tobig(s.E)) for s in run_fc(sys, mk(ps); stages=1:m, symmetry=symmetry, io=devnull, kwargs...)]
    function ratio(pts, M, e)
        e <= 0 && return 0.0
        length(pts) < 2 && return NaN
        i = findlast(p -> p[1] <= M, pts)
        i === nothing && return NaN
        i == length(pts) && (i -= 1)                    # beyond the top: extend the last interval
        (m1, e1), (m2, e2) = pts[i], pts[i+1]
        e / exp(log(e1) + (log(e2) - log(e1)) * (log(M) - log(m1)) / (log(m2) - log(m1)))
    end
    sets = collect(zip(es.α, es.β, es.γ)); hist = []
    @printf(io, "late_greedy: stage %d, rounds ≤ %d, threshold %.3g, α ∈ %s, β ∈ %s, γ ∈ %s\n", n, rounds, thr,
            string(round.(va; sigdigits=4)), exch ? "(≤ α)" : string(round.(vb; sigdigits=4)), string(round.(collect(gs); sigdigits=4))); flush(io)
    for rd in 1:rounds
        t = @elapsed b = try series(sets, n + 3) catch e; e isa InterruptException && rethrow(); [] end
        if length(b) < n + 3
            println(io, "  stop: stage n+3 = ", n + 3, " cannot be solved with the current sets"); break
        end
        Δ1 = b[end-1][3] - b[end][3]; Δ0 = b[end-2][3] - b[end-1][3]
        tl = tail_estimate(Float64(Δ1), Float64(Δ0))
        Einf = b[end][3] - (isnan(tl) ? max(Δ1, zero(Δ1)) : tl)
        pts = [(Float64(m), Float64(E - Einf)) for (_, m, E) in b if E - Einf > 0]
        cs = cands === nothing ? [(a, bb, g) for a in va for bb in vb if !(exch && a < bb) for g in gs] : [Tuple(Float64.(x)) for x in cands]
        cs = [x for x in cs if valid(x) && !any(s -> same(x, s), sets)]
        @printf(io, "  round %d: %d current sets, stages 1..%d (%.0f s), E∞ = %s, error at stage %d %.3e (Meff %d), %d candidates\n", rd, length(sets), n + 3, t,
                fmt(Einf, 25), n, Float64(b[n][3] - Einf), b[n][2], length(cs)); flush(io)
        rows = []; best = (Inf, nothing); scr = []
        if 0 < screen < n && length(cs) > topk          # measure all at stage screen, only the top topk at stage n
            for x in cs
                t = @elapsed v = try series([sets; x], screen) catch e; e isa InterruptException && rethrow(); [] end
                length(v) < screen && (@printf(io, "    stage %d: + (%.4g, %.4g, %.4g) cannot be solved\n", screen, x...); continue)
                M = v[end][2]; e = Float64(v[end][3] - Einf); r = ratio(pts, M, e)
                push!(scr, (cand=x, M=M, err=e, ratio=r))
                @printf(io, "    stage %d: + (%.4g, %.4g, %.4g): Meff %d, error %.3e, ratio at the same Meff %.3g (%.0f s)\n", screen, x..., M, e, r, t); flush(io)
            end
            sort!(scr; by=q -> isfinite(q.ratio) ? q.ratio : Inf)
            cs = [q.cand for q in scr[1:min(topk, end)]]
            @printf(io, "  measuring the top %d of stage %d at stage %d\n", length(cs), screen, n); flush(io)
        end
        for x in cs
            t = @elapsed v = try series([sets; x], n) catch e; e isa InterruptException && rethrow(); [] end
            if length(v) < n
                @printf(io, "    + (%.4g, %.4g, %.4g): stage %d cannot be solved\n", x..., n); continue
            end
            M = v[end][2]; e = Float64(v[end][3] - Einf); r = ratio(pts, M, e)
            push!(rows, (cand=x, M=M, err=e, ratio=r))
            @printf(io, "    + (%.4g, %.4g, %.4g): Meff %d, error %.3e, ratio at the same Meff %.3g (%.0f s)\n", x..., M, e, r, t); flush(io)
            isfinite(r) && r < best[1] && (best = (r, x))
        end
        ok = best[2] !== nothing && best[1] < thr
        push!(hist, (round=rd, base=b, Einf=Einf, rows=rows, screened=scr, chosen=ok ? best[2] : nothing))
        if !ok
            @printf(io, "  round %d: best ratio %.3g ≥ threshold %.3g → stop\n", rd, best[1], thr); flush(io); break
        end
        push!(sets, best[2])
        @printf(io, "  round %d: + (%.4g, %.4g, %.4g), ratio %.3g\n", rd, best[2]..., best[1]); flush(io)
    end
    es2 = mk(sets)
    println(io, "late_greedy sets: α=", es2.α, " β=", es2.β, " γ=", es2.γ, " (compare with the original sets at the same Meff at large stages before use)"); flush(io)
    (es=es2, history=hist)
end
