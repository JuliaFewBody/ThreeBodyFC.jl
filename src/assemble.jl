# Assembly of H and S.
#
#   S_ij = ⟨φ_i|φ_j⟩,   H_ij = ⟨φ_i|H|φ_j⟩ = Σ_{(m,k',c) ∈ Hφ_j} c ⟨φ_i|m e^{−(k')}⟩
#
# The element integrals are tabulated first (each distinct element computed once) and the matrices are filled by lookup.

"Create, sequentially before assembly, the exponent sets that ExpTerm maps basis functions to (the parallel part does not modify es)."
function close_exponent_sets!(hd::HamiltonianData, es::ExponentSets, basis::Vector{Key})
    for key in basis, (_, mu, _) in hd.expo
        shifted_set!(es, key[6], mu)
    end
    es
end

"""
    assemble(hd, ctx, es, basis; rows, ntasks) -> (H, S)

`H[r, j] = ⟨φ_{rows[r]}|H|φ_j⟩`, likewise `S` (size length(rows) × M).
If `rows` is everything, symmetric matrices are returned (for each pair i < j only the side with fewer Hφ terms is computed).
If `rows` is a subset, the raw unsymmetrized H is returned (used by `hermiticity`).
With `packed=true` and BigFloat, the result is the packed lower triangle (`SymPacked`, about 1/3 of the memory; bit-identical values).
`progress > 0` prints the fraction of finished columns, elapsed and remaining time to `io` every `progress` seconds.
`dscale = Ref{Any}(nothing)` (packed only) normalizes by the diagonal of S while filling (same operations as `diagonal_normalize`,
bit-identical) and stores D = diag(S)^{−1/2} in `dscale[]`.

An element is determined by "row basis (A,B,C,d,j,k) + column term shift (x,y,z,dd,R,jl,k')". Four steps:
1. Count the distinct column shifts U and build the element index table (|U| × rows, Int32) per (shift, row).
2. For elements not in memory (`ctx.elems`), collect the kernels they need and compute those first in parallel (`prefetch`).
3. Compute the missing elements in parallel (heaviest first); values also go to `ctx.elems` for the next stage.
4. Fill the matrices by table lookup only (no hashing, no integrals).
"""
function assemble(hd::HamiltonianData{T}, ctx::IntegralContext{T}, es::ExponentSets, basis::Vector{Key};
                  rows::AbstractVector{Int}=1:length(basis),
                  ntasks::Int=Threads.nthreads(), cache_cap::Int=5_000_000, progress::Real=0, io::IO=stdout,
                  prefetch::Bool=true, packed::Bool=false, release_cache::Bool=false,
                  dscale=nothing) where {T}
    tab = _filled_table(hd, ctx, es, basis; rows=rows, ntasks=ntasks, cache_cap=cache_cap, prefetch=prefetch,
                        release_cache=release_cache)
    if packed && T === BigFloat && rows == 1:length(basis)
        return _fill_packed(tab, length(basis), ntasks; progress=progress, io=io, dscale=dscale)
    end
    _fill_dense(tab, length(basis), rows, ntasks; progress=progress, io=io)
end

"Steps 1–3: build and fill the element table for rows `rows` × all columns."
function _filled_table(hd::HamiltonianData{T}, ctx::IntegralContext{T}, es::ExponentSets, basis::Vector{Key};
                       rows::AbstractVector{Int}, ntasks::Int, cache_cap::Int=5_000_000, prefetch::Bool=true,
                       release_cache::Bool=false) where {T}
    close_exponent_sets!(hd, es, basis)
    Hφ = _apply_all(hd, es, basis, hd.split ? Mono6 : Mono)
    if hd.L > 0 || _gaussian(es)
        tab = _element_table_direct(Hφ, es, basis, rows, ntasks, hd.L, ctx; unnatural=hd.unnatural)   # fills the values too
    else
        tab = _element_table(Hφ, es, basis, rows, ntasks)
        _compute_elements!(tab, ctx, ntasks, cache_cap; prefetch=prefetch && !ctx.split)
    end   # also with one task (batched computation per table is faster than one by one)
    if release_cache
        # Last stage: the integral memory (kernels and elements) is not needed again, so release it before allocating the
        # matrices. It is comparable in size to H+S; GC alone leaves the space in the heap, so malloc_trim returns it to the OS.
        Integrals._clear!(ctx.store); Integrals._clear!(ctx.elems)
        _release!()
    end
    tab
end

"Timer that counts finished columns and prints progress every `interval` seconds (nothing if `interval ≤ 0`)."
function _progress_timer(done::Threads.Atomic{Int}, M::Int, interval::Real, io::IO)
    interval > 0 || return nothing
    t0 = time()
    Timer(interval; interval=interval) do _
        d = done[]; el = time() - t0
        rest = d == 0 ? "?" : string(round(Int, el * (M - d) / d))
        @printf(io, "  assembly %3d%% (%d/%d columns) elapsed %ds remaining ~%ss\n", round(Int, 100d / M), d, M, round(Int, el), rest)
        flush(io)
    end
end

function _apply_all(hd::HamiltonianData{T}, es::ExponentSets, basis::Vector{Key}, ::Type{K}) where {T,K}
    Hφ = Vector{Vector{Tuple{K,Int,Int,T}}}(undef, length(basis))
    Threads.@threads for jj in eachindex(basis)
        Hφ[jj] = apply_H(hd, es, basis[jj]; grow=false)
    end
    Hφ
end

# --- Element table ----------------------------------------------------------------
#
# An element is determined by row basis b = (A,B,C,d,j,k) + column term shift o = (x,y,z,dd,R,jl,k'); its key is the same 8-tuple as
# in memory (`ctx.elems`):
#     (A+x+1, B+y+1, C+z+1, d+dd, (j+jl) + 4R, α_k+α_k', β_k+β_k', γ_k+γ_k')   (exponents as Float64 bits)
# (the S shift of column j is φ_j's own (A_j,B_j,C_j,d_j,0,j_j,k_j); log1 terms have R = 0). The number of distinct shifts |U|
# is orders of magnitude below the number of column terms.

"Element table: `idx[u, r]` is the element index for shift u and row r (columns contiguous per row)."
struct ElementTable{T}
    idx::Matrix{Int32}
    keys::Vector{NTuple{8,Int}}               # index → key
    vals::Vector{T}                           # index → value (filled by _compute_elements!)
    sofs::Vector{Int32}                       # S shift of column j
    hterms::Vector{Vector{Tuple{Int32,T}}}    # H terms of column j (shift, coefficient), in Hφ order (summation order unchanged)
end

_offset(m::Mono, kk, aa) = (m[1], m[2], m[3], m[4], 0, m[5], kk, aa)
_offset(m::Mono6, kk, aa) = (m[1], m[2], m[3], m[4], m[5], m[6], kk, aa)

# Use every 16th lock: SpinLocks are allocated 16 B apart, four per cache line, and contended lines bounce between threads.
# All are kept alive while every 16th is used (freeing the others would let other small objects into the gaps).
_spaced_locks(n::Int) = [Threads.SpinLock() for _ in 1:16n]
function _element_table(Hφ::Vector{Vector{Tuple{K,Int,Int,T}}}, es::ExponentSets, basis::Vector{Key}, rows, ntasks::Int) where {K,T}
    M = length(basis); R = length(rows)
    uid = Dict{NTuple{8,Int},Int32}(); offs = NTuple{8,Int}[]
    getu(o) = get!(uid, o) do
        push!(offs, o); Int32(length(offs))
    end
    sofs = Int32[getu((b[1], b[2], b[3], b[4], 0, b[5], b[6], b[7])) for b in basis]
    hterms = [Tuple{Int32,T}[(getu(_offset(m, kk, aa)), c) for (m, kk, aa, c) in Hφ[j]] for j in 1:M]
    nU = length(offs)
    idx = Matrix{Int32}(undef, nU, R)
    # key → index: per-shard dictionaries, then (shard, index in shard) is converted to a global index
    ns = 256
    shards = [Dict{NTuple{8,Int},Int32}() for _ in 1:ns]; locks = _spaced_locks(ns)
    α, β, γ = es.α, es.β, es.γ
    @sync for c in Iterators.partition(1:R, cld(R, 4ntasks))
        Threads.@spawn for r in c
            b = basis[rows[r]]
            @inbounds for u in 1:nU
                o = offs[u]
                key = (b[1] + o[1] + 1, b[2] + o[2] + 1, b[3] + o[3] + 1, b[4] + o[4], b[5] + o[6] + 4o[5],
                       reinterpret(Int, α[b[6]] + α[o[7]]), reinterpret(Int, β[b[6]] + β[o[7]]), reinterpret(Int, γ[b[6]] + γ[o[7]]))
                sh = Int(hash(key) % ns) + 1
                lk = locks[16sh]; lock(lk); d = shards[sh]; id = get!(d, key, Int32(length(d) + 1)); unlock(lk)
                idx[u, r] = Int32(sh - 1) << 23 | id
            end
        end
    end
    maximum(length, shards) < 2^23 || error("too many distinct elements (at most 2^23 per shard)")
    base = cumsum([0; length.(shards)[1:end-1]])
    keys = Vector{NTuple{8,Int}}(undef, sum(length, shards))
    for sh in 1:ns, (k, id) in shards[sh]; keys[base[sh]+id] = k; end
    Threads.@threads for r in 1:R
        @inbounds for u in 1:nU
            x = idx[u, r]
            idx[u, r] = Int32(base[(x >> 23) + 1]) + (x & Int32(2^23 - 1))
        end
    end
    ElementTable{T}(idx, keys, Vector{T}(undef, length(keys)), sofs, hterms)
end

# Direct table (L ≥ 1 or the r₁₂ Gaussian). Values are filled when the table is built (no kernel prefetch of _compute_elements!).
# L ≥ 1: each pair of row component a and column term component c is multiplied by the angular weight W_ac = Σ w r₁^p r₂^q r₁₂^s
# (`Angular.angular_weight`). Gaussian (η > 0): elements from `element_gauss`, with the Q tables per (a, c, η) built first in
# parallel and only read afterwards. Keys are the 8-tuple plus (η, a, c); values are Σ w × element(A+p, B+q, C+s, …).
function _element_table_direct(Hφ, es::ExponentSets, basis::Vector{Key}, rows, ntasks::Int, L::Int, ctx::IntegralContext{T};
                               unnatural::Bool=false) where {T}
    M = length(basis); R = length(rows)
    uid = Dict{NTuple{8,Int},Int32}(); offs = NTuple{8,Int}[]
    getu(o) = get!(uid, o) do
        push!(offs, o); Int32(length(offs))
    end
    sofs = Int32[getu((b[1], b[2], b[3], b[4], 0, b[5], b[6], b[7])) for b in basis]
    hterms = [Tuple{Int32,T}[(getu(_offset(m, kk, aa)), c) for (m, kk, aa, c) in Hφ[j]] for j in 1:M]
    nU = length(offs)
    idx = Matrix{Int32}(undef, nU, R)
    α, β, γ, η = es.α, es.β, es.γ, es.η
    # key → index with per-shard dictionaries in parallel, as in _element_table
    ns = 256
    shards = [Dict{NTuple{11,Int},Int32}() for _ in 1:ns]; locks = _spaced_locks(ns)
    @sync for c in Iterators.partition(1:R, cld(R, 4ntasks))
        Threads.@spawn for r in c
            b = basis[rows[r]]
            @inbounds for u in 1:nU
                o = offs[u]
                key = (b[1] + o[1] + 1, b[2] + o[2] + 1, b[3] + o[3] + 1, b[4] + o[4], b[5] + o[6] + 4o[5],
                       reinterpret(Int, α[b[6]] + α[o[7]]), reinterpret(Int, β[b[6]] + β[o[7]]), reinterpret(Int, γ[b[6]] + γ[o[7]]),
                       reinterpret(Int, η[b[6]] + η[o[7]]), b[7], o[8])
                sh = Int(hash(key) % ns) + 1
                lk = locks[16sh]; lock(lk); d = shards[sh]; id = get!(d, key, Int32(length(d) + 1)); unlock(lk)
                idx[u, r] = Int32(sh - 1) << 23 | id
            end
        end
    end
    maximum(length, shards) < 2^23 || error("too many distinct elements (at most 2^23 per shard)")
    base = cumsum([0; length.(shards)[1:end-1]])
    keys = Vector{NTuple{11,Int}}(undef, sum(length, shards))
    for sh in 1:ns, (k, id) in shards[sh]; keys[base[sh]+id] = k; end
    Threads.@threads for r in 1:R
        @inbounds for u in 1:nU
            x = idx[u, r]
            idx[u, r] = Int32(base[(x >> 23) + 1]) + (x & Int32(2^23 - 1))
        end
    end
    # angular weights only for component pairs that occur in the keys (stage n uses ~n components at the ends)
    W = Dict((a, c) => [(p, T(w)) for (p, w) in (L == 0 ? [((0, 0, 0), 1 // 1)] : Angular.angular_weight(L, a, c; unnatural=unnatural))]
             for (a, c) in unique((k[10], k[11]) for k in keys))
    # Gaussian tables: p, q ranges per (a, c, η), including the shifts by W
    need = Dict{NTuple{3,Int},NTuple{4,Int}}()
    for k in keys
        iszero(reinterpret(Float64, k[9])) && continue
        k[5] == 0 || throw(ArgumentError("the r₁₂ Gaussian requires no logarithm (j = 0)"))
        for ((p, q, s), _) in W[(k[10], k[11])]
            (pr, qr) = gauss_range(k[1] + p, k[2] + q, k[3] + s, k[4])
            g = (k[6], k[8], k[9])
            o = get(need, g, (typemax(Int), typemin(Int), typemax(Int), typemin(Int)))
            need[g] = (min(o[1], pr[1]), max(o[2], pr[2]), min(o[3], qr[1]), max(o[4], qr[2]))
        end
    end
    glist = collect(need)
    gtabs = Vector{GaussTable{T}}(undef, length(glist))
    Threads.@threads for i in eachindex(glist)
        local g, r_ = glist[i]
        gtabs[i] = gauss_table(T, reinterpret(Float64, g[1]), reinterpret(Float64, g[2]), reinterpret(Float64, g[3]),
                               (r_[1], r_[2]), (r_[3], r_[4]))    # α = β, so e^{−a r₁ − a r₂} = e^{−a s}
    end
    gtab = Dict(glist[i][1] => gtabs[i] for i in eachindex(glist))
    vals = Vector{T}(undef, length(keys))
    if all(k -> iszero(reinterpret(Float64, k[9])), keys)
        # No Gaussian: collect the 8-tuple elements shifted by the angular weights without duplicates, compute them with the
        # L = 0 machinery (kernel prefetch, grouping by J, heaviest first), then sum.
        # Elements whose angular integral vanishes exactly are skipped: if the σ and r₁₂ exponents do not depend on x = cos θ₁₂
        # (βσ = 0, not split, γ sum 0) and the basis power C = k[3] − 1 of r₁₂ is even with 0 ≤ C < 2 l_min(a, c), then
        # ∫ r₁₂^C W_ac dx = 0 (mirror pairs have (L+1)(L+2)/2 terms whose sum cancels to 0).
        xfree = !ctx.split && isone(ctx.λ)
        lmin = Dict(ac => (L == 0 || !xfree ? 0 : Angular.angular_lmin(L, ac...; unnatural=unnatural)) for ac in Base.keys(W))
        zeroel(k) = xfree && iszero(reinterpret(Float64, k[8])) && (C = k[3] - 1; C >= 0 && iseven(C) && C < 2lmin[(k[10], k[11])])
        # Odd C and even C ≥ 2 l_min use partial waves (Sack's expansion, src/partialwave.jl): one-dimensional quadrature per
        # Legendre component of W, with a term count independent of L and no cancellation. Only l_min > 0 (a ≠ c) and powers of
        # ln σ ≤ 2. `_PW_ENABLED[] = false` switches back to the monomial path (for checks).
        comps = Dict(ac => (L >= 1 && xfree && _PW_ENABLED[] ? Angular.legendre_components(L, ac...; unnatural=unnatural) : nothing) for ac in Base.keys(W))
        pwel(k) = xfree && iszero(reinterpret(Float64, k[8])) && comps[(k[10], k[11])] !== nothing && k[5] <= 2 &&
                  (C = k[3] - 1; C >= -1)
        loc = [Set{NTuple{8,Int}}() for _ in 1:ntasks]
        @sync for (t, c) in enumerate(Iterators.partition(eachindex(keys), cld(length(keys), ntasks)))
            Threads.@spawn for i in c
                local k = keys[i]
                (zeroel(k) || pwel(k)) && continue
                for ((p, q, s), _) in W[(k[10], k[11])]
                    push!(loc[t], (k[1] + p, k[2] + q, k[3] + s, k[4], k[5], k[6], k[7], k[8]))
                end
            end
        end
        bkeys = sort!(collect(reduce(union!, loc)))
        bid = Dict(k => Int32(i) for (i, k) in enumerate(bkeys))
        btab = ElementTable{T}(Matrix{Int32}(undef, 0, 0), bkeys, Vector{T}(undef, length(bkeys)), Int32[], Vector{Tuple{Int32,T}}[])
        isempty(bkeys) || _compute_elements!(btab, ctx, ntasks, 5_000_000; prefetch=!ctx.split)
        Threads.@threads for i in eachindex(keys)
            local k = keys[i]
            local v = zero(T)
            zeroel(k) && (vals[i] = v; continue)
            if pwel(k)
                vals[i] = pw_element(ctx, comps[(k[10], k[11])], k[1] + 1, k[2] + 1, k[3] - 1, k[4], k[5],
                                     reinterpret(Float64, k[6]), reinterpret(Float64, k[7]))
                continue
            end
            for ((p, q, s), w) in W[(k[10], k[11])]
                v += w * btab.vals[bid[(k[1] + p, k[2] + q, k[3] + s, k[4], k[5], k[6], k[7], k[8])]]
            end
            vals[i] = v
        end
        return ElementTable{T}(idx, NTuple{8,Int}[], vals, sofs, hterms)
    end
    next = Threads.Atomic{Int}(1)
    @sync for _ in 1:ntasks
        Threads.@spawn begin
            local ec = ElementCache{T}()
            while (i = Threads.atomic_add!(next, 1)) <= length(keys)
                local k = keys[i]
                local v = zero(T)
                local gt = iszero(reinterpret(Float64, k[9])) ? nothing : gtab[(k[6], k[8], k[9])]
                for ((p, q, s), w) in W[(k[10], k[11])]
                    v += w * (gt === nothing ?
                              _element_from_key!(ec, ctx, (k[1] + p, k[2] + q, k[3] + s, k[4], k[5], k[6], k[7], k[8])) :
                              element_gauss(gt, ctx.binom, k[1] + p, k[2] + q, k[3] + s, k[4]))
                end
                vals[i] = v
            end
        end
    end
    ElementTable{T}(idx, NTuple{8,Int}[], vals, sofs, hterms)
end

"Element from its key (taken from `ctx.elems` if present)."
function _element_from_key!(ec::ElementCache{T}, ctx::IntegralContext{T}, key::NTuple{8,Int}) where {T}
    A, B, C, D, code, ab, bb, cb = key
    a, b, c = reinterpret(Float64, ab), reinterpret(Float64, bb), reinterpret(Float64, cb)
    ctx.split ? element_split!(ec, ctx, A, B, C, D, code >> 2, code & 3, a, b, c) : element!(ec, ctx, A, B, C, D, code, a, b, c)
end

"Fill the values of the table: for elements not in memory, compute the needed kernels first in parallel (`prefetch`), then the elements in parallel, heaviest first."
function _compute_elements!(tab::ElementTable{T}, ctx::IntegralContext{T}, ntasks::Int, cache_cap::Int; prefetch::Bool=true) where {T}
    nE = length(tab.keys)
    missing_ = [Int[] for _ in 1:ntasks]
    @sync for (t, c) in enumerate(Iterators.partition(1:nE, cld(nE, ntasks)))
        Threads.@spawn for id in c
            v = _store_get(ctx.elems, tab.keys[id])
            v === nothing ? push!(missing_[t], id) : (tab.vals[id] = v)
        end
    end
    todo = reduce(vcat, missing_)
    isempty(todo) && return tab
    wt(id) = (k = tab.keys[id]; k[1] + k[2] + k[3] + abs(k[4]) + 4(k[5] & 3))
    sort!(todo; by=wt, rev=true)
    # hand out elements heaviest first with an atomic counter; each task returns its local memory
    run(f, list=todo; kfrozen=false) = begin
        next = Threads.Atomic{Int}(1)
        ecs = [ElementCache{T}(; cap=cache_cap, kfrozen=kfrozen) for _ in 1:ntasks]
        @sync for t in 1:ntasks
            Threads.@spawn begin
                ec = ecs[t]
                while (i = Threads.atomic_add!(next, 1)) <= length(list)
                    f(ec, list[i])
                end
            end
        end
        ecs
    end
    if ctx.split
        # split elements (element_split! and below) do not use the Laplace kernels, so there is no kernel prefetch.
        ecs = run((ec, id) -> (tab.vals[id] = _element_from_key!(ec, ctx, tab.keys[id])))
        get(ENV, "TBFC_ECDIAG", "") == "1" && _ecdiag(ecs, ctx)
        return tab
    end
    # Elements differing only in J (same (A,B,C,D; a,b,c)) use the same kernels and share monomial sums, so each group is
    # done once (kernel recording and element computation; bit-identical to one at a time).
    groups = Dict{NTuple{8,Int},Vector{Int}}()
    for id in todo                              # todo is heaviest first; groups in the order of their first element
        k = tab.keys[id]
        push!(get!(() -> Int[], groups, (k[1], k[2], k[3], k[4], 0, k[6], k[7], k[8])), id)
    end
    gl = collect(values(groups)); sort!(gl; by=g -> wt(g[1]), rev=true)
    jmax = [maximum(id -> tab.keys[id][5], g) for g in gl]
    if prefetch
        # Collect the needed kernel keys task-locally (no element arithmetic) and compute them in parallel first. The shared
        # kernel memory is then read-only, so element computation reads kernels without locks.
        rec = _recording_context(ctx)
        prefetch_kernels!(ctx, rec, run((ec, gi) -> (k = tab.keys[gl[gi][1]];
                                                     _record_element!(ec, rec, (k[1], k[2], k[3], k[4], jmax[gi], k[6], k[7], k[8]))),
                                        eachindex(gl)), ntasks)
    end
    run((ec, gi) -> begin
            g = gl[gi]; k = tab.keys[g[1]]
            v = elements_upto!(ec, ctx, k[1], k[2], k[3], k[4], jmax[gi],
                               reinterpret(Float64, k[6]), reinterpret(Float64, k[7]), reinterpret(Float64, k[8]))
            for id in g; tab.vals[id] = v[tab.keys[id][5]+1]; end
        end, eachindex(gl); kfrozen=prefetch)
    tab
end

"Diagnostics (TBFC_ECDIAG=1): entries and size (GB) of each field of the task-local memories (first task and total) and of the shared stores."
function _ecdiag(ecs, ctx)
    for f in (:store, :elems, :monos, :angys, :dpows, :epsgs, :perims)
        hasfield(typeof(ctx), f) || continue
        st = getfield(ctx, f)
        @printf("  shared %-6s %10d entries %8.3f GB (cap %d)\n", f, sum(length, st.shards), sum(Base.summarysize, st.shards) / 1e9, st.cap)
    end
    for f in fieldnames(typeof(ecs[1]))
        v = getfield(ecs[1], f); v isa AbstractDict || continue
        n1 = length(v); s1 = Base.summarysize(v) / 1e9
        nall = sum(ec -> length(getfield(ec, f)), ecs)
        @printf("  memory %-6s first %9d entries %7.3f GB, all %d %10d entries\n", f, n1, s1, length(ecs), nall)
    end
    flush(stdout)
end

# Fill dense matrices; the summation order is the order of the column terms (Hφ). With all rows, H is symmetric and only one
# side per pair i < j is computed (`_hside`); with a subset of rows everything (raw unsymmetrized H).
function _fill_dense(tab::ElementTable{T}, M::Int, rows, ntasks::Int; progress::Real=0, io::IO=stdout) where {T}
    R = length(rows)
    full = rows == 1:M
    H = zeros(T, R, M); S = zeros(T, R, M)
    idx, vals = tab.idx, tab.vals
    nt = length.(tab.hterms)
    done = Threads.Atomic{Int}(0)
    timer = _progress_timer(done, M, progress, io)
    next = Threads.Atomic{Int}(1)
    @sync for _ in 1:ntasks
        Threads.@spawn while (jj = Threads.atomic_add!(next, 1)) <= M
            su = tab.sofs[jj]; ht = tab.hterms[jj]
            @inbounds for r in 1:R
                S[r, jj] = vals[idx[su, r]]
                full && r != jj && _hside(nt, r, jj) != jj && continue
                h = zero(T)
                for (u, c) in ht
                    h += c * vals[idx[u, r]]
                end
                H[r, jj] = h
            end
            Threads.atomic_add!(done, 1)
        end
    end
    timer === nothing || close(timer)
    full || return (H, S)
    @inbounds for j in 1:M, i in 1:j-1
        _hside(nt, i, j) == j ? (H[j, i] = H[i, j]) : (H[i, j] = H[j, i])
    end
    (H, S)
end

"""
Side used for H_ij = H_ji (i ≠ j): the column c with fewer Hφ terms, against the other row (larger index on ties).
Returns c (H[other, c] = Σ over Hφ_c). H is analytically symmetric, so both sides differ only by rounding.
"""
@inline function _hside(nt, i, j)
    lo, hi = minmax(i, j)
    @inbounds nt[lo] < nt[hi] ? lo : hi
end

# Fill packed storage. The task for column j writes, for pairs i ≤ j, ⟨φ_i|Hφ_j⟩ or ⟨φ_j|Hφ_i⟩ (side with fewer terms, `_hside`)
# to the lower triangle (j, i) (same values and operation order as dense, bit-identical). Products and sums use in-place MPFR.
# Element values and H coefficients are copied into contiguous MPFR structs (PackedMatrix) and passed as pointers (a BigFloat
# object is checked on every call and lives at scattered heap addresses).
function _fill_packed(tab::ElementTable{BigFloat}, M::Int, ntasks::Int; progress::Real=0, io::IO=stdout,
                      dscale=nothing)
    p = precision(BigFloat)
    H = SymPacked(M; precision=p); S = SymPacked(M; precision=p)
    idx = tab.idx
    nt = length.(tab.hterms)
    PV = PackedMatrix(length(tab.vals), 1; precision=p)
    off = cumsum([0; nt[1:end-1]])
    PC = PackedMatrix(max(sum(nt), 1), 1; precision=p)
    hu = [Int32[u for (u, _) in t] for t in tab.hterms]
    Threads.@threads for i in eachindex(tab.vals); PV[i, 1] = tab.vals[i]; end
    Threads.@threads for j in 1:M
        for (t, (_, c)) in enumerate(tab.hterms[j]); PC[off[j] + t, 1] = c; end
    end
    # when normalizing while filling, build D first from the diagonal S_ii (same values as filled; same formula as diagonal_scale!)
    D = dscale === nothing ? BigFloat[] : [inv(sqrt(PV[idx[tab.sofs[i], i], 1])) for i in 1:M]
    sc = dscale !== nothing
    done = Threads.Atomic{Int}(0)
    timer = _progress_timer(done, M, progress, io)
    # hand out packed columns ii (rows jj ≥ ii, contiguous in memory); column length is M − ii + 1, so small ii (long columns) first
    next = Threads.Atomic{Int}(1)
    GC.@preserve H S PV PC @sync for _ in 1:ntasks
        Threads.@spawn begin
            h1, tmp = BigFloat(0; precision=p), BigFloat(; precision=p)
            GC.@preserve h1 tmp begin
                ph, pt = mpfr_ptr(h1), mpfr_ptr(tmp)
                while (ii = Threads.atomic_add!(next, 1)) <= M
                    @inbounds for jj in ii:M
                        mpfr_set!(_el(S, jj, ii), _el(PV, idx[tab.sofs[jj], ii], 1))
                        if ii == jj || _hside(nt, ii, jj) == jj
                            _hsum!(ph, pt, hu[jj], off[jj], idx, ii, PV, PC)       # ⟨φ_i|Hφ_j⟩
                        else
                            _hsum!(ph, pt, hu[ii], off[ii], idx, jj, PV, PC)       # ⟨φ_j|Hφ_i⟩
                        end
                        mpfr_set!(_el(H, jj, ii), ph)
                        sc && _scale_el!(S, H, D, jj, ii)
                    end
                    Threads.atomic_add!(done, 1)
                end
            end
        end
    end
    timer === nothing || close(timer)
    sc && (dscale[] = D)
    (H, S)
end

"Multiply element (i, j) (i ≥ j) of S, H by D_i then D_j (same order and rounding as `diagonal_scale!`)."
@inline function _scale_el!(S::SymPacked, H::SymPacked, D::Vector{BigFloat}, i::Int, j::Int)
    for P in (S, H)
        x = _el(P, i, j); mpfr_mul!(x, D[i], x); mpfr_mul!(x, x, D[j])
    end
end

# Memory latency of the scattered reads of PV dominates; prefetch the PV elements (two lines) of term t+D and the idx of term
# t+2D during the product of term t (an MPFR product is a long call that out-of-order execution does not look past).
# A fused multiply-add is not faster here and would change the rounding.
const _PFD = 4
"Prefetch the cache line at address p (llvm.prefetch, read, locality 3)."
@inline _pf(p::Ptr) = Base.llvmcall(("""
    declare void @llvm.prefetch.p0(ptr, i32, i32, i32)
    define void @entry(i64 %0) #0 {
      %p = inttoptr i64 %0 to ptr
      call void @llvm.prefetch.p0(ptr %p, i32 0, i32 3, i32 1)
      ret void
    }
    attributes #0 = { alwaysinline }""", "entry"), Cvoid, Tuple{UInt64}, UInt64(p))
@inline _pf_el(P, i) = (q = _el(P, i, 1); for o in 0:64:8P.stride-1; _pf(q + o); end)   # all lines of an element
@inline _pf_idx(idx, u, r) = _pf(pointer(idx, Int(u) + (r - 1) * size(idx, 1)))
"h = Σ_t c_t · v[idx[u_t, r]] (coefficients from rows o+t of PC, values from PV; one rounding per product and per sum, like h += c*v)."
@inline function _hsum!(h::Ptr{BigFloat}, tmp::Ptr{BigFloat}, us::Vector{Int32}, o::Int, idx, r, PV, PC)
    ccall((:mpfr_set_zero, :libmpfr), Cvoid, (Ptr{BigFloat}, Cint), h, 1)   # +0 (same as zero(T))
    n = length(us)
    @inbounds for t in 1:n
        t + 2 * _PFD <= n && _pf_idx(idx, us[t+2 * _PFD], r)
        t + _PFD <= n && _pf_el(PV, idx[us[t+_PFD], r])
        mpfr_mul!(tmp, _el(PC, o + t, 1), _el(PV, idx[us[t], r], 1)); mpfr_add!(h, h, tmp)
    end
    h
end

# Fill the matrices of the exchange-symmetrized basis Φ_a = φ_{r_a} + s·φ_{P r_a} (r = representatives) directly in packed
# storage; the table rows are the representatives only. H' is analytically symmetric (PHP = H), so one side per pair a ≥ b is
# computed: the side whose column terms (Hφ_r and Hφ_{Pr}) are fewer, against the other row (larger index on ties, as `_hside`).
# S' likewise (S_{a,Pb} and S_{b,Pa} have mirror keys under r₁ ↔ r₂ and differ only by rounding). Only two packed M/2 matrices
# are allocated, compared with the dense path (representative rows × all columns, then (H' + H'ᵀ)/2).
function _fill_packed_exchange(tab::ElementTable{BigFloat}, reps::Vector{Int}, mirror::Vector{Int}, sgn::Int, ntasks::Int;
                               progress::Real=0, io::IO=stdout,
                               dscale=nothing)
    p = precision(BigFloat)
    Mx = length(reps)
    H = SymPacked(Mx; precision=p); S = SymPacked(Mx; precision=p)
    idx = tab.idx
    nt = length.(tab.hterms)
    ntx = [nt[r] + nt[mirror[r]] for r in reps]
    PV = PackedMatrix(length(tab.vals), 1; precision=p)
    off = cumsum([0; nt[1:end-1]])
    PC = PackedMatrix(max(sum(nt), 1), 1; precision=p)
    hu = [Int32[u for (u, _) in t] for t in tab.hterms]
    Threads.@threads for i in eachindex(tab.vals); PV[i, 1] = tab.vals[i]; end
    Threads.@threads for j in eachindex(tab.hterms)
        for (t, (_, c)) in enumerate(tab.hterms[j]); PC[off[j] + t, 1] = c; end
    end
    addsub! = sgn > 0 ? mpfr_add! : mpfr_sub!
    # when normalizing while filling, build D first (with the same two operations as when filling S'_aa)
    sc = dscale !== nothing
    D = BigFloat[]
    if sc
        D = Vector{BigFloat}(undef, Mx)
        Threads.@threads for a in 1:Mx
            ra = reps[a]; pa = mirror[ra]
            x = BigFloat(; precision=p)
            GC.@preserve x begin
                mpfr_set!(mpfr_ptr(x), _el(PV, idx[tab.sofs[ra], a], 1))
                addsub!(mpfr_ptr(x), mpfr_ptr(x), _el(PV, idx[tab.sofs[pa], a], 1))
            end
            D[a] = inv(sqrt(x))
        end
    end
    done = Threads.Atomic{Int}(0)
    timer = _progress_timer(done, Mx, progress, io)
    next = Threads.Atomic{Int}(1)
    GC.@preserve H S PV PC @sync for _ in 1:ntasks
        Threads.@spawn begin
            h1, h2, tmp = BigFloat(0; precision=p), BigFloat(0; precision=p), BigFloat(; precision=p)
            GC.@preserve h1 h2 tmp begin
                p1, p2, pt = mpfr_ptr(h1), mpfr_ptr(h2), mpfr_ptr(tmp)
                while (a = Threads.atomic_add!(next, 1)) <= Mx   # packed column a (rows b ≥ a, contiguous); long columns (small a) first, as in _fill_packed
                    ra = reps[a]; pa = mirror[ra]
                    @inbounds for b in a:Mx
                        rb = reps[b]; pb = mirror[rb]
                        mpfr_set!(_el(S, b, a), _el(PV, idx[tab.sofs[rb], a], 1))
                        addsub!(_el(S, b, a), _el(S, b, a), _el(PV, idx[tab.sofs[pb], a], 1))
                        if a == b || (ntx[a] < ntx[b] ? a : b) == b
                            _hsum!(p1, pt, hu[rb], off[rb], idx, a, PV, PC)        # ⟨φ_{r_a}|Hφ_{r_b}⟩
                            _hsum!(p2, pt, hu[pb], off[pb], idx, a, PV, PC)        # ⟨φ_{r_a}|Hφ_{P r_b}⟩
                        else
                            _hsum!(p1, pt, hu[ra], off[ra], idx, b, PV, PC)        # ⟨φ_{r_b}|Hφ_{r_a}⟩
                            _hsum!(p2, pt, hu[pa], off[pa], idx, b, PV, PC)        # ⟨φ_{r_b}|Hφ_{P r_a}⟩
                        end
                        addsub!(_el(H, b, a), p1, p2)
                        sc && _scale_el!(S, H, D, b, a)
                    end
                    Threads.atomic_add!(done, 1)
                end
            end
        end
    end
    timer === nothing || close(timer)
    sc && (dscale[] = D)
    (H, S)
end
