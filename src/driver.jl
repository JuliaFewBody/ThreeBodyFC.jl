# Driver: per stage, generate → assemble → symmetrize → normalize → solve.

using Printf

_parity(::NoSymmetry) = 0
_parity(s::Exchange) = s.sign

"""
Hamiltonian used for generation. Split log variables (the new logarithmic form) generate the function set in the βσ = 0
ring (Eqs. 28 and 29 of JCP 127, 224104 give the same index set; the M_n of Table IV log1 and Table V agree at all stages).
"""
_generation_hamiltonian(sys, lv::LogVariable, hd, ::Type{T}) where {T} =
    lv.split ? HamiltonianData(sys, LogVariable(0.0), T; L=hd.L, unnatural=hd.unnatural) : hd

"Result of one stage."
Base.@kwdef struct StageResult{T}
    n::Int
    M::Int                    # number of generated functions
    Msolved::Int              # dimension after symmetrization (M_n of the papers); the solved matrix is smaller by nexact
    E::T
    error::Float64            # difference from the reference (NaN without one)
    solution::Solution{T}
    basis::Vector{Key}
    t_build::Float64
    t_solve::Float64
    nexact::Int = 0           # exact linear dependencies removed before assembly (drop_exact)
    states::Vector{Solution{T}} = Solution{T}[]   # with nstates > 1, all states from the bottom (states[1] = solution)
    errors::Vector{Float64} = Float64[]           # difference from the reference per state
    t_states::Vector{Float64} = Float64[]         # solve time per state (states[1] is t_solve)
    estimates::Vector{Float64} = Float64[]        # reference-free error estimate per state (`tail_estimate`)
    bound::Vector{Bool} = Bool[]                  # E below the dissociation threshold (bound for certain, E being an upper bound)
    order::NTuple{3,Int} = (1, 2, 3)              # internal particle a = input particle order[a] (`orient`)
end

"""
    tail_estimate(Δ, Δprev) -> Float64

Reference-free error estimate E_n − E_∞ ≈ Δ r/(1−r) (Δ = E_{n−1} − E_n, r = Δ/Δprev, assuming the remaining stages decrease
geometrically). NaN unless r ∈ (0, 1). On ground and excited states of He and H⁻ with references (n ≥ 4, 130 points) the
ratio estimate/true error has median 0.99 and 5–95% range 0.85–1.33.
"""
function tail_estimate(Δ::Real, Δprev::Real)
    (isfinite(Δ) && isfinite(Δprev) && Δ > 0 && Δprev > 0) || return NaN
    r = Δ / Δprev
    0 < r < 1 || return NaN
    Δ * r / (1 - r)
end

"Format a high-precision value with nd decimals (without going through Float64)."
fmt(x, nd::Int=21) = setprecision(BigFloat, max(256, ceil(Int, nd * 3.33) + 64)) do
    Printf.format(Printf.Format("%.$(nd)f"), _tobig(x))
end

function Base.show(io::IO, r::StageResult)
    @printf(io, "StageResult(n=%d, M=%d→%d, E=%s, err=%.3e, %s, build %.1fs, solve %.1fs)",
            r.n, r.M, r.Msolved, fmt(r.E, 25), r.error, r.solution.method, r.t_build, r.t_solve)
end
function Base.show(io::IO, ::MIME"text/plain", rs::AbstractVector{<:StageResult})
    println(io, length(rs), " stages:")
    # est: reference-free error estimate (— without two previous stages); err: difference from the reference (— without one);
    # ΔE: change from the previous stage; solved = dimension after symmetrization − exact dependencies
    @printf(io, "%4s %7s %7s  %-32s %11s %11s %11s  %-18s %9s %9s\n", "n", "M", "solved", "E", "est", "err", "ΔE", "method",
            "build[s]", "solve[s]")
    for (i, r) in enumerate(rs)
        est = isempty(r.estimates) || isnan(r.estimates[1]) ? "—" : @sprintf("%.3e", r.estimates[1])
        err = isnan(r.error) ? "—" : @sprintf("%.3e", r.error)
        dE = i == 1 ? "—" : @sprintf("%.3e", Float64(r.E - rs[i-1].E))
        above = !isempty(r.bound) && !r.bound[1] ? "  above threshold" : ""
        @printf(io, "%4d %7d %7d  %-32s %11s %11s %11s  %-18s %9.1f %9.1f%s\n", r.n, r.M, r.Msolved - r.nexact, fmt(r.E, 28), est,
                err, dE, r.solution.method, r.t_build, r.t_solve, above)
    end
end

# Stage n=0 is ψ₀ itself (the seed set); the generation chain (gen) for n ≥ 1 does not depend on whether n=0 is solved.
# Returns (input of the next stage, set solved at this stage).
_stage_range(stages) = (0 in stages ? 0 : 1):maximum(stages)
# Seed components: Exchange adds the mirror a ↔ L−a itself, so a = 0 only; without symmetrization a = 0 and L (`_maxcomp`)
_seed_comps(::Symmetry, L) = 0:0
_seed_comps(::NoSymmetry, L) = unique([0, L])
function _stage_basis(rule, hd, es, gen, n, symmetry)
    n == 0 && return gen, symmetrize_basis(symmetry, gen, es, _maxcomp(hd))
    b = symmetrize_basis(symmetry, next_stage(rule, hd, es, gen, n; parity=_parity(symmetry)), es, _maxcomp(hd))
    b, b
end

"""
    run_fc(sys, es; stages, T, rule, engine, logvar, symmetry, solver, normalize, reference, io, ndigits, ...)

Solve system `sys` by the FC method with exponent sets `es` (the caller's object is not modified). `stages` is the set of
stages n (number of FC iterations) to solve; the result is a vector of `StageResult`.

- `reference`: reference energy as a string (a Float64 is limited to 16 digits and triggers a warning); with `nstates > 1`
  a vector of strings per state.
- `progress = 30`: print assembly progress every 30 seconds.
- `drop_exact = true`: remove exact linear dependencies from σ = r₁ + r₂ by integer arithmetic before assembly (the span is
  unchanged; `StageResult.nexact` counts them). When the smallest eigenvalue of S (`r.solution.smin`) approaches eps·M of
  the working precision, the near dependence will exhaust the working precision at the next stage.
- `auto_precision = true`: instead of warning, re-solve the remaining stages in BigFloat (bits extrapolated from λmin; kernels
  in BigFloat too). The working precision does not change the result, so this is automatic. Bits are capped at `max_bits`
  (1024 by default); a stage that fails at the cap stops the run (the dense fallback runs only up to `dense_max`).
- `jump_ratio`: if E suddenly drops by more than this multiple of the previous decrease, a spurious eigenvalue is suspected
  and the stage is re-solved once with twice the bits; if that gives the same, the run stops.
- `max_mem_gb`: stages whose matrices (about `_MEM_FACTOR` = 2.2 × M² elements) would exceed it are not assembled.
  On a stop, the results so far are returned.
- `E0`, `dE0`: upper bound of the first stage's eigenvalue (e.g. the previous E) and its decrease; with them the first stage
  already uses inverse iteration on H − σS (later stages use the previous E automatically).
- `nstates = k`: the lowest k states of the symmetry (excited states by lifted inverse iteration, `solve_excited`;
  `r.states`, `r.errors`). The first stage without previous energies uses the dense eigendecomposition; without `E0`,
  lower stages are solved as warm-up (not in the results).
- `packed` (default: true for BigFloat): assemble and solve with packed lower triangles (`SymPacked`, about 1/3 of the memory).
- `L ≥ 1`: total angular momentum with natural parity; seed components a = 0 ((l₁,l₂) = (0,L)), a = 0 and L without
  symmetrization; `seed_components` (e.g. `0:L`) overrides. `unnatural = true`: unnatural parity ((−1)^{L+1}, l₁ + l₂ = L + 1,
  𝒴_a = (z₁u₂ − z₂u₁) u₁^a u₂^{L−1−a}, a = 0..L−1). `symmetry` is the exchange sign of the spatial wave function.
- Basis order: the previous stage's basis comes first in its previous order, new functions after (in key order), so the
  leading block of the matrix is the previous stage's matrix. `reuse_factor = true` then carries the Cholesky factor of
  H − σS over and factors only the new columns (`chol_lower_ozaki!(A; done=m)`, work ∝ M³ − m³). This applies to packed
  BigFloat, Ozaki-size matrices, no exchange projection, nstates = 1 and a previous energy, and needs `FCRule(dedup=false)`.
"""
function run_fc(sys::ThreeBody, es0::ExponentSets;
                stages=1:3, T::Type=Float64x4, rule::FCRule=FCRule(),
                engine::KernelEngine=KernelQuadrature(), logvar::LogVariable=LogVariable(),
                symmetry::Symmetry=NoSymmetry(), solver::Solver=default_solver(),
                normalize::Bool=true, reference=nothing, io::IO=stdout, ndigits::Int=21,
                ntasks::Int=Threads.nthreads(), outdir=nothing, checkpoint::Bool=outdir !== nothing,
                resume::Bool=true, save_wavefunction::Bool=outdir !== nothing, progress::Real=0,
                drop_exact::Bool=true, auto_precision::Bool=true, skip_fallback::Bool=false, max_bits::Int=1024, max_mem_gb::Real=_default_mem_gb(),
                jump_ratio::Real=100,
                E0=nothing, dE0=nothing,
                packed::Union{Nothing,Bool}=nothing, nstates::Int=1, context=nothing, threshold=:auto, L::Int=0,
                seed_components=nothing, unnatural::Bool=false, reuse_factor::Bool=true,
                _order::NTuple{3,Int}=(1, 2, 3), _retry_stage::Int=0)
    sys_in = sys
    sys, es0, π = orient(sys, es0, symmetry; io=io)    # reorder internally if the identical pair is not at 1, 2
    order = π == (1, 2, 3) ? _order : π
    es = copy(es0)
    check_symmetry(symmetry, sys)
    isone(logvar.w2) || symmetry isa NoSymmetry || throw(ArgumentError("w₂ ≠ 1 breaks the exchange invariance of σ; use NoSymmetry"))
    isone(logvar.w2) || !_gaussian(es0) || throw(ArgumentError("the r₁₂ Gaussian elements require w₂ = 1"))
    isone(logvar.w2) || L == 0 || throw(ArgumentError("w₂ ≠ 1 requires L = 0"))
    check_exponents(es)
    hd = HamiltonianData(sys, logvar, T; L=L, unnatural=unnatural)
    xsym = _component_symmetry(symmetry, hd); Lc = _maxcomp(hd)
    bw = binary_warning(es; shifts=[m for (_, m, _) in hd.expo])
    bw === nothing || (println(io, "warning: ", bw); flush(io))
    hdgen = _generation_hamiltonian(sys, logvar, hd, T)
    # context: integral memory shared with other runs (same T, engine, logvar; auto_exponents reuses it across candidates)
    ctx = context === nothing ? IntegralContext(T, engine, logvar) : context
    ctx isa IntegralContext{T} || throw(ArgumentError("context does not match T=$T"))
    cptag = sprint() do h
        describe(h, sys); describe(h, logvar); println(h, " T=", T); describe(h, engine)
        L > 0 && println(h, " L=", L)
        unnatural && println(h, " unnatural")
    end
    reference isa Union{Float64,Float32} &&
        @warn "reference is a $(typeof(reference)), limited to 16 digits; pass a string"
    _ref1(r) = r === nothing ? nothing : (r isa AbstractString ? exact(T, r) : T(r))
    refs = reference isa AbstractVector ? [_ref1(r) for r in reference] : [_ref1(reference)]
    ref = refs[1]
    nstates >= 1 || throw(ArgumentError("nstates ≥ 1 required"))

    # dissociation threshold (two-body subsystems): computed for :auto, a number is used as is, nothing disables the check
    th = threshold === :auto ? thresholds(sys_in) : threshold      # printed in the input numbering
    Eth = th isa Thresholds ? th.E : th === nothing ? Inf : Float64(th)
    hdr = sprint() do h
        println(h, "=== ThreeBodyFC run ===")
        describe(h, sys)
        println(h, "exponents α=", es.α, " β=", es.β, " γ=", es.γ, _gaussian(es) ? " η=$(es.η) (r₁₂ Gaussian)" : "")
        describe(h, logvar); println(h, "  T=", T, " eps=", Float64(eps(T)),
                T === BigFloat ? "  precision=$(precision(BigFloat))" : "", "  threads=", Threads.nthreads(),
                "  normalize=", normalize, "  stages=", collect(stages))
        describe(h, rule); describe(h, engine); describe(h, symmetry); describe(h, solver)
        ref === nothing || println(h, "reference=", join([r === nothing ? "—" : fmt(r, ndigits) for r in refs], ", "))
        nstates > 1 && println(h, "nstates=", nstates)
        L > 0 && println(h, "L=", L, unnatural ? " (unnatural parity, components 𝒴_a = (z₁u₂ − z₂u₁) u₁^a u₂^{L−1−a}, a = 0..L−1)" :
                                       " (natural parity, components 𝒴_a = u₁^a u₂^{L−a}, a = 0..L)")
        th isa Thresholds && describe(h, th)
        outdir === nothing || println(h, "outdir=", outdir, " checkpoint=", checkpoint, " resume=", resume)
    end
    print(io, hdr); flush(io)
    outdir === nothing || _write_header(outdir, hdr)

    results = StageResult{T}[]
    gen = seed_basis(rule, es, seed_components === nothing ? _seed_comps(symmetry, Lc) : seed_components)
    _vec(x) = x === nothing ? nothing : x isa AbstractVector ? collect(x) : [x]
    Eprevs = E0 === nothing ? nothing : T.(_vec(E0))       # previous energies per state (shift guides)
    dEprevs = _vec(dE0)                                    # previous decreases per state
    Eprev = Eprevs === nothing ? nothing : Eprevs[1]
    dEprev = dEprevs === nothing ? nothing : dEprevs[1]
    sminprev = NaN
    ordprev = Key[]                 # basis order of the previous stage (placed first in the next)
    fac = nothing; fackeys = Key[]  # factor of H − σS of the previous stage (L, σ, reused) and its basis
    ddnote = Ref(false)             # printed the dedup hint
    # Excited states without previous energies: solve lower stages first as warm-up (otherwise the first requested stage
    # starts from the dense canonical_roots). Only the starting point of the iteration changes, so this is automatic.
    warm = (nstates > 1 && E0 === nothing && _retry_stage == 0) ? [m for m in _stage_range(stages) if 1 <= m < minimum(stages)] : Int[]
    # Re-solve the remaining stages `rest` in BigFloat with `bits` bits (kernels in BigFloat). `retry` is the stage that hit the wall.
    function _rerun(rest, bits, retry)
        if context === nothing           # the integral memory so far is not reused; release it first
            Integrals._clear!(ctx.store); Integrals._clear!(ctx.elems)
        end
        _release!()
        setprecision(BigFloat, bits) do
            run_fc(sys, es0; stages=rest, T=BigFloat, rule=rule, engine=KernelQuadrature(BigFloat), logvar=logvar,
                   symmetry=symmetry, solver=solver, normalize=normalize, reference=reference, io=io, ndigits=ndigits,
                   ntasks=ntasks, outdir=outdir === nothing ? nothing : joinpath(outdir, "bigfloat_$bits"),
                   checkpoint=checkpoint, resume=resume, packed=packed,
                   save_wavefunction=save_wavefunction, progress=progress, drop_exact=drop_exact, auto_precision=true,
                   nstates=nstates, threshold=th, _order=order, _retry_stage=retry, L=L, seed_components=seed_components, unnatural=unnatural,
                   max_bits=max_bits, max_mem_gb=max_mem_gb, jump_ratio=jump_ratio, reuse_factor=reuse_factor,
                   # if no stage was solved in this call (nested re-solves), pass on the E0, dE0 received, so that the jump
                   # detector still works at the last precision
                   E0=isempty(results) ? E0 : [x.E for x in results[end].states],
                   dE0=length(results) >= 2 ? Float64.([x.E for x in results[end-1].states] .- [x.E for x in results[end].states]) :
                       length(results) == 1 ? (E0 === nothing ? nothing : Float64.(T.(_vec(E0)) .- [x.E for x in results[end].states][1:length(_vec(E0))])) : dE0)
        end
    end
    for n in _stage_range(stages)
        gen, basis = _stage_basis(rule, hdgen, es, gen, n, symmetry)
        basis = _prefix_order(ordprev, basis); ordprev = basis      # previous functions first, in their previous order
        iswarm = n in warm
        (n in stages || iswarm) || continue
        t0 = time(); c0 = cputime()
        Mgen, Mgen_sym = length(basis), _symdim(xsym, basis, es, Lc)
        nexact = 0
        # d is a power of s = r₁ + r₂ if split, otherwise of σ = r₁ + r₂ + βσ r₁₂ (the dependence expansion uses that variable)
        drop_exact && ((basis, nexact) = exact_independent(basis, es, xsym, Lc; c2=logvar.w2, c12=logvar.split ? 0.0 : logvar.βσ))
        Mdim = _symdim(xsym, basis, es, Lc)
        need = max(_matrix_gb(T, Mdim), _build_gb(T, Mdim))
        if need > max_mem_gb
            println(io, @sprintf("stop: stage n=%d has M=%d and needs about %.3g GB in %s (max_mem_gb=%.3g GB)", n, Mdim, need,
                                 T === BigFloat ? "BigFloat $(precision(BigFloat)) bits" : string(T), max_mem_gb))
            flush(io); return results
        end
        # carry the previous factor over if the leading block is the previous basis and the extra matrix fits
        if fac !== nothing
            mprev = fac.L.M; fgb = sympacked_elbytes(fac.L.prec) * mprev * (mprev + 1) / 2 / 1e9
            usable = reuse_factor && Mdim == length(basis) && length(basis) > mprev && view(basis, 1:mprev) == fackeys &&
                     fac.L.prec == _bits(T) && Eprev !== nothing
            # The old σ is farther from the new E; the per-iteration contraction is (E − σ)/gap, so the iteration count grows as
            # 1/log(contraction). With a gap of 1e-3|E|, skip the reuse if it would need more than 1.33× the iterations of the usual σ.
            if usable
                gap = 1e-3 * abs(Float64(Eprev)); dold = Float64(Eprev - fac.σ); dnew = 2 * (dEprev === nothing ? 1e-6 : Float64(dEprev))
                usable = 0 < dold < gap && 0 < dnew < gap && log(dold / gap) / log(dnew / gap) >= 0.75
            end
            usable && need + fgb > max_mem_gb &&
                (println(io, @sprintf("stage n=%d: carrying the previous factor (%.3g GB) would exceed max_mem_gb=%.3g GB; not reused", n, fgb, max_mem_gb)); usable = false)
            usable || (fac = nothing; fackeys = Key[]; _release!())
        end
        close_exponent_sets!(hd, es, basis)
        # Check that the elements fit the exponent range of Float64-based types (for high L the diagonal of S can exceed 1e500).
        # Assemble a few diagonal elements at the extremes; if out of range, switch to BigFloat without assembling.
        if T !== BigFloat
            lo, hi = _magnitude_probe(hd, ctx, es, basis, ntasks)
            if !(_MAG_LO < lo && hi < _MAG_HI)
                bits = max(256, 64 * cld(_bits(T), 64))
                what = isfinite(lo) && isfinite(hi) ? @sprintf("diagonal magnitudes 1e%.0f to 1e%.0f", lo, hi) : "non-finite or zero diagonal"
                if auto_precision && bits <= max_bits
                    println(io, "stage n=$n: ", what, " outside the exponent range of $T (about 1e$(_MAG_LO) to 1e$(_MAG_HI)); ",
                            "switching to BigFloat $bits bits with BigFloat kernels")
                    flush(io)
                    more = _rerun([m for m in stages if m >= n], bits, 0)
                    return StageResult[results; more]
                end
                println(io, "warning: stage n=$n: ", what, " outside the exponent range of $T; ", auto_precision ? "max_bits=$max_bits" : "auto_precision=false",
                        ", not switching (pass T=BigFloat)"); flush(io)
            end
        end
        loaded = (outdir !== nothing && resume) ? _load_checkpoint(outdir, n, basis, es, cptag) : nothing
        # Normalize while filling (packed, no checkpoint: checkpoints hold the unnormalized H, S; bit-identical to diagonal_normalize).
        # With its own integral memory, the last stage releases the memory during assembly.
        dsc = normalize && loaded === nothing && !(checkpoint && !iswarm) ? Ref{Any}(nothing) : nothing
        H, S, C = loaded === nothing ? assemble_projected(symmetry, hd, ctx, es, basis; ntasks=ntasks, progress=progress, io=io,
                                                                    packed=packed === nothing ? T === BigFloat : packed,
                                                                    release_cache=context === nothing && n == maximum(stages),
                                                                    dscale=dsc) : loaded
        loaded === nothing && checkpoint && !iswarm && _save_checkpoint(outdir, n, basis, es, H, S, C, cptag)
        D = dsc === nothing ? nothing : dsc[]
        normalize && D === nothing && ((H, S, D) = diagonal_normalize(H, S))
        _release!(size(S, 1))                     # release assembly scratch before solving
        t1 = time(); c1 = cputime()
        # If the precision can still be raised, a failed Cholesky path does not run the fallback (its result would be discarded).
        curbits = _bits(T)
        canraise = auto_precision && drop_exact && curbits < max_bits
        # skip_fallback: the caller (auto_exponents candidate evaluation) discards non-Cholesky results and re-solves itself
        skipfb = (canraise || skip_fallback) && solver isa CholeskyInverseIteration
        # Carry the factor to the next stage (packed BigFloat, no projection, ground state only, more stages to come).
        # dedup = true drops some previous keys at the next stage, so the leading block would differ: print a hint once instead.
        dd = rule isa FCRule && rule.dedup && rule.collect === :monomial
        kp0 = reuse_factor && solver isa CholeskyInverseIteration && T === BigFloat && H isa SymPacked && C === nothing &&
              nstates == 1 && n < maximum(stages)
        if kp0 && dd && !ddnote[] && solver.ozaki_min <= size(S, 1)
            println(io, "hint: FCRule(dedup=false) lets the next stage reuse this factorization (it spans the same space)")
            ddnote[] = true
        end
        kp = kp0 && !dd ? Ref{Any}(nothing) : nothing
        fac !== nothing && C !== nothing && (fac = nothing)
        okw = solver isa CholeskyInverseIteration ? (; reuse=fac, keep=kp) : (;)
        fac = nothing; fackeys = Key[]
        sol = skipfb ? solve(solver, H, S; σ0=Eprev, δ0=dEprev, nofallback=true, okw...) : solve(solver, H, S; σ0=Eprev, δ0=dEprev, okw...)
        okw = (;)                                 # drop the reference to the previous factor
        sol.method === :cholesky_failed && println(io, "stage n=$n: Cholesky path failed (", sol.note, "); ",
                                                   canraise ? "raising the precision instead of the fallback" : "precision cap max_bits=$max_bits")
        # Unusual drop: variational energies decrease nearly geometrically. A drop of more than jump_ratio times the previous
        # decrease suggests a spurious eigenvalue (fallback results included).
        jump = (isfinite(sol.E) && Eprev !== nothing && dEprev !== nothing && dEprev > 0) ?
               Float64(Eprev - sol.E) / dEprev : NaN
        suspicious = jump > jump_ratio
        suspicious && println(io, @sprintf("stage n=%d: E dropped by %.1e times the previous decrease (suspected spurious eigenvalue)", n, jump))
        t2 = time(); c2 = cputime()
        sols = Solution{T}[sol]; tst = [t2 - t1]
        # if the ground state hit the precision wall, this stage is re-solved, so excited states are skipped
        chol = solver isa CholeskyInverseIteration
        if nstates > 1 && !((auto_precision || skip_fallback) && drop_exact && ((chol && sol.method !== :cholesky) || suspicious))
            if Eprevs === nothing || length(Eprevs) < nstates   # no previous energies: dense eigendecomposition
                te = time()
                append!(sols, canonical_roots(H, S, nstates)[2:end])
                append!(tst, fill((time() - te) / (nstates - 1), nstates - 1))
            else
                _release!(size(S, 1))
                work = _work_like(S)
                for j in 2:nstates
                    te = time()
                    d = dEprevs === nothing || length(dEprevs) < j ? nothing : dEprevs[j]
                    push!(sols, solve_excited(solver isa PivotedSubset ? solver.inner : solver, H, S, sols; σ0=Eprevs[j], δ0=d, work=work))
                    push!(tst, time() - te)
                end
                work = nothing
            end
        end
        Msolve = size(S, 1)
        if kp !== nothing && kp[] !== nothing && sol.method === :cholesky
            fac = kp[]; fackeys = copy(basis)
        end
        kp = nothing
        H = S = nothing; _release!(Msolve)
        err = ref === nothing ? NaN : Float64(sol.E - ref)
        errs = [j <= length(refs) && refs[j] !== nothing ? Float64(sols[j].E - refs[j]) : NaN for j in eachindex(sols)]
        ests = [Eprevs !== nothing && dEprevs !== nothing && j <= min(length(Eprevs), length(dEprevs)) ?
                tail_estimate(Float64(Eprevs[j] - sols[j].E), dEprevs[j]) : NaN for j in eachindex(sols)]
        bnd = [Float64(x.E) < Eth for x in sols]
        r = StageResult{T}(n=n, M=Mgen, Msolved=Mgen_sym, E=sol.E, error=err, solution=sol,
                           basis=copy(basis), t_build=t1 - t0, t_solve=t2 - t1, nexact=nexact,
                           states=sols, errors=errs, t_states=tst, estimates=ests, bound=bnd, order=order)
        iswarm || push!(results, r)
        if outdir !== nothing && !iswarm
            _append_result(outdir, r, length(es), c1 - c0, c2 - c1, loaded !== nothing)
            if save_wavefunction && sol.method !== :cholesky_failed
                for (k, sk) in enumerate(sols)
                    sk.method === :cholesky_failed && continue
                    x = D === nothing ? sk.c : D .* sk.c
                    _save_wavefunction(outdir, n, basis, es, C === nothing ? x : C * x, sk.E, order, L, unnatural; state=k)
                end
            end
        end
        # quadrature floor: kernels accepted at the step cap above tolerance (cumulative); nonzero means elements lost digits
        _capnote(e) = e isa KernelQuadrature && e.ncapped[] > 0 ? @sprintf("kq_capped=%d(+%.1fdig) ", e.ncapped[], e.capexcess[]) : ""
        _tail(j) = (j <= length(refs) && refs[j] !== nothing ? @sprintf(" err=%+.6e", errs[j]) : "") * (bnd[j] ? "" : " above_threshold")
        iswarm && @printf(io, "warm-up stage n=%d M=%d (starting point for excited states, not in the results) E=%s\n", n, r.Msolved, join([fmt(x.E, 20) for x in sols], " "))
        iswarm || @printf(io, "STAGE n=%d M=%d Msolved=%d exact_dep=%d E=%s est=%.3e%s method=%s residual=%.3e conv=%s nit=%d drop=%d smin=%.1e sets=%d build=%.1fs solve=%.1fs%s %s\n",
                n, r.M, r.Msolved, r.nexact, fmt(sol.E, ndigits), ests[1], _tail(1), sol.method, sol.residual, sol.converged,
                sol.iterations, sol.ndropped, sol.smin, length(es), r.t_build, r.t_solve, loaded === nothing ? "" : " (resumed)", _capnote(ctx.engine) * sol.note)
        for j in (iswarm ? (2:1) : (2:length(sols)))
            x = sols[j]
            @printf(io, "STATE n=%d k=%d E=%s est=%.3e%s method=%s residual=%.3e conv=%s nit=%d solve=%.1fs %s\n",
                    n, j, fmt(x.E, ndigits), ests[j], _tail(j), x.method, x.residual, x.converged, x.iterations, tst[j], x.note)
        end
        flush(io)
        # Hit the wall at this stage (or E is suspicious, or an excited state failed for lack of precision): re-solve from here.
        # With an explicitly non-Cholesky solver (e.g. CanonicalCut), method !== :cholesky is not a wall.
        exfail = canraise && n != _retry_stage && any(x -> x.method === :excited_failed, @view sols[2:end])
        exfail && println(io, "stage n=$n: excited states failed (", join([x.note for x in sols[2:end] if x.method === :excited_failed], "; "), "); raising the precision from this stage")
        hit = (chol && sol.method !== :cholesky) || suspicious || exfail
        rest = [m for m in stages if hit ? m >= n : m > n]
        wall = suspicious || exfail || (chol && _precision_wall(sol, sminprev, T, n, Msolve, drop_exact; warn=!auto_precision || isempty(rest), engine=ctx.engine))
        # stop: the stage failed and the precision cannot be raised, or E stays suspicious after raising it
        if (sol.method === :cholesky_failed && !canraise) || (suspicious && (n == _retry_stage || !canraise))
            iswarm || pop!(results)
            binhint = binary_warning(es0) === nothing ? "" : "; exponents are not on a binary grid (rounded Float64 sums across sets can break S), retry with `binary_exponents(es)`"
            println(io, "stop: stage n=$n ", suspicious ? (n == _retry_stage ?
                    "still drops unusually after raising the precision (not a precision problem; suspect the elements or amplification by near dependence" * binhint * ")" :
                    "drops unusually and " * (auto_precision ? "cannot be re-solved at the cap max_bits=$max_bits" : "auto_precision=false") *
                    " (suspected spurious eigenvalue" * binhint * ")") :
                    "failed and " * (!auto_precision ? "auto_precision=false (BigFloat $(curbits) bits)" :
                                     !drop_exact ? "drop_exact=false (BigFloat $(curbits) bits)" :
                                     "the precision cannot be raised beyond max_bits=$max_bits (BigFloat $(curbits) bits); raise max_bits or use fewer exponent sets") *
                    # with Float64x4 kernels the elements stop at ~1e-64, so raising only the working precision does not help
                    (ctx.engine isa KernelQuadrature && !(ctx.engine isa KernelQuadrature{BigFloat}) && isfinite(sminprev) && sminprev < 1e-55 ?
                     @sprintf("; the previous smallest eigenvalue of S, %.1e, is near the kernel precision (Float64x4, ~1e-64): use engine=KernelQuadrature(BigFloat)", sminprev) : ""))
            flush(io); return results
        end
        if auto_precision && wall && !isempty(rest) && canraise
            cur = curbits
            if hit
                λlast = NaN
                bits = 64 * ceil(Int, 2cur / 64)
            else
                ratio = isfinite(sminprev) && sminprev > sol.smin ? sol.smin / sminprev : 1e-11
                λlast = sol.smin * ratio^(maximum(rest) - n)
                digits = -log10(λlast) + log10(8 * Msolve) + 20
                bits = max(64 * ceil(Int, digits * log2(10) / 64), cur + 64)
            end
            if bits > max_bits
                println(io, "  estimated $bits bits exceed max_bits=$max_bits; using $max_bits",
                        hit ? "" : " (stage n=$(maximum(rest)) is not expected to be reached)")
                bits = max_bits
            end
            hit && !iswarm && pop!(results)
            println(io, "raising the precision: stages n=$(join(rest, ",")) in BigFloat $bits bits (about $(floor(Int, bits * log10(2))) digits) ",
                    "with BigFloat kernels (", hit ? "wall hit at this stage, doubling the bits)" : "extrapolated smallest eigenvalue of S $(@sprintf("%.1e", λlast)))")
            logvar.split && println(io, "  note: the angular grid of the split log variable is validated to 1e-63")
            flush(io)
            more = _rerun(rest, bits, hit ? n : 0)
            return StageResult[results; more]
        end
        sminprev = sol.smin
        # failed excited states (E = NaN) keep the previous E (an upper bound, used as the next shift guide)
        Enow = [isnan(x.E) && Eprevs !== nothing && j <= length(Eprevs) ? Eprevs[j] : x.E for (j, x) in enumerate(sols)]
        dEprevs = Eprevs === nothing || length(Eprevs) != length(Enow) ? nothing : Float64.(Eprevs .- Enow)
        Eprevs = Enow
        dEprev = dEprevs === nothing ? nothing : dEprevs[1]
        Eprev = sol.E
    end
    results
end

"The previous order prev (those present in b) first, then the rest of b in b's order."
function _prefix_order(prev::Vector{Key}, b::Vector{Key})
    isempty(prev) && return b
    inb = Set(b); front = Key[k for k in prev if k in inb]
    isempty(front) && return b
    seen = Set(front)
    [front; Key[k for k in b if !(k in seen)]]
end

# Drop the threads' last-task references (assembly tables, Cholesky storage), run a full GC and return freed heap to the
# OS (malloc_trim, Linux). Only on the root task (not when auto_exponents evaluates candidates in parallel). With `M`, small
# matrices skip it.
_release!() = current_task() === Base.roottask ? (release_task_refs(); GC.gc(); _trim_heap(); nothing) : nothing
_release!(M::Integer) = M >= _RELEASE_MIN ? _release!() : nothing
const _RELEASE_MIN = 2000
_trim_heap() = (Sys.islinux() && ccall(:malloc_trim, Cint, (Csize_t,), 0); nothing)

_bits(::Type{T}) where {T} = T === BigFloat ? precision(BigFloat) : round(Int, -log2(Float64(eps(T))))

# Magnitude range (log10) in which Float64-exponent types (Float64x4 etc.) keep all digits of the elements: below Float64's
# 1e308 with a margin for sums, and above 1e-244 so that the last limb (relative 1e-64) does not underflow.
const _MAG_LO, _MAG_HI = -220, 280
# (min, max) of log10 of the S and H diagonals of the functions likely to be extreme; (−Inf, Inf) if non-finite or S_ii = 0.
# Candidates per (exponent set) × (component: all, min, max): the functions of largest and smallest degree
# b[1]+b[2]+b[3]+b[4] (signed power of σ), ties broken by the power of ln σ.
function _magnitude_probe(hd, ctx, es, basis::Vector{Key}, ntasks::Int)
    fs = (b -> (b[1] + b[2] + b[3] + b[4], b[5]),)
    cmin, cmax = extrema(x -> x[7], basis)
    cand = Set{Int}()
    for k in unique(x[6] for x in basis), csel in (c -> true, ==(cmin), ==(cmax))
        ids = findall(b -> b[6] == k && csel(b[7]), basis)
        isempty(ids) && continue
        for f in fs
            v = f.(basis[ids])
            push!(cand, ids[argmax(v)], ids[argmin(v)])
        end
    end
    cand = sort!(collect(cand))
    # only diagonals are needed, so assemble one function at a time in parallel
    out = Vector{NTuple{3,Float64}}(undef, length(cand))
    Threads.@threads for c in eachindex(cand)
        H, S = assemble(hd, ctx, es, basis[[cand[c]]]; ntasks=1)
        h, s_ = H[1, 1], S[1, 1]
        out[c] = (isfinite(s_) && isfinite(h) && !iszero(s_)) ?
                 (log10(abs(Float64(s_))), iszero(h) ? NaN : log10(abs(Float64(h))), 0.0) : (NaN, NaN, 1.0)
    end
    any(o -> o[3] != 0, out) && return (-Inf, Inf)
    vals = filter(!isnan, [x for o in out for x in o[1:2]])
    minimum(vals), maximum(vals)
end

# Memory estimates: H, S and the Cholesky storage take about 1.5·M² elements, 2.2·M² with heap overhead. A BigFloat element is
# 56 + bits/8 bytes.
_elem_bytes(::Type{T}) where {T} = T === BigFloat ? 56 + precision(BigFloat) ÷ 8 : sizeof(T)
const _MEM_FACTOR = 2.2
_matrix_gb(::Type{T}, M::Integer) where {T} = _MEM_FACTOR * Float64(M)^2 * _elem_bytes(T) / 1e9
_default_mem_gb() = 0.5 * Sys.total_memory() / 1e9
"Peak of the assembly (GB): element table, integral memory, Hφ; c·M^1.5 bounds measurements for BigFloat (0 for other types)."
_build_gb(::Type{T}, M::Integer) where {T} = T === BigFloat ? _BUILD_C[] * Float64(M)^1.5 / 1e9 : 0.0
const _BUILD_C = Ref(10930.0)

# Precision wall. After removing exact dependencies S is positive definite in exact arithmetic, so a failing Cholesky or an
# unconverged residual means near dependence (λmin of S reached eps·M). λmin decreases by a nearly constant ratio per stage,
# so the next stage is extrapolated from the previous ratio and the wall is announced one stage ahead.
function _precision_wall(sol::Solution, sminprev::Float64, ::Type{T}, n::Int, M::Int, drop_exact::Bool;
                         warn::Bool=true, engine=nothing) where {T}
    drop_exact || return false
    hint = T === BigFloat ? "raise setprecision(BigFloat, …)" : "use T=BigFloat (with setprecision)"
    T === BigFloat && engine isa KernelQuadrature && !(engine isa KernelQuadrature{BigFloat}) && precision(BigFloat) > 212 &&
        (hint = "the kernel quadrature is still Float64x4 (elements to ~1e-64), so raising only the working precision does not help; use engine=KernelQuadrature(BigFloat) (automatic with auto_precision=true)")
    ε = Float64(eps(T))
    if sol.method !== :cholesky
        warn && @warn "stage n=$n: S is not tractable in the working precision even after removing exact dependencies ($(sol.method)); near dependence exhausts the precision. Without enough digits E is not even an upper bound, so check that E is stable when raising the precision. $hint" maxlog = 1
        return true
    end
    s = sol.smin
    isfinite(s) || return false
    next = isfinite(sminprev) && sminprev > s ? s * (s / sminprev) : s * 1e-6
    if s < 1e3 * M * ε || next < 10 * 2M * ε
        f(x) = @sprintf("%.1e", x)
        warn && @warn "stage n=$n: smallest eigenvalue of S ≈ $(f(s)) (previous $(f(sminprev))); the next stage's estimate $(f(next)) reaches eps·M ($(f(2M * ε))), so the working precision will not suffice. $hint" maxlog = 1
        return true
    end
    false
end

"""
    count_basis(sys, es; stages, T, rule, logvar, symmetry, io) -> Vector{NamedTuple}

Only generate the bases and return M per stage (and the dense H, S memory). No assembly or solve; for planning.
"""
function count_basis(sys::ThreeBody, es0::ExponentSets; stages=1:5, T::Type=Float64x4,
                     rule::FCRule=FCRule(), logvar::LogVariable=LogVariable(),
                     symmetry::Symmetry=NoSymmetry(), io::IO=stdout, L::Int=0, unnatural::Bool=false)
    sys, es0, _ = orient(sys, es0, symmetry; io=io)
    es = copy(es0)
    hd = _generation_hamiltonian(sys, logvar, HamiltonianData(sys, logvar, T; L=L, unnatural=unnatural), T)
    bytes = T === BigFloat ? 16 + 8 * cld(precision(BigFloat), 64) + 40 : sizeof(T)
    gen = seed_basis(rule, es, _seed_comps(symmetry, _maxcomp(hd)))
    out = NamedTuple[]
    for n in _stage_range(stages)
        gen, basis = _stage_basis(rule, hd, es, gen, n, symmetry)
        n in stages || continue
        M = length(basis)
        gb = 2 * M^2 * bytes / 1e9
        push!(out, (n=n, M=M, sets=length(es), HS_GB=gb))
        @printf(io, "n=%d M=%d sets=%d  H+S ≈ %.2f GB (%s)\n", n, M, length(es), gb, T)
    end
    out
end

"""
    optimize_alpha(sys, n; bracket, es, tol, io, kwargs...) -> (α, E, evals)

Minimize E of stage n over the exponent α (golden section in `bracket = (lo, hi)`, assumed unimodal). `es(α)` builds the
exponent sets (default: one set with α = β, γ = 0). Other keywords go to `run_fc`. Warns if the minimum is at an end.
"""
function optimize_alpha(sys::ThreeBody, n::Int; bracket::Tuple{Real,Real}, es=α -> ExponentSets([α], [α], [0.0]),
                        tol::Real=1e-4, io::IO=stdout, kwargs...)
    evals = Tuple{Float64,Any}[]
    function f(α)
        E = run_fc(sys, es(α); stages=n:n, io=devnull, kwargs...)[end].E
        push!(evals, (α, E))
        @printf(io, "  n=%d α=%.6f E=%s\n", n, α, fmt(E, 25)); flush(io)
        E
    end
    lo, hi = Float64.(bracket)
    g = (sqrt(5) - 1) / 2
    x1 = hi - g * (hi - lo); x2 = lo + g * (hi - lo)
    f1 = f(x1); f2 = f(x2)
    while hi - lo > tol
        if f1 < f2
            hi, x2, f2 = x2, x1, f1
            x1 = hi - g * (hi - lo); f1 = f(x1)
        else
            lo, x1, f1 = x1, x2, f2
            x2 = lo + g * (hi - lo); f2 = f(x2)
        end
    end
    αb, Eb = f1 < f2 ? (x1, f1) : (x2, f2)
    (αb - bracket[1] < 2tol || bracket[2] - αb < 2tol) &&
        @warn "minimum at the end of $(bracket) (α=$αb); widen the bracket"
    (α=αb, E=Eb, evals=evals)
end

"""
    optimize_exponents(sys, n, x0; make_es, feasible, step, tol, maxeval, io, kwargs...) -> (x, E, evals)

Minimize E of stage n over the exponent vector `x` (Nelder–Mead). `make_es(x)` builds the exponent sets, e.g. two mirrored
sets `x -> ExponentSets([x[1], x[2]], [x[2], x[1]], [0.0, 0.0])`. It may also return `(es, kw)` with extra `run_fc`
keywords, e.g. `x -> (ExponentSets([x[1]], [x[1]], [0.0]), (; logvar=LogVariable(x[2]; split=true)))`. Energies are
compared in the working precision. `step` is the initial simplex size and `tol` the convergence size (relative to x);
points with `feasible(x) == false` are rejected without evaluation. Other keywords go to `run_fc`.
"""
function optimize_exponents(sys::ThreeBody, n::Int, x0::AbstractVector{<:Real}; make_es, feasible=x -> true,
                            step::Real=0.1, tol::Real=1e-3, maxeval::Int=200, io::IO=stdout, kwargs...)
    evals = Tuple{Vector{Float64},Any}[]
    function f(x)
        (any(<=(0), x) || !feasible(x)) && return nothing
        m = make_es(x)
        es, kw = m isa ExponentSets ? (m, (;)) : m
        E = run_fc(sys, es; stages=n:n, io=devnull, kwargs..., kw...)[end].E
        push!(evals, (copy(x), E))
        @printf(io, "  n=%d x=%s E=%s\n", n, round.(x; digits=6), fmt(E, 25)); flush(io)
        E
    end
    better(a, b) = b === nothing || (a !== nothing && a < b)
    d = length(x0)
    pts = [Float64.(collect(x0))]
    for i in 1:d
        x = copy(pts[1]); x[i] *= 1 + step; push!(pts, x)
    end
    vals = Any[f(x) for x in pts]
    while length(evals) < maxeval
        ord = sortperm(1:d+1; lt=(i, j) -> better(vals[i], vals[j]))
        pts = pts[ord]; vals = vals[ord]
        maximum(maximum(abs.(p .- pts[1]) ./ abs.(pts[1])) for p in pts) < tol && break
        c = sum(pts[1:d]) / d
        xr = c .+ (c .- pts[end]); fr = f(xr)
        if better(fr, vals[1])
            xe = c .+ 2 .* (c .- pts[end]); fe = f(xe)
            pts[end], vals[end] = better(fe, fr) ? (xe, fe) : (xr, fr)
        elseif better(fr, vals[d])
            pts[end], vals[end] = xr, fr
        else
            xc = c .+ 0.5 .* (pts[end] .- c); fc = f(xc)
            if better(fc, vals[end])
                pts[end], vals[end] = xc, fc
            else
                for i in 2:d+1
                    pts[i] = pts[1] .+ 0.5 .* (pts[i] .- pts[1]); vals[i] = f(pts[i])
                end
            end
        end
    end
    ib = 1
    for k in 2:d+1
        better(vals[k], vals[ib]) && (ib = k)
    end
    (x=pts[ib], E=vals[ib], evals=evals)
end

"""
    energy_error(E, ref) -> Float64

`E − ref` subtracted in the working precision, then converted to Float64 (ref preferably a string).
"""
function energy_error(E, ref)
    setprecision(BigFloat, 512) do
        Float64(_tobig(E) - (ref isa AbstractString ? parse(BigFloat, ref) : _tobig(ref)))
    end
end

"""
    hermiticity(sys, es; logvar, T, basis) -> Float64

Asymmetry max|H − Hᵀ| / max|H| of the raw (unsymmetrized) H. About eps of the working precision if the action of H and the
integrals are consistent; the first check after adding a potential, log variable or exponent set. The default basis is a
small set with d ≠ 0 and logarithms (A, B, C ∈ 0:1, d ∈ −1:1, j ∈ 0:1, exponent set 1).
"""
function hermiticity(sys::ThreeBody, es0::ExponentSets; logvar::LogVariable=LogVariable(), T::Type=Float64x4,
                     basis::Vector{Key}=Key[(A, B, C, d, j, 1, 0) for A in 0:1 for B in 0:1 for C in 0:1 for d in -1:1
                                            for j in 0:1 if A + B + C + d >= 0])
    check_exponents(es0)
    hd = HamiltonianData(sys, logvar, T); ctx = IntegralContext(T, KernelQuadrature(), logvar)
    M = length(basis)
    H1 = assemble(hd, ctx, copy(es0), basis; rows=1:M-1)[1]       # a subset of rows is not symmetrized
    H2 = assemble(hd, ctx, copy(es0), basis; rows=M:M)[1]
    H = vcat(H1, H2)
    Float64(maximum(abs.(H - transpose(H))) / maximum(abs.(H)))
end

# ---------------------------------------------------------------------------
# Default mode: run with only the system
# ---------------------------------------------------------------------------

"Hydrogen-like decay κ = μ|c| from the attractive Coulomb term between particle i (1 or 2) and particle 3; nothing if absent."
function _hydrogenic_decay(sys::ThreeBody, i::Int)
    pow = i == 1 ? (-1, 0, 0) : (0, -1, 0)
    c = sum((_parse_rat(t.c) for t in sys.terms if t.pow == pow); init=0 // 1)
    c < 0 || return nothing
    m(s) = s == "Inf" ? Inf : Float64(_big(() -> exact(BigFloat, s)))
    mi, m3 = m(sys.masses[i]), m(sys.masses[3])
    μ = isinf(m3) ? mi : isinf(mi) ? m3 : mi * m3 / (mi + m3)
    isfinite(μ) || return nothing
    μ * Float64(-c)
end
_dyadic(x; bits=20) = round(x * 2.0^bits) / 2.0^bits     # non-binary exponents break S through rounded sums

"""
    default_exponents(sys) -> ExponentSets

Default exponent sets: hydrogen-like decays κᵢ = μᵢ₃ |qᵢq₃| from the attractive Coulomb terms with particle 3, and two sets
with 1 and 1/4 times κ₂ on the side of particle 2 (for a diffuse outer particle); κ = 1 without such a term.
"""
function default_exponents(sys::ThreeBody)
    κ = [something(_hydrogenic_decay(sys, i), 1.0) for i in 1:2]
    ExponentSets(fill(_dyadic(κ[1]), 2), [_dyadic(κ[2] * s) for s in (1.0, 0.25)], zeros(2))
end

"Exchange(+1) if there is an exchange-symmetric identical pair (reordered internally if needed), otherwise NoSymmetry()."
default_symmetry(sys::ThreeBody) = exchange_order(sys) === nothing ? NoSymmetry() : Exchange(+1)

"""
    run_fc(sys; kwargs...)

Default mode: exponent sets (`default_exponents`) and symmetry (`default_symmetry`) from the system, printed first. Other
keywords as in `run_fc(sys, es; ...)` (`stages` defaults to 1:3).
"""
function run_fc(sys::ThreeBody; symmetry=nothing, io::IO=stdout, kwargs...)
    sym = symmetry === nothing ? default_symmetry(sys) : symmetry
    sysi, _, π = orient(sys, nothing, sym; io=devnull)            # exponents in internal numbering, passed back in input numbering
    es = permute(default_exponents(sysi), _inverse(π))
    κ = [_hydrogenic_decay(sysi, i) for i in 1:2]
    println(io, "default mode: exponents α=", es.α, " β=", es.β, " γ=0 (from hydrogen-like decays κ=",
            [k === nothing ? "none→1" : round(k; sigdigits=6) for k in κ], "), symmetry ", sym isa Exchange ? "exchange(+)" : "none")
    r = run_fc(sys, es; symmetry=sym, io=io, merge((; auto_precision=true), kwargs)...)
    println(io, "next: more stages (stages=1:5), optimize exponents (auto_exponents, optimize_exponents)",
            _logvar_hint(sys, sym, get(kwargs, :logvar, LogVariable())))
    r
end

# The default log variable stays ln(r₁+r₂) (one-dimensional kernels for any exponents); ln(r₁+r₂+r₁₂) changes the function
# space, so it is only suggested for the systems where it helped.
function _logvar_hint(sys::ThreeBody, sym::Symmetry, lv::LogVariable)
    lv.split && return ""
    coul = all(_hydrogenic_decay(sys, i) !== nothing for i in 1:2)
    sym isa Exchange && coul ? "; for Coulomb systems with an identical pair, also try logvar=NewLog() (ln(r₁+r₂+r₁₂))" : ""
end

"Size estimate of the default mode (same exponent sets and symmetry as `run_fc(sys)`)."
function count_basis(sys::ThreeBody; symmetry=nothing, kwargs...)
    sym = symmetry === nothing ? default_symmetry(sys) : symmetry
    sysi, _, π = orient(sys, nothing, sym; io=devnull)
    count_basis(sys, permute(default_exponents(sysi), _inverse(π)); symmetry=sym, kwargs...)
end
