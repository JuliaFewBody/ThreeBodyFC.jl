# Solvers: lowest eigenvalue of the generalized problem H c = E S c.
#
# Default: Cholesky projection with shifted inverse iteration (no eigendecomposition of S).
#   S = L Lᵀ, P = L⁻¹ H L⁻ᵀ (two triangular solves), Cholesky of (P − σI) and forward/backward substitution.
# With the energy of the previous stage, H − σS is factored directly instead. If S is not positive definite in the
# working precision or the iteration does not converge, the solver falls back to a pivoted subset or canonical
# orthogonalization.

using LinearAlgebra
using GenericLinearAlgebra

abstract type Solver end

"""
    CanonicalCut(; rel)

Eigendecompose S and keep the directions with λ > rel·λmax·M (`rel = nothing` means eps(T)) (canonical orthogonalization).
"""
Base.@kwdef struct CanonicalCut <: Solver
    rel::Union{Nothing,Float64} = nothing
end

"""
    PivotedSubset(; rel, inner)

Diagonally pivoted Cholesky that drops functions whose squared distance from the span of those already chosen (the
Schur complement diagonal) is below rel·max(diag S)·M, and solves the remaining subset with `inner`. A subset keeps the
variational upper bound. Used when exact or near dependencies make S indefinite in the working precision.
"""
Base.@kwdef struct PivotedSubset <: Solver
    rel::Union{Nothing,Float64} = nothing
    inner::Solver = CholeskyInverseIteration(fallback=CanonicalCut())
end

"""
    CholeskyInverseIteration(; maxit, restol, fallback, ozaki_min, dense_max, factor_S, etol_rel)

- `restol`: acceptance threshold of the residual ‖Hc − ESc‖/(‖Hc‖ + |E|‖Sc‖) (never stricter than √eps of the working
  precision); otherwise `fallback` is used.
- `ozaki_min`: for M ≥ ozaki_min, Cholesky and triangular solves use Ozaki-scheme dgemms (`chol_lower_ozaki!`);
  below it the persistent-worker `chol_lower_persist!`. `typemax(Int)` disables it.
- `dense_max`: largest M for the dense fallbacks (full eigendecomposition, pivoted Cholesky); larger problems fail instead.
- `factor_S = false`: with the previous stage's energy, skip the Cholesky of S (only used to check positive definiteness and
  to estimate its smallest eigenvalue `smin`, which then becomes NaN). E and the wave function are unchanged.
- `etol_rel = r > 0`: with the previous stage's energy, stop the inverse iteration once the change of E is below r·δ0
  (δ0 = decrease of E at the previous stage; `etol` of `lowest_eig_si`). The default 0 iterates the vector to eps.
"""
Base.@kwdef struct CholeskyInverseIteration <: Solver
    maxit::Int = 30
    restol::Float64 = 1e-30
    fallback::Union{Nothing,Solver} = nothing
    ozaki_min::Int = 512
    dense_max::Int = 2000
    factor_S::Bool = true
    etol_rel::Float64 = 0.0
end
"Default solver: Cholesky + inverse iteration, falling back to a pivoted subset, then canonical orthogonalization."
default_solver() = CholeskyInverseIteration(fallback=PivotedSubset())

describe(io::IO, s::CholeskyInverseIteration) =
    println(io, "solver=cholesky+inverse-iteration maxit=$(s.maxit) restol=$(s.restol) ozaki_min=$(s.ozaki_min) fallback=",
            s.fallback === nothing ? "none" : nameof(typeof(s.fallback)), s.factor_S ? "" : " factor_S=false",
            s.etol_rel > 0 ? " etol_rel=$(s.etol_rel)" : "")
describe(io::IO, s::PivotedSubset) = (print(io, "solver=pivoted-subset rel=", s.rel === nothing ? "eps" : s.rel, " inner: "); describe(io, s.inner))
describe(io::IO, s::CanonicalCut) = println(io, "solver=canonical-cut rel=", s.rel === nothing ? "eps" : s.rel)

"Result of a solve. Diagnostics the solver could not measure are NaN."
Base.@kwdef struct Solution{T}
    E::T
    c::Vector{T}
    method::Symbol
    residual::Float64
    converged::Bool
    iterations::Int = 0
    ndropped::Int = 0
    lambda_min::Float64 = NaN
    lambda_max::Float64 = NaN
    note::String = ""          # reason for a fallback
    smin::Float64 = NaN        # estimate of the smallest eigenvalue of S (inverse power iteration with its Cholesky factor)
end
_with_note(s::Solution{T}, note) where {T} =
    Solution{T}(E=s.E, c=s.c, method=s.method, residual=s.residual, converged=s.converged, iterations=s.iterations,
                ndropped=s.ndropped, lambda_min=s.lambda_min, lambda_max=s.lambda_max, note=note, smin=s.smin)

"Scale by the diagonal of S: S' = DSD, H' = DHD, D = diag(S)^{−1/2} (eigenvalues unchanged; packed storage in place)."
diagonal_normalize(H::SymPacked, S::SymPacked) = (D = diagonal_scale!(H, S); (H, S, D))
function diagonal_normalize(H::Matrix{T}, S::Matrix{T}) where {T}
    D = [inv(sqrt(S[i, i])) for i in axes(S, 1)]
    Hn = similar(H); Sn = similar(S)
    Threads.@threads for j in axes(S, 2)
        for i in axes(S, 1)
            Sn[i, j] = D[i] * S[i, j] * D[j]
            Hn[i, j] = D[i] * H[i, j] * D[j]
        end
    end
    (Hn, Sn, D)
end

function _residual(H, S, E, c)
    Hc = symv!(similar(c), H, c); Sc = symv!(similar(c), S, c)
    Float64(norm(Hc - E * Sc) / (norm(Hc) + abs(E) * norm(Sc)))
end

# The non-Cholesky solvers work on dense matrices.
solve(s::Union{PivotedSubset,CanonicalCut}, H::SymPacked, S::SymPacked; kw...) = solve(s, Matrix(H), Matrix(S); kw...)
_dense(A) = A
_dense(P::SymPacked) = Matrix(P)
_chol_fn(oz::Bool) = oz ? chol_lower_ozaki! : chol_lower_persist!

# `nofallback = true`: if the Cholesky path fails, return method = :cholesky_failed (E = NaN) without running the fallback
# (used by auto_precision, which re-solves at higher precision anyway).
# `reuse = (L, σ)`: factor of H − σS of the previous stage (its leading block equals the current one); the same σ is used,
# the old factor is copied and only the rest is factored.
# `keep = Ref{Any}()`: on success, receives (L=, σ=, reused=) of this stage for the next stage's `reuse`.
# Both apply only with Ozaki-scheme factorization (M ≥ ozaki_min) and a previous energy.
function solve(s::CholeskyInverseIteration, H::AbstractMatrix{T}, S::AbstractMatrix{T}; σ0=nothing, δ0=nothing,
               nofallback::Bool=false, reuse=nothing, keep=nothing) where {T}
    oz = size(H, 1) >= s.ozaki_min
    tol = max(s.restol, sqrt(Float64(eps(T))))             # reachable in the working precision
    r = _solve_chol(H, S, σ0, s.maxit, oz, δ0; factor_S=s.factor_S, etol_rel=s.etol_rel, restol=tol, reuse=reuse, keep=keep)
    if r isa Symbol
        why = String(r)
    else
        E, c, nit, conv, smin, mre = r
        res = _residual(H, S, E, c)
        # Without a previous energy, P = L⁻¹HL⁻ᵀ is formed, and with an ill-conditioned S its rounding limits the residual.
        # Re-solve once by factoring H − σS directly with σ just below the E found (σ = E − 2δ0, δ0 = 1000 × residual).
        if σ0 === nothing && conv && isfinite(Float64(E))
            r2 = _solve_chol(H, S, E, s.maxit, oz, 1e3 * res * max(abs(Float64(E)), 1.0); factor_S=s.factor_S, restol=tol)
            if !(r2 isa Symbol)
                E2, c2, nit2, conv2 = r2
                res2 = _residual(H, S, E2, c2)
                if conv2 && res2 < res
                    E, c, conv, res = E2, c2, conv2, res2
                    nit += nit2
                end
            end
        end
        conv && res < tol &&
            return Solution{T}(E=E, c=c, method=:cholesky, residual=res, converged=true, iterations=nit, smin=smin,
                               note=mre > 0 ? "factor_reused=$mre" : "")
        why = conv ? "residual=$(round(res; sigdigits=3))" : "not_converged(nit=$nit)"
    end
    keep === nothing || (keep[] = nothing)                 # a failed stage does not hand its factor on
    nofallback && return Solution{T}(E=T(NaN), c=T[], method=:cholesky_failed, residual=NaN, converged=false, note=why)
    size(H, 1) > s.dense_max &&
        return Solution{T}(E=T(NaN), c=T[], method=:cholesky_failed, residual=NaN, converged=false,
                           note=why * " no_dense_fallback(M>$(s.dense_max))")
    s.fallback === nothing && error("Cholesky path failed ($why) and there is no fallback")
    f = solve(s.fallback, _dense(H), _dense(S); σ0=σ0, δ0=δ0)
    _with_note(f, isempty(f.note) ? "cholesky_fallback:" * why : "cholesky_fallback:" * why * " " * f.note)
end

"""
    pivoted_select(S, τ) -> (indices, smallest kept distance², largest dropped distance²)

Left-looking diagonally pivoted Cholesky, stopped when the Schur complement diagonal (squared distance from the span of
the chosen functions) falls below τ. A dropped side near eps far below the kept side means exact dependence (the span is
unchanged); a continuous transition means near dependence (insufficient working precision).
"""
function pivoted_select(S::Matrix{T}, τ::T) where {T}
    M = size(S, 1)
    d = [S[i, i] for i in 1:M]
    piv = collect(1:M)
    cols = Vector{Vector{T}}()
    r = 0
    dkeep = T(Inf); ddrop = zero(T)
    for k in 1:M
        j = k - 1 + argmax(@view d[piv[k:M]])
        piv[k], piv[j] = piv[j], piv[k]
        p = piv[k]
        d[p] <= τ && (ddrop = d[p]; break)
        dkeep = min(dkeep, d[p])
        lp = sqrt(d[p])
        col = zeros(T, M)
        col[p] = lp
        rest = @view piv[k+1:M]
        Threads.@threads for ii in eachindex(rest)
            i = rest[ii]
            v = S[i, p]
            for l in 1:r; v -= cols[l][i] * cols[l][p]; end
            v /= lp
            col[i] = v
            d[i] -= v * v
        end
        push!(cols, col); r += 1
    end
    (sort(piv[1:r]), dkeep, ddrop)
end

function solve(s::PivotedSubset, H::Matrix{T}, S::Matrix{T}; σ0=nothing, δ0=nothing) where {T}
    M = size(S, 1)
    rel = s.rel === nothing ? eps(T) : T(s.rel)
    smax = maximum(S[i, i] for i in 1:M)
    keep, dkeep, ddrop = pivoted_select(S, rel * smax * M)
    sub = solve(s.inner, H[keep, keep], S[keep, keep]; σ0=σ0, δ0=δ0)
    gap = @sprintf("pivot:kept_min=%.1e,dropped_max=%.1e", Float64(dkeep / smax), Float64(ddrop / smax))
    c = zeros(T, M); c[keep] = sub.c
    Solution{T}(E=sub.E, c=c, method=Symbol("pivoted_", sub.method), residual=_residual(H, S, sub.E, c), smin=sub.smin,
                converged=sub.converged, iterations=sub.iterations, ndropped=M - length(keep) + sub.ndropped,
                lambda_min=sub.lambda_min, lambda_max=sub.lambda_max, note=isempty(sub.note) ? gap : gap * " " * sub.note)
end

solve(s::CanonicalCut, H::Matrix{T}, S::Matrix{T}; σ0=nothing, δ0=nothing) where {T} = canonical_roots(s, H, S, 1)[1]

"""
    canonical_roots(s::CanonicalCut, H, S, k) -> Vector{Solution}

The lowest k eigenpairs by canonical orthogonalization (full eigendecomposition of S; several times M³). Used for the first
stage of excited states and as a fallback.
"""
function canonical_roots(s::CanonicalCut, H::Matrix{T}, S::Matrix{T}, k::Int) where {T}
    ev = eigen_mt(Symmetric(S))
    λ = ev.values; V = ev.vectors
    rel = s.rel === nothing ? eps(T) : T(s.rel)
    keep = findall(>(rel * abs(λ[end]) * length(λ)), λ)
    X = V[:, keep] * Diagonal(inv.(sqrt.(λ[keep])))
    P = xtHx_mt(X, H)
    ep = eigen_mt(Symmetric((P + P') / 2))
    map(1:min(k, length(keep))) do j
        c = X * ep.vectors[:, j]; E = ep.values[j]
        Solution{T}(E=E, c=c, method=:canonical_cut, residual=_residual(H, S, E, c), converged=true,
                    ndropped=length(λ) - length(keep), lambda_min=Float64(λ[1]), lambda_max=Float64(λ[end]))
    end
end
canonical_roots(H, S, k) = canonical_roots(CanonicalCut(), _dense(H), _dense(S), k)

"""
    solve_excited(s, H, S, lower; σ0, δ0, work) -> Solution

The (length(lower)+1)-th state of the same matrices; `lower` are the converged lower states (S-normalized). The lower
states are lifted by c·(SX)(SX)ᵀ and H − σS + c·(SX)(SX)ᵀ is factored for inverse iteration (`lift` of `lowest_eig_si`).
σ is the previous stage's energy of this state lowered by twice its decrease, as for the ground state. A successful
factorization proves σ lies below the lowest state not lifted, so the result is the state with this index.
`work` is storage shaped like H, reused across states. If a lower state was not solved by the Cholesky path, or this
path fails, the dense eigendecomposition (`canonical_roots`) is used.
"""
function solve_excited(s::CholeskyInverseIteration, H::AbstractMatrix{T}, S::AbstractMatrix{T}, lower::Vector{Solution{T}};
                       σ0, δ0=nothing, work=nothing) where {T}
    k = length(lower) + 1
    why = "lower_not_cholesky"
    if isfinite(σ0) && all(l -> l.method === :cholesky || l.method === :cholesky_lift, lower)
        chol! = _chol_fn(size(H, 1) >= s.ozaki_min)
        X = reduce(hcat, [l.c for l in lower])
        E1 = lower[1].E
        work === nothing && (work = _work_like(S))
        δ = δ0 === nothing || !(δ0 > 0) ? 1e-6 : 2 * Float64(δ0)
        why = "shift_not_below_E$k"
        for _ in 1:6
            σ = T(σ0) - T(δ)
            # Put the lifted eigenvalues E_m + c above the target E_k (≤ σ0); c > σ0 − E₁ lifts all of them above σ0.
            # (Choosing c from the distance to σ lets a lifted ground state come closer to σ than E_k when σ is lowered.)
            c = 3 * abs(T(σ0) - E1) + 3 * abs(T(σ0) - σ) + abs(σ) / 1000
            r = lowest_eig_si(H, S, σ; chol! = chol!, maxit=s.maxit, rounds=_REFINE_ROUNDS, work=work, lift=(X, c))
            if r !== nothing
                E, x, nit, conv = r
                res = _residual(H, S, E, x)
                tol = max(s.restol, sqrt(Float64(eps(T))))
                ov = maximum(abs, X' * symv!(similar(x), S, x))     # overlap with the lower states (rounding level if lifted)
                if conv && res < tol && ov < sqrt(Float64(eps(T)))
                    return Solution{T}(E=E, c=x, method=:cholesky_lift, residual=res, converged=true, iterations=nit)
                end
                why = !conv ? "not_converged(nit=$nit)" : ov >= sqrt(Float64(eps(T))) ? "overlap_with_lower=$(round(Float64(ov); sigdigits=3))" :
                      "residual=$(round(res; sigdigits=3))"
                break
            end
            δ *= 100
        end
    end
    if size(H, 1) > s.dense_max         # report failure (E = NaN); the driver keeps using the previous E as the shift
        return Solution{T}(E=T(NaN), c=zeros(T, size(H, 1)), method=:excited_failed, residual=NaN, converged=false,
                           note="excited_failed:" * why)
    end
    f = canonical_roots(H, S, k)
    length(f) < k && error("excited state $k: only $(length(f)) dimensions remain after canonical orthogonalization")
    _with_note(f[k], "excited_fallback:" * why)
end
solve_excited(s::Solver, H, S, lower; kw...) = _with_note(canonical_roots(H, S, length(lower) + 1)[end], "excited:canonical")

"Scratch storage shaped like S (distinct elements for dense BigFloat)."
_work_like(S::SymPacked) = SymPacked(size(S, 1); precision=S.prec)
_work_like(S::AbstractMatrix) = fresh(S)

const _REFINE_ROUNDS = 6

"Cholesky factor of P − σI, or nothing if not positive definite."
function _shifted_chol(P::Matrix{T}, σ, chol!) where {T}
    A = fresh(P)
    @inbounds for i in axes(A, 1); A[i, i] -= σ; end
    chol!(A) ? A : nothing
end

"""
Smallest eigenvalue of S = LLᵀ by inverse power iteration (approaches from above, so slow convergence overestimates).
The smallest Cholesky pivot is not a substitute for λmin.
"""
function _smin_estimate(L::AbstractMatrix{T}; maxit::Int=30, rtol::Float64=1e-2) where {T}
    M = size(L, 1)
    x = T[one(T) + T(k % 7) / 8 for k in 1:M]; x ./= norm(x)
    est = Inf
    for _ in 1:maxit
        y = llt_solve!(fresh(x), L)                  # (L Lᵀ)⁻¹ x
        μ = norm(y)
        new = Float64(inv(μ))
        x = y ./ μ
        abs(new - est) <= rtol * new && return new
        est = new
    end
    est
end

function _solve_chol(H::AbstractMatrix{T}, S::AbstractMatrix{T}, σ0, maxit, oz::Bool=false, δ0=nothing;
                     factor_S::Bool=true, etol_rel::Float64=0.0, restol::Float64=1e-30, reuse=nothing, keep=nothing) where {T}
    chol! = _chol_fn(oz)
    trsm! = oz ? trsm_lower_ozaki! : ltri_solve!
    if !factor_S && σ0 !== nothing
        L = _work_like(S); smin = NaN               # only storage for H − σS (overwritten by lowest_eig_si)
    else
        L = fresh(S)
        chol!(L) || return :S_not_positive_definite
        tril!(L)
        smin = _smin_estimate(L)
    end
    if σ0 !== nothing
        # With the previous energy (an upper bound of E₁), factor H − σS directly (no P = L⁻¹HL⁻ᵀ, saving two M³ solves).
        # σ = σ0 − 2δ0 with δ0 the previous decrease of E; the decreases shrink, so σ lands just below E₁. If H − σS is
        # not positive definite (σ above E₁), σ is lowered. H − σS is built in the storage of S's factor.
        δ = δ0 === nothing || !(δ0 > 0) ? 1e-6 : 2 * Float64(δ0)
        etol = etol_rel > 0 && δ0 !== nothing && δ0 > 0 ? T(etol_rel * Float64(δ0)) : nothing
        canfac = oz && L isa SymPacked                     # factors can be reused across stages
        canfac || (reuse = nothing; keep = nothing)
        # a factor whose σ was moved up during the rounds is not handed on (it may lie above the next stage's E)
        _keep!(r, m, σin) = keep === nothing || r[5] != σin || (keep[] = (L=L, σ=r[5], reused=m))
        # Reuse of the previous factor: same σ, copy the old factor into the leading block and factor only the new
        # columns. Only the first factorization is a restart; if it fails (σ above the new E₁) the usual σ is used.
        if reuse !== nothing && reuse.L isa SymPacked && reuse.L.prec == L.prec && reuse.L.M < size(L, 1)
            m = reuse.L.M; first = Ref(true)
            chr!(A) = first[] ? (first[] = false; copy_leading!(A, reuse.L); chol!(A; done=m)) : chol!(A)
            r = lowest_eig_si(H, S, reuse.σ; chol! = chr!, maxit=maxit, rounds=_REFINE_ROUNDS, work=L, etol=etol, etol_dx=1e-3 * restol)
            σr = reuse.σ; reuse = nothing
            r === nothing || (_keep!(r, m, σr); return (r[1], r[2], r[3], r[4], smin, m))
        end
        for _ in 1:6
            r = lowest_eig_si(H, S, T(σ0) - T(δ); chol! = chol!, maxit=maxit, rounds=_REFINE_ROUNDS, work=L, etol=etol, etol_dx=1e-3 * restol)
            r === nothing || (_keep!(r, 0, T(σ0) - T(δ)); return (r[1], r[2], r[3], r[4], smin, 0))
            δ *= 100
        end
        return :shift_not_below_E1
    end
    Y = H isa SymPacked ? Matrix(H) : fresh(H); trsm!(Y, L)   # L⁻¹H (dense)
    P = permutedims(Y); trsm!(P, L)                            # L⁻¹HL⁻ᵀ (H symmetric)
    M = size(P, 1)
    P64 = Float64.(P); P64 .= (P64 .+ P64') ./ 2
    e = eigen(Symmetric(P64), 1:1)
    σc, v0, δ = e.values[1], e.vectors[:, 1], max(1e-9, maximum(abs, P64) * 1e-12)
    A = nothing; σ = T(σc) - T(δ)
    chold! = _chol_fn(oz)
    for _ in 1:6
        A = _shifted_chol(P, σ, chold!)
        A === nothing || break
        δ *= 100; σ = T(σc) - T(δ)
    end
    A === nothing && return :shift_not_below_E1
    v = T[T(x) for x in v0]
    v ./= norm(v)
    Pv = similar(v); E = T(Inf); Eprev = T(Inf); conv = false; nit = 0
    # The rate is (E₁−σ)/(E₂−σ); for small gaps (weakly bound states near threshold) the first shift is slow, so if not
    # converged σ ← E − (E−σ)/10 and refactor. Stop when the change of v is ≤ 10 eps or stops shrinking.
    for round in 1:_REFINE_ROUNDS
        dxprev = Inf
        for _ in 1:maxit
            vold = v
            v = llt_solve!(fresh(v), A); v ./= norm(v)
            symv!(Pv, P, v); E = dot(v, Pv); nit += 1
            dx = Float64(maximum(abs, v - vold) / maximum(abs, v))
            if nit >= 2 && abs(E - Eprev) <= sqrt(eps(T)) * abs(E) && (dx <= 10 * Float64(eps(T)) || dx >= dxprev)
                conv = true; break
            end
            Eprev = E; dxprev = dx
        end
        (conv || round == _REFINE_ROUNDS) && break
        gap = E - σ
        Anew = nothing
        for f in (10.0, 3.0, 1.5)
            Anew = _shifted_chol(P, E - gap / f, chold!)
            Anew === nothing || (σ = E - gap / f; break)
        end
        Anew === nothing && break
        A = Anew
    end
    c = lt_solve!(fresh(v), L)                    # c = L⁻ᵀ v
    (E, c, nit, conv, smin, 0)
end
