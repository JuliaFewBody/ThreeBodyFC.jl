# Output: per-stage result table, wave functions and checkpoints.
#
# outdir/
#   run.txt                 run conditions (output of describe)
#   results.tsv             one row per stage (E with 40 digits)
#   stage_NNN.wf.jls        basis, exponent sets, coefficients on the raw basis, E
#   stage_NNN.hs.jls        symmetrized H, S and the map C (for restarts, with checkpoint=true)

using Serialization

"CPU time of this process in seconds (from /proc on Linux; NaN otherwise)."
function cputime()
    try
        f = split(read("/proc/self/stat", String))
        (parse(Int, f[14]) + parse(Int, f[15])) / 100
    catch
        NaN
    end
end

const RESULT_COLUMNS = ["n", "M", "Msolved", "E", "error", "method", "residual", "converged", "iterations",
                        "ndropped", "sets", "t_build", "t_solve", "cpu_build", "cpu_solve", "resumed", "note", "nexact"]

function _write_header(outdir, text)
    mkpath(outdir)
    write(joinpath(outdir, "run.txt"), text)
    f = joinpath(outdir, "results.tsv")
    isfile(f) || write(f, join(RESULT_COLUMNS, '\t') * "\n")
end

function _append_result(outdir, r, nsets, cpu_build, cpu_solve, resumed)
    s = r.solution
    row = Any[r.n, r.M, r.Msolved, fmt(r.E, 40), r.error, s.method, s.residual, s.converged, s.iterations,
              s.ndropped, nsets, round(r.t_build; digits=1), round(r.t_solve; digits=1),
              round(cpu_build; digits=1), round(cpu_solve; digits=1), resumed, s.note, r.nexact]
    open(joinpath(outdir, "results.tsv"), "a") do io
        println(io, join(string.(row), '\t'))
    end
end

_stagefile(outdir, n, kind) = joinpath(outdir, @sprintf("stage_%03d.%s.jls", n, kind))

# `tag` describes the operator and the integrals (system, log variable, precision, kernel); restarts require it to match.
function _save_checkpoint(outdir, n, basis, es, H, S, C, tag::String="")
    f = _stagefile(outdir, n, "hs")
    serialize(f * ".tmp", (basis=basis, α=es.α, β=es.β, γ=es.γ, η=es.η, H=H, S=S, C=C, tag=tag))
    mv(f * ".tmp", f; force=true)
end

function _load_checkpoint(outdir, n, basis, es, tag::String="")
    f = _stagefile(outdir, n, "hs")
    isfile(f) || return nothing
    d = deserialize(f)
    (d.basis == basis && d.α == es.α && d.β == es.β && d.γ == es.γ && get(d, :η, zeros(length(es))) == es.η) || return nothing
    get(d, :tag, "") == tag || (@warn "checkpoint conditions differ; not used" f; return nothing)
    (d.H, d.S, get(d, :C, nothing))
end

function _save_wavefunction(outdir, n, basis, es, coef, E, order=(1, 2, 3), L=0, unnatural=false; state::Int=1)
    serialize(_stagefile(outdir, n, state == 1 ? "wf" : "wf$state"),
              (basis=basis, α=copy(es.α), β=copy(es.β), γ=copy(es.γ), η=copy(es.η), c=coef, E=E, Estr=fmt(E, 40), order=order, L=L, unnatural=unnatural))
end

"""
    load_wavefunction(outdir, n; state=1) -> NamedTuple

Wave function of stage n: basis keys, exponent sets, coefficients c on the raw basis, E, particle order `order` and total
angular momentum `L`. ψ = Σ_i c_i r₁^A r₂^B r₁₂^C σ^d (ln σ)^j exp(−α_k r₁ − β_k r₂ − γ_k r₁₂) in internal distances; internal
particle a is input particle `order[a]`. Stages re-solved at higher precision (`auto_precision`) are searched in
`outdir/bigfloat_<bits>/`. With `nstates > 1`, `state = k` reads the k-th state from the bottom.
"""
function load_wavefunction(outdir, n; state::Int=1)
    kind = state == 1 ? "wf" : "wf$state"
    f = _stagefile(outdir, n, kind)
    isfile(f) && return deserialize(f)
    # stages re-solved at higher precision are written under outdir/bigfloat_<bits>/ (possibly nested); prefer the most bits
    for sub in sort(filter(d -> startswith(d, "bigfloat_") && isdir(joinpath(outdir, d)), readdir(outdir));
                    by=d -> parse(Int, d[10:end]), rev=true)
        g = _stagefile(joinpath(outdir, sub), n, kind)
        isfile(g) && return deserialize(g)
        w = try load_wavefunction(joinpath(outdir, sub), n; state=state) catch; nothing end
        w === nothing || return w
    end
    throw(SystemError("no wave function for stage n=$n (state $state) in $outdir or its bigfloat_* subdirectories", 2))
end

"""
    load_results(outdir) -> Vector{NamedTuple}

Read `results.tsv`; E as BigFloat, n and M as integers.
"""
function load_results(outdir)
    lines = readlines(joinpath(outdir, "results.tsv"))
    hdr = split(lines[1], '\t')
    rows = NamedTuple[]
    # restarts repeat stages; keep the last row of each
    last = Dict{String,String}()
    order = String[]
    for l in lines[2:end]
        k = split(l, '\t')[1]
        haskey(last, k) || push!(order, k)
        last[k] = l
    end
    for l in (last[k] for k in order)
        v = split(l, '\t')
        d = Dict(Symbol(h) => x for (h, x) in zip(hdr, v))
        push!(rows, (n=parse(Int, d[:n]), M=parse(Int, d[:M]), E=setprecision(() -> parse(BigFloat, d[:E]), BigFloat, 256),
                     method=String(d[:method]), note=String(get(d, :note, ""))))
    end
    rows
end

"""
    compare(results, reference; io) -> Vector{NamedTuple}

Compare the energies per stage with `reference::Dict{Int,<:AbstractString}` (n => printed value). `results` is the
return value of `run_fc` or an output directory. Agreeing digits are counted after the decimal point.
"""
function compare(results, reference::AbstractDict; io::IO=stdout)
    rows = results isa AbstractString ? load_results(results) : [(n=r.n, M=r.M, E=_tobig(r.E)) for r in results]
    out = NamedTuple[]
    setprecision(BigFloat, 256) do
        for r in rows
            haskey(reference, r.n) || continue
            ref = reference[r.n]
            nd = length(split(ref, '.')[end])
            d = r.E - parse(BigFloat, ref)
            agree = iszero(round(d; digits=nd)) ? nd : max(0, floor(Int, -log10(abs(d)) - 0.5))
            push!(out, (n=r.n, M=r.M, E=fmt(r.E, nd), ref=ref, diff=Float64(d), digits=agree))
            @printf(io, "n=%d M=%d E=%s ref=%s diff=%+.3e agree=%d/%d\n", r.n, r.M, fmt(r.E, nd), ref, Float64(d), agree, nd)
        end
    end
    out
end
