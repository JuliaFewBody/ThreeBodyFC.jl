# Symmetry: exchange of particles 1 ↔ 2.
#
# P φ(A,B,C,d,j; α,β,γ) = φ(B,A,C,d,j; β,α,γ); σ is invariant under P for any βσ.
# Each representative gives φ ± Pφ, and H' = CᵀHC, S' = CᵀSC.

abstract type Symmetry end
"No symmetrization."
struct NoSymmetry <: Symmetry end
"""
    Exchange(sign)

`sign = +1` (symmetric) or `−1` (antisymmetric). A function whose mirror key is not in the basis is kept alone if
symmetric and dropped if antisymmetric.
"""
struct Exchange <: Symmetry
    sign::Int
    Exchange(s::Integer) = abs(s) == 1 ? new(Int(s)) : throw(ArgumentError("sign must be ±1"))
end
describe(io::IO, ::NoSymmetry) = println(io, "symmetry=none")
describe(io::IO, s::Exchange) = println(io, "symmetry=exchange(", s.sign > 0 ? "+" : "-", ")")

"""
    check_symmetry(sym, sys)

Check that H is invariant under 1 ↔ 2 (m₁ = m₂ and the potential terms close under exchange, compared by value).
`run_fc` reorders systems whose identical pair is elsewhere first (`orient`).
"""
check_symmetry(::NoSymmetry, sys) = nothing
function check_symmetry(::Exchange, sys::ThreeBody)
    _rat(sys.masses[1]) == _rat(sys.masses[2]) || throw(ArgumentError("Exchange: m₁ ≠ m₂ ($(sys.masses[1]) / $(sys.masses[2]))"))
    _exchange_invariant(sys) || throw(ArgumentError("Exchange: the potential is not symmetric under 1 ↔ 2"))
    nothing
end

# P swaps r₁ ↔ r₂ and maps the component 𝒴_a = u₁^a u₂^{L−a} to 𝒴_{L−a} (no sign); L is the largest component index.
# Unnatural parity maps 𝒴_a → −𝒴_{L−1−a}: callers pass L − 1 and the sign-flipped `_component_symmetry`.
_mirror_key(t::Key, es::ExponentSets, L::Int=0) = (t[2], t[1], t[3], t[4], t[5], find_set(es, es.β[t[6]], es.α[t[6]], es.γ[t[6]], es.η[t[6]]), L - t[7])

"""
    symmetrize_basis(sym, basis, es) -> basis

With `Exchange`, close the basis under exchange (add mirror exponent sets and keys); `NoSymmetry` does nothing.
"""
symmetrize_basis(::NoSymmetry, basis, es, L::Int=0) = basis
function symmetrize_basis(::Exchange, basis::Vector{Key}, es::ExponentSets, L::Int=0)
    s = Set(basis)
    for t in basis
        k = t[6]
        kp = find_set(es, es.β[k], es.α[k], es.γ[k], es.η[k])
        if kp == 0
            push!(es.α, es.β[k]); push!(es.β, es.α[k]); push!(es.γ, es.γ[k]); push!(es.η, es.η[k]); kp = length(es)
        end
        push!(s, (t[2], t[1], t[3], t[4], t[5], kp, L - t[7]))
    end
    sort!(collect(s))
end

# Symmetry acting on component mirrors (sign flipped for unnatural parity, `_xfactor`).
_component_symmetry(s::NoSymmetry, hd) = s
_component_symmetry(s::Exchange, hd) = Exchange(s.sign * _xfactor(hd))

"Dimension after symmetrization (antisymmetric keys that are their own mirror vanish and are not counted)."
_symdim(::NoSymmetry, basis, es, L::Int=0) = length(basis)
function _symdim(sym::Exchange, basis::Vector{Key}, es::ExponentSets, L::Int=0)
    idx = Dict(b => i for (i, b) in enumerate(basis))
    count(i -> (m = idx[_mirror_key(basis[i], es, L)]; i <= m && !(sym.sign < 0 && m == i)), eachindex(basis))
end

"""
    assemble_projected(sym, hd, ctx, es, basis; ntasks) -> (H', S', C)

Matrices in the symmetrized basis Φ_a = φ_i + s·Pφ_i (representatives i ≤ Pi). Since P H P = H,
⟨Φ_a|H|Φ_b⟩ = 2(H_ij + s·H_{i,Pj}), so only the representative rows (about M/2) times all columns are needed (the
factor 2 is dropped). `C` maps the raw basis to the symmetrized one (Φ = columns of C).
"""
function assemble_projected(::NoSymmetry, hd, ctx, es, basis; ntasks::Int=Threads.nthreads(), kw...)
    H, S = assemble(hd, ctx, es, basis; ntasks=ntasks, kw...)
    (H, S, nothing)
end

function assemble_projected(sym::Exchange, hd::HamiltonianData{T}, ctx, es::ExponentSets, basis::Vector{Key};
                            ntasks::Int=Threads.nthreads(), packed::Bool=false, progress::Real=0, io::IO=stdout,
                            dscale=nothing, kw...) where {T}
    idx = Dict(b => i for (i, b) in enumerate(basis))
    sym = _component_symmetry(sym, hd)                  # unnatural parity flips the sign of component mirrors
    mirror = [get(idx, _mirror_key(t, es, _maxcomp(hd)), 0) for t in basis]
    any(==(0), mirror) && error("basis is not closed under exchange (call symmetrize_basis first)")
    reps = [i for i in eachindex(basis) if i <= mirror[i] && !(sym.sign < 0 && mirror[i] == i)]
    C = ExchangeMap(length(basis), reps, mirror, sym.sign)
    if packed && T === BigFloat
        tab = _filled_table(hd, ctx, es, basis; rows=reps, ntasks=ntasks, kw...)
        H, S = _fill_packed_exchange(tab, reps, mirror, sym.sign, ntasks; progress=progress, io=io, dscale=dscale)
        return (H, S, C)
    end
    H, S = assemble(hd, ctx, es, basis; rows=reps, ntasks=ntasks, progress=progress, io=io, kw...)
    Mx = length(reps); s = T(sym.sign)
    Hx = Matrix{T}(undef, Mx, Mx); Sx = Matrix{T}(undef, Mx, Mx)
    Threads.@threads for b in 1:Mx
        j = reps[b]; pj = mirror[j]
        for a in 1:Mx
            Hx[a, b] = H[a, j] + s * H[a, pj]
            Sx[a, b] = S[a, j] + s * S[a, pj]
        end
    end
    Hx = (Hx + Hx') / 2; Sx = (Sx + Sx') / 2
    (Hx, Sx, C)
end

"""
    ExchangeMap

The map C from the raw basis to the symmetrized one (Φ_a = φ_{r_a} + s·φ_{P r_a}, two entries per column). `C * x` gives raw coefficients.
"""
struct ExchangeMap
    M::Int
    reps::Vector{Int}       # representatives
    mirror::Vector{Int}     # index of the mirror
    sign::Int
end
Base.size(C::ExchangeMap) = (C.M, length(C.reps))
function Base.:*(C::ExchangeMap, x::AbstractVector{T}) where {T}
    length(x) == length(C.reps) || throw(DimensionMismatch("C is $(size(C)), x has length $(length(x))"))
    y = zeros(T, C.M)
    for (a, i) in enumerate(C.reps)
        y[i] += x[a]; y[C.mirror[i]] += C.sign * x[a]
    end
    y
end
Base.Matrix(C::ExchangeMap) = (A = zeros(Int, size(C)); for (a, i) in enumerate(C.reps); A[i, a] += 1; A[C.mirror[i], a] += C.sign; end; A)
