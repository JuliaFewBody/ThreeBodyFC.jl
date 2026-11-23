# Integrals: element integrals
#
#   I(A,B,C,D,J; a,b,c) = ∫ r₁^A r₂^B r₁₂^C σ^D (ln σ)^J e^{−a r₁ − b r₂ − c r₁₂} dr₁ dr₂ dr₁₂
#
# The volume element r₁r₂r₁₂ is included by the caller (A,B,C + 1). The constant 8π² is dropped (cancels in Rayleigh quotients).
#
# In perimetric coordinates r₁ = v+w, r₂ = u+w, r₁₂ = u+v (Jacobian 2) the exponential separates into
# −ρu u − ρv v − ρw w (ρu = b+c, ρv = a+c, ρw = a+b), and σ = λ(c_u u + v + c_w w) with
# λ = 1+βσ, c_u = (w₂+βσ)/λ, c_w = (1+w₂)/λ (σ = r₁ + w₂ r₂ + βσ r₁₂; c_u = 1, c_w = 2/λ for w₂ = 1). Negative powers and logarithms of σ use the Laplace form
#     σ̃^{−R} = Γ(R)^{−1} ∫ t^{R−1} e^{−tσ̃} dt
# which reduces them to one-dimensional kernels
#     K_j(κ; Nu,Nv,Nw) = ∫₀^∞ t^κ (ln t)^j (ρu+c_u t)^{−Nu} (ρv+t)^{−Nv} (ρw+c_w t)^{−Nw} dt
# The kernel method is a pluggable engine (`KernelQuadrature`, …).
#
# Lowest-level submodule, independent of the FC method (generation, systems, symmetry). The log variable comes as values (βσ, split).
# Choices that do not change results (closed radial forms for a = b, two-dimensional quadrature for a ≠ b, …) are made here from the exponents.

module Integrals

using MultiFloats
using Printf
using HighPrecisionLinearAlgebra: mpfr_powi!, mpfr_mul!, mpfr_add!, mpfr_sub!, mpfr_div!, mpfr_mul_si!, mpfr_mul_2si!, mpfr_set!, mpfr_ptr, PackedMatrix
using HighPrecisionLinearAlgebra: _el, free!

export KernelEngine, KernelQuadrature, kernels, IntegralContext, ElementCache, element!, element_split!, elements_upto!,
       prefetch_kernels!, describe, GaussTable, gauss_table, element_gauss, gauss_range

"An extended-precision value as BigFloat (Float64x4 exactly, as the sum of its limbs)."
_tobig(x) = hasproperty(x, :_limbs) ? sum(BigFloat.(x._limbs)) : BigFloat(x)

# --- Constants at integer arguments (computed in BigFloat with 64 bits more than the working precision, at least 512, then rounded to T)
# Their precision must follow the working precision: fixed 512-bit constants produced spurious eigenvalues once the smallest
# eigenvalue of S fell below their error.
_bigprec() = max(512, precision(BigFloat) + 64)
_big(f) = setprecision(f, BigFloat, _bigprec())
_euler() = BigFloat(Base.MathConstants.eulergamma)     # MPFR computes it at the current precision
gamma_inv(::Type{T}, R::Int) where {T} = T(_big(() -> inv(BigFloat(factorial(BigInt(R - 1))))))
digamma_int(::Type{T}, R::Int) where {T} = T(_big(() -> sum((inv(BigFloat(i)) for i in 1:R-1); init=BigFloat(0)) - _euler()))
trigamma_int(::Type{T}, R::Int) where {T} = T(_big(() -> BigFloat(π)^2 / 6 - sum((inv(BigFloat(i))^2 for i in 1:R-1); init=BigFloat(0))))
euler_gamma(::Type{T}) where {T} = T(_big(_euler))
pi2over6(::Type{T}) where {T} = T(_big(() -> BigFloat(π)^2 / 6))
factT(::Type{T}, n::Int) where {T} = T(factorial(BigInt(n)))

# =============================================================================
# Kernel engines
# =============================================================================
abstract type KernelEngine end

"""
    KernelQuadrature(Q; h, slo, shi, cut)

Kernels by the trapezoidal rule in s = ln t. The integrand is positive (no cancellation), so the precision of `Q` is reached.
`Q = Float64x4` (default): h = 1/12 (discretization error e^{−2π²/h} ≈ 1e-103), s ∈ [−190, 200], cut = 170.
`Q = BigFloat` with b = `precision(BigFloat)` bits at construction: h = 2π²/(b ln 2 + 100), cut = b ln 2 + 20,
s ∈ ±(b ln 2 + 35) (truncation below 2^{−b} e^{−20}). The margin 100 in h covers the prefactor of the discretization error
(about 2^{m−1} h^{−(m−1)} for a pole of order m). The discretization error is estimated per kernel and the step refined if
needed (`kernels` halves it; the assembly prefetch refines by quarter octaves). `cut` drops nodes below e^{−cut} of the peak;
the peak and the kept range are found with Float64 shadows and only that range is computed in `Q` (the log of the integrand
is concave in s, so the kept nodes form one interval). One table per (ρu, ρv, ρw, c_w) is shared.
"""
struct KernelQuadrature{Q} <: KernelEngine
    h::Float64
    slo::Float64
    shi::Float64
    cut::Float64
    L::Float64          # truncation/range reference e^{−L} (BigFloat: 2^{−b} e^{−20})
    logtol::Float64     # acceptance threshold of the discretization error estimate (log; eps/10 of the working precision)
    maxlevel::Int       # refine the step down to h/2^ℓ (table level q in quarter octaves: step h/2^{q/4}, q ≤ 4 maxlevel)
    tables::Dict{Tuple{Float64,Float64,Float64,Float64,Float64,Int},Any}       # (ρu, ρv, ρw, c_w, c_u, level)
    building::Dict{Tuple{Float64,Float64,Float64,Float64,Float64,Int},Threads.Event}   # tables being built (by one task only)
    lock::ReentrantLock
    nrefined::Threads.Atomic{Int}   # kernels whose step was refined (diagnostic)
    ncapped::Threads.Atomic{Int}    # kernels accepted at maxlevel with the estimate above threshold (quadrature floor, diagnostic)
    capexcess::Threads.Atomic{Float64}   # largest excess in digits (log10) among those
end
function KernelQuadrature(::Type{Q}=Float64x4; h=nothing, slo=nothing, shi=nothing, cut=nothing, maxlevel::Int=6) where {Q}
    lne = Q === BigFloat ? precision(BigFloat) * log(2) : -log(Float64(eps(Q)))   # −ln eps
    L = lne + 20
    hx = h === nothing ? (Q === BigFloat ? 2π^2 / (lne + 100) : Q === Float64x4 ? 1 / 12 : 1 / 16) : Float64(h)
    cx = cut === nothing ? (Q === BigFloat ? L : Q === Float64x4 ? 170.0 : 300.0) : Float64(cut)
    lo = slo === nothing ? (Q === BigFloat ? -(L + 15) : -190.0) : Float64(slo)
    hi = shi === nothing ? (Q === BigFloat ? L + 15 : 200.0) : Float64(shi)
    KernelQuadrature{Q}(hx, lo, hi, cx, L, -lne - log(10), maxlevel, Dict{Tuple{Float64,Float64,Float64,Float64,Float64,Int},Any}(),
                        Dict{Tuple{Float64,Float64,Float64,Float64,Float64,Int},Threads.Event}(), ReentrantLock(),
                        Threads.Atomic{Int}(0), Threads.Atomic{Int}(0), Threads.Atomic{Float64}(0.0))
end
describe(io::IO, e::KernelQuadrature{Q}) where {Q} =
    @printf(io, "kernel=quadrature Q=%s h=%.6g s=[%.1f,%.1f] cut=%.1f tol=e^%.0f maxlevel=%d\n",
            Q, e.h, e.slo, e.shi, e.cut, e.logtol, e.maxlevel)

struct _KQTable{Q}
    h::Float64      # step (s_i = slo + (i−1)h exactly in Q)
    s::Vector{Q}; lnu::Vector{Q}; lnv::Vector{Q}; lnw::Vector{Q}
    s64::Vector{Float64}; lnu64::Vector{Float64}; lnv64::Vector{Float64}; lnw64::Vector{Float64}
    # BigFloat only: t and 1/(ρu+t), 1/(ρv+t), 1/(ρw+c_w t) (the integrand is built from products of powers; exp in MPFR allocates and is 54× slower)
    t::Vector{Q}; iu::Vector{Q}; iv::Vector{Q}; iw::Vector{Q}
end

# One table per (ρu, ρv, ρw, c_w, step level). Other tasks wanting the same table wait for it to be built (building it
# simultaneously in all tasks piles up allocations and stalls in GC).
function _kq_table(e::KernelQuadrature{Q}, ρu, ρv, ρw, cw, level::Int=0, cu=1) where {Q}   # level in quarter octaves
    key = (Float64(ρu), Float64(ρv), Float64(ρw), Float64(cw), Float64(cu), level)
    tb, ev = lock(e.lock) do
        t = get(e.tables, key, nothing)
        t === nothing || return (t, nothing)
        ev = get(e.building, key, nothing)
        ev === nothing || return (nothing, ev)
        e.building[key] = Threads.Event()
        (nothing, nothing)
    end
    tb === nothing || return tb::_KQTable{Q}
    if ev !== nothing
        wait(ev)
        return _kq_table(e, ρu, ρv, ρw, cw, level, cu)
    end
    try
        tb = _kq_build(e, Q(ρu), Q(ρv), Q(ρw), Q(cw), _kq_h(e, level), Q(cu))
        lock(() -> (e.tables[key] = tb), e.lock)
    finally
        ev = lock(() -> pop!(e.building, key), e.lock)
        notify(ev)
    end
    tb::_KQTable{Q}
end

# node values are built by all threads (thousands of BigFloat exp, log and inverses per table)
function _kq_build(e::KernelQuadrature{Q}, ρu::Q, ρv::Q, ρw::Q, cw::Q, h::Float64, cu::Q=one(Q)) where {Q}
    u1 = isone(cu)
    n = round(Int, (e.shi - e.slo) / h) + 1
    s, t, lnu, lnv, lnw = (Vector{Q}(undef, n) for _ in 1:5)
    big = Q === BigFloat
    iu, iv, iw = big ? (Vector{Q}(undef, n), Vector{Q}(undef, n), Vector{Q}(undef, n)) : (Q[], Q[], Q[])
    chunks = Iterators.partition(1:n, cld(n, 4Threads.nthreads()))
    @sync for c in chunks
        Threads.@spawn for i in c
            si = Q(e.slo) + (i - 1) * Q(h); ti = exp(si)
            u, v, w = u1 ? ρu + ti : ρu + cu * ti, ρv + ti, ρw + cw * ti
            s[i] = si; t[i] = ti; lnu[i] = log(u); lnv[i] = log(v); lnw[i] = log(w)
            big && (iu[i] = inv(u); iv[i] = inv(v); iw[i] = inv(w))
        end
    end
    _KQTable{Q}(h, s, lnu, lnv, lnw, Float64.(s), Float64.(lnu), Float64.(lnv), Float64.(lnw),
                big ? t : Q[], iu, iv, iw)
end

# peak Emax (Float64), the interval lo:hi of nodes within e^{−cut} of it, and the peak node imax.
@inline _kq_E(tb::_KQTable, i, k1, Nu, Nv, Nw) = @inbounds k1 * tb.s64[i] - Nu * tb.lnu64[i] - Nv * tb.lnv64[i] - Nw * tb.lnw64[i]
# imax (the first maximum of a full forward scan) is found by bisection and a scan of a window around the peak.
# E(s) = k₁ s − Σ N ln(ρ + c eˢ) is strictly concave in s (each term has second derivative −N ρ c eˢ/(ρ + c eˢ)² < 0). With δ the
# difference between computed and exact values, scan outward from the bisection point p to the first node L (R on the right)
# with E < E(p) − τ. By concavity Ẽ(i) ≤ Ẽ(L) for i beyond L (else Ẽ(L) ≥ min(Ẽ(i), Ẽ(p)) = Ẽ(p), contradicting
# E(L) ≥ E(p) − 2δ > E(p) − τ), so E(i) < E(p) − τ + 2δ < E(p) cannot be the maximum. τ = 1e-9 S (S bounds the size of the
# terms) is orders of magnitude above δ ≲ 10 eps S.
function _kq_range(tb::_KQTable, k1::Int, Nu::Int, Nv::Int, Nw::Int, cut::Float64)
    n = length(tb.s64)
    E(i) = _kq_E(tb, i, k1, Nu, Nv, Nw)
    a, b = 1, n                                    # first i in [a, b] with E(i+1) − E(i) ≤ 0 (n if none)
    while a < b
        m = (a + b) >>> 1
        E(m + 1) - E(m) <= 0 ? (b = m) : (a = m + 1)
    end
    p = a; Ep = E(p)
    S = 1.0 + k1 * max(abs(tb.s64[1]), abs(tb.s64[n])) + Nu * max(abs(tb.lnu64[1]), abs(tb.lnu64[n])) +
        Nv * max(abs(tb.lnv64[1]), abs(tb.lnv64[n])) + Nw * max(abs(tb.lnw64[1]), abs(tb.lnw64[n]))   # ln(ρ + c eˢ) is monotonic
    τ = 1e-9 * S
    L = p; while L > 1 && E(L - 1) >= Ep - τ; L -= 1; end
    R = p; while R < n && E(R + 1) >= Ep - τ; R += 1; end
    Emax = -Inf; imax = L
    for i in L:R
        Ei = E(i)
        Ei > Emax && (Emax = Ei; imax = i)
    end
    lo = imax; hi = imax
    while lo > firstindex(tb.s64) && _kq_E(tb, lo - 1, k1, Nu, Nv, Nw) >= Emax - cut; lo -= 1; end
    while hi < lastindex(tb.s64) && _kq_E(tb, hi + 1, k1, Nu, Nv, Nw) >= Emax - cut; hi += 1; end
    (Emax, lo, hi, imax)
end

function _kq_sum(tb::_KQTable{Q}, k1::Int, Nu::Int, Nv::Int, Nw::Int, Emax::Float64, lo::Int, hi::Int, jc::Int) where {Q}
    a0 = zero(Q); a1 = zero(Q); a2 = zero(Q); ev = zero(Q); Em = Q(Emax)
    @inbounds for i in lo:hi
        f = exp(k1 * tb.s[i] - Nu * tb.lnu[i] - Nv * tb.lnv[i] - Nw * tb.lnw[i] - Em)
        fs = f * tb.s[i]
        a0 += f; a1 += fs; a2 += fs * tb.s[i]
        iseven(i) && (ev += f)
    end
    (a0, a1, a2, ev)
end

# BigFloat builds the integrand as products of powers f = [t^{κ+1} (ρu+t)^{−Nu}] · [(ρv+t)^{−Nv} (ρw+c_w t)^{−Nw}] (exp in MPFR
# allocates and is 54× slower). In the batched computation per table (`_kernels_batch!`) each bracket becomes a table of
# pairs, so one product per node remains (the one-at-a-time `_kq_sum` builds the same pair values and the same sums).
# Nodes are equally spaced in s (s_j = s_c + (j − c)h, c the peak node), so Σ s f and Σ s² f follow from Σ f, Σ m f, Σ m² f
# (m = j − c), built by cumulative sums without multiplying by m, m² (`_kq_moments!`).
# The sums are fixed-point integers (f > 0, no cancellation): unit 2^{eref − F} (eref = exponent of the peak f + 8,
# F = 64nl + 40); f_j is the product of the pair mantissas (nl × nl words) shifted and truncated. Each truncation is
# ≤ 2^{−F+9} f_peak, orders below eps (2^{−64nl}); additions are only word carries.

"Fixed-point cumulative sum workspace (na = nl + 2 words, headroom up to 2^{45} f_peak) and the BigFloat targets."
struct _KQAcc
    nl::Int; na::Int; F::Int
    pr::Vector{UInt64}                       # product (2nl words)
    v::Vector{UInt64}                        # f_j (na words)
    C::Vector{UInt64}; E::Vector{UInt64}; G::Vector{UInt64}; ev::Vector{UInt64}
    S0::Vector{UInt64}; S1::Vector{UInt64}; S2::Vector{UInt64}; W::Vector{UInt64}
    a::BigFloat; b::BigFloat; x::BigFloat; g::BigFloat   # scratch for building pair values one at a time
    z::BigInt
    s0::BigFloat; s1::BigFloat; s2::BigFloat; evb::BigFloat   # Σ f, Σ m f, Σ m² f, Σ f over even nodes
end
function _KQAcc(p::Int)
    nl = cld(p, 64); na = nl + 2
    w() = zeros(UInt64, na)
    _KQAcc(nl, na, 64nl + 40, zeros(UInt64, 2nl), w(), w(), w(), w(), w(), w(), w(), w(), w(),
           (BigFloat(0; precision=p) for _ in 1:4)..., Base.GMP.MPZ.realloc2(64na + 64), (BigFloat(0; precision=p) for _ in 1:4)...)
end

@inline _mpfr_exp(p::Ptr{BigFloat}) = unsafe_load(Ptr{Clong}(p + 16))
@inline _mpfr_d(p::Ptr{BigFloat}) = unsafe_load(Ptr{Ptr{UInt64}}(p + 24))

# v = floor(src · 2^{−r}) (src has ns words; r < 0 shifts left)
@inline function _fix_shift!(v::Vector{UInt64}, src, ns::Int, na::Int, r::Int)
    if r >= 0
        q = r >> 6; b = r & 63
        @inbounds for l in 1:na
            k = q + l
            lo = k <= ns ? src[k] : UInt64(0)
            v[l] = b == 0 ? lo : (lo >> b) | ((k + 1 <= ns ? src[k+1] : UInt64(0)) << (64 - b))
        end
    else
        s = -r; q = s >> 6; b = s & 63
        @inbounds for l in 1:na
            k = l - q
            hi = 1 <= k <= ns ? src[k] : UInt64(0)
            v[l] = b == 0 ? hi : (hi << b) | (1 <= k - 1 <= ns ? src[k-1] >> (64 - b) : UInt64(0))
        end
    end
    v
end
struct _LimbPtr; d::Ptr{UInt64}; end
@inline Base.getindex(p::_LimbPtr, i::Int) = unsafe_load(p.d, i)

# f_j = A·B (A alone if B is C_NULL) as an integer in units 2^{eref − F} into acc.v
@inline function _fix_f!(acc::_KQAcc, pa::Ptr{BigFloat}, pb::Ptr{BigFloat}, eref::Int)
    nl = acc.nl
    if pb == C_NULL
        return _fix_shift!(acc.v, _LimbPtr(_mpfr_d(pa)), nl, acc.na, eref - _mpfr_exp(pa) + 64nl - acc.F)
    end
    pr = acc.pr; dA = _mpfr_d(pa); dB = _mpfr_d(pb)
    # mantissa product (2nl words) with GMP's mpn_mul_n (same value as schoolbook multiplication)
    GC.@preserve pr ccall((:__gmpn_mul_n, :libgmp), Cvoid, (Ptr{UInt64}, Ptr{UInt64}, Ptr{UInt64}, Clong), pr, dA, dB, nl)
    _fix_shift!(acc.v, pr, 2nl, acc.na, eref - _mpfr_exp(pa) - _mpfr_exp(pb) + 128nl - acc.F)
end
@inline function _fix_add!(a::Vector{UInt64}, b::Vector{UInt64}, na::Int)
    c = UInt64(0)
    @inbounds for l in 1:na
        t = UInt128(a[l]) + b[l] + c
        a[l] = t % UInt64; c = (t >> 64) % UInt64
    end
    a
end
@inline function _fix_sub!(a::Vector{UInt64}, b::Vector{UInt64}, na::Int)   # a −= b (a ≥ b)
    c = UInt64(0)
    @inbounds for l in 1:na
        x = a[l]; y = b[l] + c
        cy = (y < c) | (x < y)
        a[l] = x - y; c = UInt64(cy)
    end
    a
end
@inline function _fix_lt(a::Vector{UInt64}, b::Vector{UInt64}, na::Int)
    @inbounds for l in na:-1:1
        a[l] != b[l] && return a[l] < b[l]
    end
    false
end
# x = (±)a · 2^{e} (one rounding)
function _fix_tobig!(x::BigFloat, z::BigInt, a::Vector{UInt64}, na::Int, e::Int, neg::Bool=false)
    n = na
    @inbounds while n > 0 && a[n] == 0; n -= 1; end
    @inbounds for l in 1:n; unsafe_store!(z.d, a[l], l); end
    z.size = neg ? -Cint(n) : Cint(n)
    ccall((:mpfr_set_z_2exp, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigInt}, Clong, Int32), x, z, e, 0)
    x
end

# One side of the peak, from the outside toward the peak (decreasing |m|): after C += f, E += C, G += E,
# C = Σ f, E = Σ |m| f, G = Σ |m|(|m|+1)/2 f.
@inline function _kq_side!(acc::_KQAcc, ab::F, js, eref::Int) where {F}
    na = acc.na; C, E, G, ev, v = acc.C, acc.E, acc.G, acc.ev, acc.v
    fill!(C, 0); fill!(E, 0); fill!(G, 0)
    @inbounds for j in js
        pa, pb = ab(j)
        _fix_f!(acc, pa, pb, eref)
        _fix_add!(C, v, na); _fix_add!(E, C, na); _fix_add!(G, E, na)
        iseven(j) && _fix_add!(ev, v, na)
    end
    nothing
end

"""
From f_j = A_j B_j on nodes lo:hi (`ab(j)` returns two MPFR pointers, B may be C_NULL), build Σ f, Σ m f, Σ m² f
(m = j − jc) and Σ f over even nodes in acc.s0, s1, s2, evb (exact integer sums, each rounded once). Order: right of the
peak (hi → jc+1), left (lo → jc−1), the peak. Σ m² f = 2 Σ m(m+1)/2 f − Σ m f.
"""
function _kq_moments!(acc::_KQAcc, ab::F, lo::Int, jc::Int, hi::Int) where {F}
    na = acc.na
    pa, pb = ab(jc)
    eref = _mpfr_exp(pa) + (pb == C_NULL ? 0 : _mpfr_exp(pb)) + 8
    S0, S1, S2, W, C, E, G = acc.S0, acc.S1, acc.S2, acc.W, acc.C, acc.E, acc.G
    fill!(acc.ev, 0)
    _kq_side!(acc, ab, hi:-1:jc+1, eref)
    copyto!(S0, C); copyto!(S1, E); copyto!(S2, G)          # right: S2 first gets G_R
    _kq_side!(acc, ab, lo:jc-1, eref)
    _fix_add!(S0, C, na); _fix_add!(S2, G, na)
    _fix_add!(S2, S2, na)                                    # 2(G_R + G_L)
    copyto!(W, S1); _fix_add!(W, E, na); _fix_sub!(S2, W, na)   # − (E_R + E_L)
    neg = _fix_lt(S1, E, na)                                 # Σ m f = E_R − E_L
    neg ? (copyto!(W, E); _fix_sub!(W, S1, na); copyto!(S1, W)) : _fix_sub!(S1, E, na)
    pa, pb = ab(jc)
    _fix_f!(acc, pa, pb, eref)
    _fix_add!(S0, acc.v, na)
    iseven(jc) && _fix_add!(acc.ev, acc.v, na)
    e = eref - acc.F
    _fix_tobig!(acc.s0, acc.z, S0, na, e); _fix_tobig!(acc.s1, acc.z, S1, na, e, neg)
    _fix_tobig!(acc.s2, acc.z, S2, na, e); _fix_tobig!(acc.evb, acc.z, acc.ev, na, e)
    acc
end

"(Σ f, Σ s f, Σ s² f, Σ f over even nodes) · e^{−Emax} (s_c is s at the peak node, h the step)."
function _kq_finish(acc::_KQAcc, sc::BigFloat, h::Float64, Emax::Float64)
    p = precision(acc.s0)
    hb = BigFloat(h; precision=p)
    t1 = hb * acc.s1                                   # h Σ m f
    a1 = sc * acc.s0 + t1                              # Σ s f = s_c Σ f + h Σ m f
    a2 = sc * (sc * acc.s0 + 2t1) + hb * hb * acc.s2   # Σ s² f = s_c² Σ f + 2 s_c h Σ m f + h² Σ m² f
    e = exp(BigFloat(-Emax; precision=p))              # the caller multiplies by e^{Emax}, so divide here
    (acc.s0 * e, a1 * e, a2 * e, acc.evb * e)
end

# how the pair product [(ρv+t)^{−Nv} (ρw+c_w t)^{−Nw}] is built (1 if both are 0, then f is the first bracket itself)
@inline _kq_vw(Nv, Nw) = Nv > 0 && Nw > 0 ? 3 : Nv > 0 ? 1 : Nw > 0 ? 2 : 0

function _kq_sum(tb::_KQTable{BigFloat}, k1::Int, Nu::Int, Nv::Int, Nw::Int, Emax::Float64, lo::Int, hi::Int, jc::Int)
    p = precision(tb.s[1])
    acc = _KQAcc(p)
    vw = _kq_vw(Nv, Nw)
    ab = j -> begin                             # same values as the pair tables of the batched computation (same operations, same order)
        a, b, x, g = acc.a, acc.b, acc.x, acc.g
        mpfr_powi!(a, tb.t[j], k1, x)
        Nu > 0 && (mpfr_powi!(g, tb.iu[j], Nu, x); mpfr_mul!(a, a, g))
        vw & 1 != 0 && mpfr_powi!(b, tb.iv[j], Nv, x)
        vw == 2 && mpfr_powi!(b, tb.iw[j], Nw, x)
        vw == 3 && (mpfr_powi!(g, tb.iw[j], Nw, x); mpfr_mul!(b, b, g))
        (mpfr_ptr(a), vw == 0 ? Ptr{BigFloat}(C_NULL) : mpfr_ptr(b))
    end
    GC.@preserve acc _kq_moments!(acc, ab, lo, jc, hi)
    _kq_finish(acc, tb.s[jc], tb.h, Emax)
end

"""
    kernels(engine, T, κ, Nu, Nv, Nw, ρu, ρv, ρw, cw, cu=1) -> (K₀, K₁, K₂)

Kernels for j = 0, 1, 2 together (the same integrand with (ln t)^j weights, so node evaluations are shared).
Compares the sum with step h and the sum over even nodes (step 2h): relative difference δ. The discretization error is
ε(h) = P(h) e^{−2π²/h}, so δ ≈ ε(2h); the prefactor P grows like h^{−(m−1)} for a pole of order m = Nu+Nv+Nw, hence
ε(h) ≈ δ e^{−π²/h} 2^{m−1}. If that exceeds eps/10 of the working precision, the table with half the step is used. The
rounding floor of the sum is about 100 eps, so accepted kernels lie within it.
"""
function kernels(e::KernelQuadrature{Q}, ::Type{T}, κ::Int, Nu::Int, Nv::Int, Nw::Int,
                 ρu::T, ρv::T, ρw::T, cw::T, cu::T=one(T)) where {Q,T}
    k1 = κ + 1
    for level in 0:4:4e.maxlevel                  # halve the step each time
        tb = _kq_table(e, ρu, ρv, ρw, cw, level, cu)
        Emax, lo, hi, jc = _kq_range(tb, k1, Nu, Nv, Nw, e.cut)
        r = _kq_accept(e, T, level, Nu, Nv, Nw, Emax, _kq_sum(tb, k1, Nu, Nv, Nw, Emax, lo, hi, jc)...)
        r === nothing || return r
    end
end

"Step h/2^{q/4} of table level q (quarter octaves; exactly h over a power of 2 when q is a multiple of 4)."
_kq_h(e::KernelQuadrature, q::Int) = e.h / 2^(q / 4)

"Estimate ln ε(h) of the discretization error (see `kernels`)."
_kq_logerr(e::KernelQuadrature, q, m, a0, ev) = log(Float64(abs(a0 - 2ev) / a0)) - π^2 / _kq_h(e, q) + (m - 1) * log(2)

# expected acceptance level for a kernel with ln ε = le at level q: ε(h') = ε(h) e^{2π²/h − 2π²/h'} (h/h')^{m−1} (prefactor ∝ h^{−(m−1)})
function _kq_nextlevel(e::KernelQuadrature, q, m, le)
    h = _kq_h(e, q)
    for q2 in q+1:4e.maxlevel
        h2 = _kq_h(e, q2)
        le + 2π^2 / h - 2π^2 / h2 + (m - 1) * log(h / h2) <= e.logtol && return q2
    end
    4e.maxlevel
end

# (K₀, K₁, K₂) if the sum at step level `level` (quarter octaves) is accepted, nothing if the step must be refined
function _kq_accept(e::KernelQuadrature{Q}, ::Type{T}, level, Nu, Nv, Nw, Emax, a0, a1, a2, ev) where {Q,T}
    h = _kq_h(e, level)
    le = _kq_logerr(e, level, Nu + Nv + Nw, a0, ev)
    if le <= e.logtol || level >= 4e.maxlevel
        level > 0 && Threads.atomic_add!(e.nrefined, 1)
        if !(le <= e.logtol)          # accepted at the step cap (or NaN): the kernel error exceeds the working-precision threshold
            Threads.atomic_add!(e.ncapped, 1)
            Threads.atomic_max!(e.capexcess, isfinite(le) ? (le - e.logtol) / log(10) : Inf)
        end
        sc = Q(h) * exp(Q(Emax))
        return (T(a0 * sc), T(a1 * sc), T(a2 * sc))
    end
    nothing
end

# BigFloat kernels of one table (ρu, ρv, ρw, c_w) together: tables of the node powers t^k, (ρ+t)^{−N}, and of the pairs
# [t^{κ+1}(ρu+t)^{−Nu}] and [(ρv+t)^{−Nv}(ρw+c_w t)^{−Nw}] over the node ranges of the kernels that use them (one product per
# node). Every power and pair product uses the same operations in the same order as `_kq_sum`, so the result is bit-identical
# to `kernels`. Kernels not accepted at step h (needing finer tables) are returned with their expected level for a redo.
function _kernels_batch!(st, e::KernelQuadrature{BigFloat}, items, cw::T, ntasks::Int, q::Int=0, cu::T=one(T)) where {T}
    _, (_, _, _, _, ρu, ρv, ρw) = items[1]
    tb = _kq_table(e, ρu, ρv, ρw, cw, q, cu)
    n = length(tb.s); p = precision(tb.s[1])
    rng = Vector{NTuple{4,Int}}(undef, length(items))      # per kernel (lo, hi, peak, bits of Emax)
    Threads.@threads for i in eachindex(items)
        κ, Nu, Nv, Nw = items[i][2]
        Emax, lo, hi, jc = _kq_range(tb, κ + 1, Nu, Nv, Nw, e.cut)
        rng[i] = (lo, hi, jc, reinterpret(Int, Emax))
    end
    # per pair, the union of the node ranges of the kernels that use it
    ra = Dict{Tuple{Int,Int},Tuple{Int,Int}}(); rb = Dict{Tuple{Int,Int},Tuple{Int,Int}}()
    for (i, it) in enumerate(items)
        κ, Nu, Nv, Nw = it[2]; lo, hi = rng[i]
        for (d, k) in ((ra, (κ + 1, Nu)), (rb, (Nv, Nw)))
            r = get(d, k, (lo, hi)); d[k] = (min(r[1], lo), max(r[2], hi))
        end
    end
    kmax = maximum(k -> k[1], keys(ra))
    numax, nvmax, nwmax = maximum(k -> k[2], keys(ra)), maximum(k -> k[1], keys(rb)), maximum(k -> k[2], keys(rb))
    # power tables: power k only over the union [lo_k, hi_k] of the ranges of the pairs using it, packed in one buffer
    # (node j of power k of table s is row off[s][k] + j of PP)
    span = [(fill(n + 1, max(m, 1)), fill(0, max(m, 1))) for m in (kmax, numax, nvmax, nwmax)]
    widen!(s, k, r) = k > 0 && (s[1][k] = min(s[1][k], r[1]); s[2][k] = max(s[2][k], r[2]))
    for ((k1, Nu), r) in ra; widen!(span[1], k1, r); widen!(span[2], Nu, r); end
    for ((Nv, Nw), r) in rb; widen!(span[3], Nv, r); widen!(span[4], Nw, r); end
    off = [zeros(Int, length(sp[1])) for sp in span]; seg = NTuple{4,Int}[]; np = 0    # seg: (table, power, first node, last node)
    for s in 1:4, k in eachindex(off[s])
        lo, hi = span[s][1][k], span[s][2][k]
        hi >= lo || continue
        off[s][k] = np - lo + 1; np += hi - lo + 1
        for c in Iterators.partition(lo:hi, 1024); push!(seg, (s, k, first(c), last(c))); end
    end
    # pair tables: node j of pair c is row (oa[c] + j − lo_c) of PA (packed in one buffer)
    pairs_a = sort!(collect(keys(ra))); pairs_b = sort!(collect(keys(rb)))
    lay(pairs, d) = (o = Dict{Tuple{Int,Int},Int}(); tot = 0; for k in pairs; o[k] = tot; tot += d[k][2] - d[k][1] + 1; end; (o, tot))
    oa, na = lay(pairs_a, ra); ob, nb = lay(pairs_b, rb)
    # the three buffers are malloc'ed and freed right after use (left to the GC, per-batch buffers pile up until the next GC);
    # the power tables are freed once the pair tables are built
    PP = PackedMatrix(max(np, 1), 1; precision=p, malloc=true)
    PA = PackedMatrix(na, 1; precision=p, malloc=true); PB = PackedMatrix(max(nb, 1), 1; precision=p, malloc=true)
    try
        bases = (tb.t, tb.iu, tb.iv, tb.iw)
        nxs = Threads.Atomic{Int}(1)
        GC.@preserve PP @sync for _ in 1:ntasks
            Threads.@spawn begin
                x = BigFloat(; precision=p); z = BigFloat(; precision=p)
                while (is = Threads.atomic_add!(nxs, 1)) <= length(seg)
                    s, k, lo, hi = seg[is]; b = bases[s]; o = off[s][k]
                    @inbounds for j in lo:hi; mpfr_set!(_el(PP, o + j, 1), mpfr_powi!(z, b[j], k, x)); end
                end
            end
        end
        pw(s, j, k) = _el(PP, off[s][k] + j, 1)
        GC.@preserve PP PA PB begin
            @sync for c in Iterators.partition(eachindex(pairs_a), cld(length(pairs_a), 4ntasks))
                Threads.@spawn for ic in c
                    k1, Nu = pairs_a[ic]; lo, hi = ra[(k1, Nu)]; o = oa[(k1, Nu)]
                    @inbounds for j in lo:hi
                        y = _el(PA, o + j - lo + 1, 1)
                        Nu > 0 ? mpfr_mul!(y, pw(1, j, k1), pw(2, j, Nu)) : mpfr_set!(y, pw(1, j, k1))
                    end
                end
            end
            @sync for c in Iterators.partition(eachindex(pairs_b), cld(length(pairs_b), 4ntasks))
                Threads.@spawn for ic in c
                    Nv, Nw = pairs_b[ic]; lo, hi = rb[(Nv, Nw)]; o = ob[(Nv, Nw)]; vw = _kq_vw(Nv, Nw)
                    vw == 0 && continue
                    @inbounds for j in lo:hi
                        y = _el(PB, o + j - lo + 1, 1)
                        vw == 1 ? mpfr_set!(y, pw(3, j, Nv)) : vw == 2 ? mpfr_set!(y, pw(4, j, Nw)) :
                                  mpfr_mul!(y, pw(3, j, Nv), pw(4, j, Nw))
                    end
                end
            end
        end
        free!(PP)
        # kernels: hand out in an order that keeps kernels sharing a pair together (the result does not depend on the order)
        order = sortperm(eachindex(items); by=i -> (items[i][2][1], items[i][2][2], items[i][2][3], items[i][2][4]))
        next = Threads.Atomic{Int}(1)
        retry = [Tuple{eltype(items),Int}[] for _ in 1:ntasks]
        @sync for t in 1:ntasks
            Threads.@spawn begin
                acc = _KQAcc(p)
                while (ii = Threads.atomic_add!(next, 1)) <= length(items)
                    i = order[ii]
                    key, (κ, Nu, Nv, Nw, _, _, _) = items[i]
                    lo, hi, jc, eb = rng[i]; Emax = reinterpret(Float64, eb)
                    ka = (κ + 1, Nu); kb = (Nv, Nw)
                    da = oa[ka] + 1 - ra[ka][1]; db = ob[kb] + 1 - rb[kb][1]; vw = _kq_vw(Nv, Nw)
                    GC.@preserve PA PB begin
                        ab = vw == 0 ? (j -> (_el(PA, da + j, 1), Ptr{BigFloat}(C_NULL))) : (j -> (_el(PA, da + j, 1), _el(PB, db + j, 1)))
                        _kq_moments!(acc, ab, lo, jc, hi)
                    end
                    r = _kq_accept(e, T, q, Nu, Nv, Nw, Emax, _kq_finish(acc, tb.s[jc], tb.h, Emax)...)
                    if r === nothing
                        m = Nu + Nv + Nw
                        a0, _, _, ev = _kq_finish(acc, tb.s[jc], tb.h, Emax)
                        push!(retry[t], (items[i], _kq_nextlevel(e, q, m, _kq_logerr(e, q, m, a0, ev))))
                    else
                        _store_put!(st, key, r)
                    end
                end
            end
        end
        reduce(vcat, retry)
    finally
        free!(PP); free!(PA); free!(PB)
    end
end

# =============================================================================
# Element integrals
# =============================================================================
"""
Memo shared across tasks (kernels and element integrals). Values depend only on the key, so any task computes the same bits.
Sharded by hash with a lock per shard. When the total exceeds `cap`, all shards are emptied (bounded memory), except while
`hold` is set (during kernel prefetch: dropping prefetched kernels before the elements would make every task recompute them).
The excess is emptied by `_trim!` before the next dry run.
"""
struct SharedStore{K,V}
    shards::Vector{Dict{K,V}}
    inflight::Vector{Dict{K,Threads.Event}}    # keys being computed (so that several tasks do not compute the same key)
    locks::Vector{Threads.SpinLock}
    count::Threads.Atomic{Int}
    cap::Int
    waited_ns::Threads.Atomic{Int}    # total time spent waiting for other tasks (diagnostic)
    hold::Threads.Atomic{Bool}        # while true, do not empty beyond cap
end
SharedStore{K,V}(; nshards::Int=256, cap::Int=20_000_000) where {K,V} =
    SharedStore{K,V}([Dict{K,V}() for _ in 1:nshards], [Dict{K,Threads.Event}() for _ in 1:nshards],
                     [Threads.SpinLock() for _ in 1:nshards], Threads.Atomic{Int}(0), cap, Threads.Atomic{Int}(0),
                     Threads.Atomic{Bool}(false))
const KernelStore{T} = SharedStore{NTuple{7,Int},NTuple{3,T}}
const ElementStore{T} = SharedStore{NTuple{8,Int},T}

# The locked sections are only Dict get/setindex! (no exceptions), so no try/finally: try/finally or lock(f, l) boxes the
# return values (Float64x4 or kernel triples) and allocates on every read.
function _store_get(st::SharedStore{K,V}, key) where {K,V}
    i = Int(hash(key) % length(st.shards)) + 1
    lock(st.locks[i]); v = get(st.shards[i], key, nothing); unlock(st.locks[i])
    v
end
function _trim!(st::SharedStore)
    st.count[] > st.cap || return st
    for i in eachindex(st.shards)
        lock(st.locks[i]); empty!(st.shards[i]); unlock(st.locks[i])
    end
    Threads.atomic_xchg!(st.count, 0)
    st
end
# drop all memory (at the last stage, before packing; values remain only in the element table)
function _clear!(st::SharedStore)
    for i in eachindex(st.shards)
        lock(st.locks[i]); empty!(st.shards[i]); sizehint!(st.shards[i], 0); unlock(st.locks[i])
    end
    Threads.atomic_xchg!(st.count, 0)
    st
end
function _store_put!(st::SharedStore{K,V}, key, v::V) where {K,V}
    st.hold[] || _trim!(st)
    i = Int(hash(key) % length(st.shards)) + 1
    lock(st.locks[i])
    haskey(st.shards[i], key) || (st.shards[i][key] = v; Threads.atomic_add!(st.count, 1))
    unlock(st.locks[i])
    v
end

# value of a key; if absent, one task computes f() and others asking for the same key wait for it
function _store_get!(f, st::SharedStore{K,V}, key) where {K,V}
    i = Int(hash(key) % length(st.shards)) + 1
    lock(st.locks[i])
    v = get(st.shards[i], key, nothing)
    if v !== nothing
        unlock(st.locks[i]); return v
    end
    ev = get(st.inflight[i], key, nothing)
    owner = ev === nothing
    owner && (ev = st.inflight[i][key] = Threads.Event())
    unlock(st.locks[i])
    if !owner
        t0 = time_ns(); wait(ev); Threads.atomic_add!(st.waited_ns, Int(time_ns() - t0))
        w = _store_get(st, key)
        return w === nothing ? f()::V : w      # emptied by cap while waiting (or the computation threw)
    end
    r = try
        f()::V
    catch
        lock(st.locks[i]); delete!(st.inflight[i], key); unlock(st.locks[i]); notify(ev)
        rethrow()
    end
    _store_put!(st, key, r)
    lock(st.locks[i]); delete!(st.inflight[i], key); unlock(st.locks[i]); notify(ev)
    r
end

"(x, y) grid for a ≠ b: Gauss–Legendre nodes on [0,1] (shared by x and y) and φ, ln φ at the nodes."
struct ABGrid{T}
    n::Int
    x::Vector{T}
    w::Vector{T}
    φ::Vector{T}
    lnφ::Vector{T}
end

"""
    IntegralContext(T, engine; βσ=0.0, split=false, ...)

What the element integrals need. Each task gets its own `ElementCache`. The log variable is σ = r₁ + r₂ + βσ r₁₂;
`split=true` is the new logarithmic form (powers of s × ln σ). ThreeBodyFC calls `IntegralContext(T, engine, lv::LogVariable)`.
"""
struct IntegralContext{T,E<:KernelEngine}
    engine::E
    store::KernelStore{T}
    elems::ElementStore{T}    # element integrals (shared across stages and tasks)
    λ::T
    lnλ::T
    cw::T
    γE::T
    π2o6::T
    fact::Vector{T}          # fact[n+1] = n!
    binom::Matrix{T}         # binom[n+1, k+1] = C(n,k)
    invΓ::Vector{T}          # invΓ[R] = 1/Γ(R)
    ψ::Vector{T}             # ψ[R]
    ψ1::Vector{T}            # ψ₁[R]
    lnfact::Vector{T}        # lnfact[n+1] = ln n!
    split::Bool              # LogVariable(split=true): s^d [ln σ]^j
    βs::T                    # βσ (angular part of split: φ(x) = 1 + βσ(1 − x))
    xleg::Vector{T}          # Gauss–Legendre nodes on [0,1] (angular part of split)
    wleg::Vector{T}
    φleg::Vector{T}          # φ(x) and ln φ(x) at the nodes
    lnφleg::Vector{T}
    abgrids::Dict{Int,ABGrid{T}} # (x, y) grids for a ≠ b (node count → grid), chosen by exponent ratio and built on demand (_abgrid)
    ablock::ReentrantLock
    nab::Int
    monos::SharedStore{NTuple{4,Int},Vector{Tuple{Int,Int,Int,T}}}   # monomial lists of the σ path (per A,B,C,m, shared across tasks)
    cu::T                    # coefficient of u in σ̃ = c_u u + v + c_w w (1 for w₂ = 1)
    # quantities on the split and a≠b grids (shared across tasks, read only; per-task copies multiplied the memory)
    angys::SharedStore{NTuple{6,Int},NTuple{3,Vector{T}}}
    dpows::SharedStore{NTuple{4,Int},Matrix{T}}
    epsgs::SharedStore{NTuple{3,Int},Tuple{Vector{T},Vector{T},Matrix{T},Matrix{T}}}
    # perimetric expansions (v+w)^A (u+w)^B (u+v)^C (per A,B,C, shared across tasks, read only; per-task copies duplicated
    # (A+1)(B+1)(C+1) values per term in every task)
    perims::SharedStore{NTuple{3,Int},Vector{Tuple{Int,Int,Int,T}}}
end

function IntegralContext(::Type{T}, engine::KernelEngine; βσ::Float64=0.0, split::Bool=false, w2::Float64=1.0, nmax::Int=300,
                         kernel_cap::Int=20_000_000, elem_cap::Int=20_000_000, nleg::Int=192, nab::Int=80) where {T}
    λb = _big(() -> 1 + BigFloat(βσ))     # 1+βσ without Float64 rounding
    split && w2 != 1 && throw(ArgumentError("split requires w₂ = 1"))
    cwb = _big(() -> (1 + BigFloat(w2)) / λb); cub = _big(() -> (BigFloat(w2) + BigFloat(βσ)) / λb)
    xl, wl = split ? gauss_legendre01(T, nleg) : (T[], T[])
    βb = _big(() -> BigFloat(βσ))
    φl = [T(_big(() -> 1 + βb * (1 - _tobig(x)))) for x in xl]
    lnφl = [T(_big(() -> log(1 + βb * (1 - _tobig(x))))) for x in xl]
    abgrids = Dict{Int,ABGrid{T}}()
    if split
        for m in (nab, round(Int, 1.6nab))     # defaults 80, 128; grids for exponent ratios above 10 are built on demand
            abgrids[m] = _make_abgrid(T, m, βb)
        end
    end
    fact = [factT(T, n) for n in 0:nmax]
    binom = [k <= n ? T(binomial(BigInt(n), BigInt(k))) : zero(T) for n in 0:nmax, k in 0:nmax]
    IntegralContext{T,typeof(engine)}(engine, KernelStore{T}(; cap=kernel_cap), ElementStore{T}(; cap=elem_cap), T(λb), T(_big(() -> log(λb))), T(cwb),
                                      euler_gamma(T), pi2over6(T), fact, binom,
                                      [gamma_inv(T, R) for R in 1:nmax], [digamma_int(T, R) for R in 1:nmax],
                                      [trigamma_int(T, R) for R in 1:nmax],
                                      [T(_big(() -> logfactorial_big(n))) for n in 0:nmax],
                                      split, T(βb), xl, wl, φl, lnφl, abgrids, ReentrantLock(), nab,
                                      SharedStore{NTuple{4,Int},Vector{Tuple{Int,Int,Int,T}}}(; nshards=64, cap=typemax(Int)), T(cub),
                                      # capped by entry count; beyond it emptied and recomputed (same values)
                                      SharedStore{NTuple{6,Int},NTuple{3,Vector{T}}}(; nshards=64, cap=200_000),
                                      SharedStore{NTuple{4,Int},Matrix{T}}(; nshards=64, cap=4_000),
                                      SharedStore{NTuple{3,Int},Tuple{Vector{T},Vector{T},Matrix{T},Matrix{T}}}(; nshards=16, cap=2_000),
                                      SharedStore{NTuple{3,Int},Vector{Tuple{Int,Int,Int,T}}}(; nshards=64, cap=typemax(Int)))
end

logfactorial_big(n::Int) = sum((log(BigFloat(i)) for i in 2:n); init=BigFloat(0))

"""
    gauss_legendre01(T, n) -> (x, w)

Gauss–Legendre nodes and weights on [0, 1], computed by Newton's method in BigFloat (`_bigprec()` bits) and rounded to T.
"""
function gauss_legendre01(::Type{T}, n::Int) where {T}
    # Newton starts from Float64 iterates, uses the ±z symmetry (half the roots), and distributes roots over threads.
    # The BigFloat recurrence uses in-place MPFR (no per-operation allocation or precision lookups), and the Newton precision
    # doubles 106 → 212 → … with only the last two iterations at full precision.
    function legp(z, n)
        p0 = one(z); p1 = z
        for k in 2:n
            p0, p1 = p1, ((2k - 1) * z * p1 - (k - 1) * p0) / k
        end
        p1, n * (z * p1 - p0) / (z * z - 1)
    end
    _big() do
        prec = precision(BigFloat)
        x = Vector{BigFloat}(undef, n); w = Vector{BigFloat}(undef, n)
        m = cld(n, 2)
        levels = Int[]; b = 106
        while b < prec
            push!(levels, b); b *= 2
        end
        append!(levels, (prec, prec))
        Threads.@threads for i in 1:m
            zf = cos(π * (i - 0.25) / (n + 0.5))
            for _ in 1:10
                p, pp = legp(zf, n); dz = p / pp; zf -= dz
                abs(dz) < 4eps() && break
            end
            local z, pp
            if isodd(n) && i == m
                z = BigFloat(0; precision=prec)
                pp = _legp_big!(z, n, ntuple(_ -> BigFloat(; precision=prec), 5))[2]
            else
                z = BigFloat(zf; precision=levels[1])
                k = 0
                while true
                    k += 1
                    lv = levels[min(k, length(levels))]
                    zl = BigFloat(z; precision=lv)
                    buf = ntuple(_ -> BigFloat(; precision=lv), 5)
                    p, pp = _legp_big!(zl, n, buf)
                    dz = BigFloat(; precision=lv); mpfr_div!(dz, p, pp)
                    mpfr_sub!(zl, zl, dz)
                    z = zl
                    # compare as BigFloat exponents (Float64 2.0^-(prec−12) underflows to 0 for prec > 1086)
                    k >= length(levels) && (iszero(dz) || exponent(dz) < 12 - prec) && break
                    k > length(levels) + 100 && error("gauss_legendre01: Newton does not converge (n = $n, root $i)")
                end
                pp = _legp_big!(z, n, ntuple(_ -> BigFloat(; precision=prec), 5))[2]
            end
            x[i] = (1 - z) / 2; w[i] = 1 / ((1 - z * z) * pp * pp)
            x[n+1-i] = (1 + z) / 2; w[n+1-i] = w[i]
        end
        (T.(x), T.(w))
    end
end

_mpfr_div_si!(z::BigFloat, x::BigFloat, k::Integer) =
    (ccall((:mpfr_div_si, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Clong, Int32), z, x, k, 0); z)
# P_n(z) and P_n'(z) with in-place MPFR (the 5 numbers in buf have z's precision; returns two of them)
function _legp_big!(z::BigFloat, n::Int, buf::NTuple{5,BigFloat})
    p0, p1, t, u, v = buf
    ccall((:mpfr_set_si, :libmpfr), Int32, (Ref{BigFloat}, Clong, Int32), p0, 1, 0)
    mpfr_set!(p1, z)
    for k in 2:n
        mpfr_mul!(t, z, p1); mpfr_mul_si!(t, t, 2k - 1)
        mpfr_mul_si!(u, p0, k - 1); mpfr_sub!(t, t, u); _mpfr_div_si!(t, t, k)
        p0, p1, t = p1, t, p0
    end
    # P_n' = n (z P_n − P_{n−1}) / (z² − 1)
    mpfr_mul!(u, z, p1); mpfr_sub!(u, u, p0); mpfr_mul_si!(u, u, n)
    mpfr_mul!(v, z, z); ccall((:mpfr_sub_si, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Clong, Int32), v, v, 1, 0)
    mpfr_div!(u, u, v)
    (p1, u)
end

"Task-local cache (elements and kernels), not shared between threads."
struct ElementCache{T}
    elem::Dict{NTuple{8,Int},T}
    kern::Dict{NTuple{7,Int},NTuple{3,T}}
    ang::Dict{NTuple{8,Int},T}       # angular part of split
    angy::Dict{NTuple{6,Int},NTuple{3,Vector{T}}}   # split, a≠b: y integrals per x node (k = 0,1,2)
    epsg::Dict{NTuple{3,Int},Tuple{Vector{T},Vector{T},Matrix{T},Matrix{T}}}   # split, a≠b: ln E₀, ln F, D, ln D
    wvec::Dict{NTuple{5,Int},Vector{T}}   # split, a≠b: w x^r (1−x)^P φ^{−R} (ln φ)^e at the x nodes (per grid size)
    term::Dict{NTuple{9,Int},T}           # split, a≠b: integral of one perimetric term (p,q,r,N,R,J; a,b,c)
    dpow::Dict{NTuple{4,Int},Matrix{T}}   # split, a≠b: D^m on the (x,t) grid
    wty::Dict{NTuple{3,Int},Vector{T}}    # split, a≠b: t weights w t^p (1−t)^q
    perim::Dict{NTuple{3,Int},Vector{Tuple{Int,Int,Int,T}}}   # perimetric expansions (per A,B,C)
    mono::Dict{NTuple{4,Int},Vector{Tuple{Int,Int,Int,T}}}    # σ path: perimetric expansion × σ̃^m as monomials (per A,B,C,m)
    ipow::Dict{NTuple{3,Int},NTuple{3,Vector{T}}}             # σ path: powers of 1/ρu, 1/ρv, 1/ρw (per exponent set)
    cap::Int
    kfrozen::Bool    # the shared kernel memory is read only (prefetch done, no writers): read without locks, no local copies
    rbuf::Vector{Tuple{UInt,NTuple{7,Int}}}   # dry run: recorded kernel keys (with hashes), moved to the shared set when full
    rρ::Dict{NTuple{3,Int},NTuple{3,T}}      # dry run: ρ of exponent sets whose Float64 sums are inexact (first seen)
    # split, a = b, c = 0 (s,t,u) path (_element_split_stu): sums with cancellation in BigFloat at working precision + 64 bits
    stuS::Dict{NTuple{5,Int},T}              # S_i(A,B,C,R) = Σ_k c_k/(k+1) Ω(C+k+1, R, i) (rounded to the working precision)
    stuc::Dict{NTuple{2,Int},Vector{BigFloat}}   # c_k/(k+1) (even k, per A,B, high precision)
    stuΩ::Dict{NTuple{3,Int},BigFloat}       # Ω(m,R,i) = ∫₀¹ y^m φ^{−R} (ln φ)^i, φ = 1 + βσ y (high precision)
    stuleg::Dict{Int,NTuple{4,Vector{BigFloat}}}   # high-precision Gauss–Legendre (per bit count): y, w, φ, ln φ
end
ElementCache{T}(; cap::Int=5_000_000, kfrozen::Bool=false) where {T} =
    ElementCache{T}(Dict{NTuple{8,Int},T}(), Dict{NTuple{7,Int},NTuple{3,T}}(), Dict{NTuple{8,Int},T}(),
                    Dict{NTuple{6,Int},NTuple{3,Vector{T}}}(), Dict{NTuple{3,Int},Tuple{Vector{T},Vector{T},Matrix{T},Matrix{T}}}(),
                    Dict{NTuple{5,Int},Vector{T}}(), Dict{NTuple{9,Int},T}(),
                    Dict{NTuple{4,Int},Matrix{T}}(), Dict{NTuple{3,Int},Vector{T}}(),
                    Dict{NTuple{3,Int},Vector{Tuple{Int,Int,Int,T}}}(),
                    Dict{NTuple{4,Int},Vector{Tuple{Int,Int,Int,T}}}(), Dict{NTuple{3,Int},NTuple{3,Vector{T}}}(), cap, kfrozen,
                    Tuple{UInt,NTuple{7,Int}}[], Dict{NTuple{3,Int},NTuple{3,T}}(),
                    Dict{NTuple{5,Int},T}(), Dict{NTuple{2,Int},Vector{BigFloat}}(), Dict{NTuple{3,Int},BigFloat}(),
                    Dict{Int,NTuple{4,Vector{BigFloat}}}())

@inline _bits(x::Float64) = reinterpret(Int64, x)

function _kernels_cached!(ec::ElementCache{T}, ctx::IntegralContext{T}, κ, Nu, Nv, Nw,
                          ρu::T, ρv::T, ρw::T, rkey::NTuple{3,Int}) where {T}
    key = (κ, Nu, Nv, Nw, rkey...)
    if ec.kfrozen          # nobody writes the shared memory, so read without locks (concurrent Dict reads are safe)
        st = ctx.store
        v = get(st.shards[Int(hash(key) % length(st.shards)) + 1], key, nothing)
        v === nothing || return v
        # not prefetched (e.g. emptied by cap): compute locally without writing to the shared memory
        return get!(() -> kernels(ctx.engine, T, κ, Nu, Nv, Nw, ρu, ρv, ρw, ctx.cw, ctx.cu), ec.kern, key)
    end
    v = get(ec.kern, key, nothing)
    v === nothing || return v
    length(ec.kern) > ec.cap && empty!(ec.kern)
    v = _store_get!(() -> kernels(ctx.engine, T, κ, Nu, Nv, Nw, ρu, ρv, ρw, ctx.cw, ctx.cu), ctx.store, key)
    ec.kern[key] = v
    v
end

"""
Engine that only collects the kernels needed: in a dry run elements record kernel keys and return 0. Used by the assembly
dry run (`prefetch_kernels!`); the collected kernels are then computed in parallel by the real engine.

Keys are buffered task-locally (`ElementCache.rbuf`) and moved, when not in `ctx.store`, to the shared set `seen` (per-task key
dictionaries would duplicate almost all kernels in every task). ρ is not stored in the key: when the Float64 sums of exponents
are exact, ρ = T(Float64 sum) follows from `rkey`; only inexact sets are kept in `rρ`.
"""
struct KernelRecorder{T,E<:KernelEngine} <: KernelEngine
    inner::E
    store::KernelStore{T}                  # real kernel memory (nobody writes it during the dry run, so it is read without locks)
    seen::Vector{Set{NTuple{7,Int}}}       # recorded keys not in store (shared by all tasks, a lock per shard)
    locks::Vector{ReentrantLock}
end

const _RBUF = 1 << 16

"Whether x + y is exact in Float64 (zero TwoSum error)."
@inline _exact_sum(x::Float64, y::Float64) = (s = x + y; t = s - x; (x - (s - t)) + (y - t) == 0)

# dry-run elements only record the kernel keys they need, without element arithmetic
function _element_tilde(ec::ElementCache{T}, ctx::IntegralContext{T,<:KernelRecorder}, A, B, C, D, J,
                        a::Float64, b::Float64, c::Float64) where {T}
    J == 0 && D >= 0 && return zero(T)
    rkey = (_bits(b + c), _bits(a + c), _bits(a + b))
    _exact_sum(b, c) && _exact_sum(a, c) && _exact_sum(a, b) ||
        get!(() -> (T(b) + T(c), T(a) + T(c), T(a) + T(b)), ec.rρ, rkey)
    κ, m = D < 0 ? (-D - 1, 0) : (0, D + 1)
    for (p, q, r, _) in _monomials!(ec, ctx, A, B, C, m)
        key = (κ, p + 1, q + 1, r + 1, rkey...)
        push!(ec.rbuf, (hash(key), key))
    end
    length(ec.rbuf) >= _RBUF && _flush_recorded!(ec, ctx.engine)
    zero(T)
end

"Move buffered keys, deduplicated and not in store, to the shared set."
function _flush_recorded!(ec::ElementCache{T}, rec::KernelRecorder{T}) where {T}
    buf = ec.rbuf
    sort!(buf); unique!(buf)
    st = rec.store; nst = length(st.shards); ns = length(rec.seen)
    i = 1
    while i <= length(buf)
        sh = Int(buf[i][1] % ns) + 1
        j = i
        lock(rec.locks[sh])
        while j <= length(buf) && Int(buf[j][1] % ns) + 1 == sh
            h, key = buf[j]
            haskey(st.shards[Int(h % nst) + 1], key) || push!(rec.seen[sh], key)
            j += 1
        end
        unlock(rec.locks[sh])
        i = j
    end
    empty!(buf)
    nothing
end

"Record the kernels needed by element `key` in a dry run (bypassing the element memory; keys one at a time). The kernel keys do not depend on the power J of ln σ̃ (no kernels only for J = 0 and D ≥ 0), so once."
function _record_element!(ec::ElementCache{T}, rec::IntegralContext{T,<:KernelRecorder}, key::NTuple{8,Int}) where {T}
    A, B, C, D, J, ab, bb, cb = key
    _element_tilde(ec, rec, A, B, C, D, J, reinterpret(Float64, ab), reinterpret(Float64, bb), reinterpret(Float64, cb))
end

"A recording-only context with the same constants as `ctx` (empty memory)."
function _recording_context(ctx::IntegralContext{T,E}; nshards::Int=256) where {T,E}
    _trim!(ctx.store)     # empty the kernels of previous stages if beyond cap (the dry run then collects all kernels needed against this store)
    rec = KernelRecorder{T,E}(ctx.engine, ctx.store, [Set{NTuple{7,Int}}() for _ in 1:nshards], [ReentrantLock() for _ in 1:nshards])
    IntegralContext{T,typeof(rec)}(rec, KernelStore{T}(; cap=ctx.store.cap), ElementStore{T}(; cap=ctx.elems.cap),
                                   (getfield(ctx, i) for i in 4:nfields(ctx))...)
end

"Compute in parallel the kernels recorded in the dry-run task memories `ecs` that are not in `ctx.store`."
function prefetch_kernels!(ctx::IntegralContext{T}, rec::IntegralContext{T,<:KernelRecorder}, ecs::Vector{ElementCache{T}},
                           ntasks::Int) where {T}
    prefetch_kernels!(ctx, _kernel_requests(ctx, rec, ecs, ntasks), ntasks)
end

"""
List of kernels not in `ctx.store` from the dry-run keys: [(key, (κ, Nu, Nv, Nw, ρu, ρv, ρw))], in key order (deterministic).
ρ is shared per `rkey` (inexact sets use the first seen in task order).
"""
function _kernel_requests(ctx::IntegralContext{T}, rec::IntegralContext{T,<:KernelRecorder}, ecs::Vector{ElementCache{T}},
                          ntasks::Int) where {T}
    r = rec.engine
    Threads.@threads for t in eachindex(ecs)
        _flush_recorded!(ecs[t], r)
    end
    inexact = Dict{NTuple{3,Int},NTuple{3,T}}()
    for ec in ecs, (k, ρ) in ec.rρ
        haskey(inexact, k) || (inexact[k] = ρ)
    end
    ρof = Dict{NTuple{3,Int},NTuple{3,T}}()
    for s in r.seen, k in s
        rk = (k[5], k[6], k[7])
        haskey(ρof, rk) || (ρof[rk] = get(() -> map(x -> T(reinterpret(Float64, x)), rk), inexact, rk))
    end
    out = Vector{Vector{Tuple{NTuple{7,Int},Tuple{Int,Int,Int,Int,T,T,T}}}}(undef, length(r.seen))
    Threads.@threads for sh in eachindex(r.seen)
        out[sh] = [(k, (k[1], k[2], k[3], k[4], ρof[(k[5], k[6], k[7])]...)) for k in r.seen[sh]]
        empty!(r.seen[sh])
    end
    sort!(reduce(vcat, out); by=first)
end

"""
    prefetch_kernels!(ctx, todo, ntasks) -> number of kernels computed

Compute the kernels `todo` (`_kernel_requests`) with `ntasks` tasks into `ctx.store` before the element computation (so no
task waits for the same kernel). BigFloat quadrature is batched per table.
"""
function prefetch_kernels!(ctx::IntegralContext{T}, todo::Vector, ntasks::Int) where {T}
    ctx.store.hold[] = true       # do not empty by cap during prefetch (the store seen by the dry run must survive until the elements)
    try
        _prefetch_kernels!(ctx, todo, ntasks)
    finally
        ctx.store.hold[] = false
    end
end
function _prefetch_kernels!(ctx::IntegralContext{T}, todo::Vector, ntasks::Int) where {T}
    if ctx.engine isa KernelQuadrature{BigFloat}
        groups = Dict{NTuple{3,Float64},Vector{eltype(todo)}}()
        for kv in todo
            push!(get!(() -> eltype(todo)[], groups, (Float64(kv[2][5]), Float64(kv[2][6]), Float64(kv[2][7]))), kv)
        end
        for g in values(groups)
            # The step level depends mostly on the pole order m: try a few kernels per m at level 0 and start the rest at the
            # expected level; kernels not accepted are redone together at a finer level chosen from the error estimate
            msum(kv) = kv[2][2] + kv[2][3] + kv[2][4]
            sort!(g; by=first)                    # choice of trial kernels independent of thread order (deterministic)
            bym = Dict{Int,Vector{eltype(g)}}()
            for kv in g; push!(get!(() -> eltype(g)[], bym, msum(kv)), kv); end
            sample = eltype(g)[]; rest = eltype(g)[]
            for v in values(bym), (i, kv) in enumerate(v)
                push!(i in round.(Int, range(1, length(v); length=min(4, length(v)))) ? sample : rest, kv)
            end
            qm = Dict{Int,Int}()
            pending = Dict{Int,Vector{eltype(g)}}()
            for (kv, q2) in _kernels_batch!(ctx.store, ctx.engine, sample, ctx.cw, ntasks, 0, ctx.cu)
                qm[msum(kv)] = max(get(qm, msum(kv), 0), q2)
                push!(get!(() -> eltype(g)[], pending, q2), kv)
            end
            for kv in rest; push!(get!(() -> eltype(g)[], pending, get(qm, msum(kv), 0)), kv); end
            while !isempty(pending)
                q = minimum(keys(pending)); its = pop!(pending, q)
                sort!(its; by=kv -> kv[2][2] + kv[2][3] + kv[2][4], rev=true)
                for (kv, q2) in _kernels_batch!(ctx.store, ctx.engine, its, ctx.cw, ntasks, q, ctx.cu)
                    push!(get!(() -> eltype(g)[], pending, q2), kv)
                end
            end
        end
        return length(todo)
    end
    sort!(todo; by=kv -> kv[2][2] + kv[2][3] + kv[2][4], rev=true)
    next = Threads.Atomic{Int}(1)
    @sync for _ in 1:ntasks
        Threads.@spawn while true
            i = Threads.atomic_add!(next, 1)
            i > length(todo) && break
            key, (κ, Nu, Nv, Nw, ρu, ρv, ρw) = todo[i]
            _store_get!(() -> kernels(ctx.engine, T, κ, Nu, Nv, Nw, ρu, ρv, ρw, ctx.cw, ctx.cu), ctx.store, key)
        end
    end
    length(todo)
end

"Expansion of (v+w)^A (u+w)^B (u+v)^C as (power p of u, q of v, r of w, coefficient)."
_expand_perimetric(ec::ElementCache{T}, ctx::IntegralContext{T}, A::Int, B::Int, C::Int) where {T} =
    get!(ec.perim, (A, B, C)) do   # the task-local dictionary only references the shared table (no copy; no lock after the first access)
        _store_get!(() -> _expand_perimetric(ctx, A, B, C), ctx.perims, (A, B, C))   # shared across tasks
    end
function _expand_perimetric(ctx::IntegralContext{T}, A::Int, B::Int, C::Int) where {T}
    out = Tuple{Int,Int,Int,T}[]
    bn = ctx.binom
    for i in 0:A, k in 0:B, l in 0:C
        push!(out, (k + l, (A - i) + (C - l), i + (B - k), bn[A+1, i+1] * bn[B+1, k+1] * bn[C+1, l+1]))
    end
    out
end


"""
    element!(ec, ctx, A,B,C,D,J, a,b,c) -> T

Element integral (powers include the volume element); the result is memoized in the task-local `ec`.
"""
function element!(ec::ElementCache{T}, ctx::IntegralContext{T}, A::Int, B::Int, C::Int, D::Int, J::Int,
                  a::Float64, b::Float64, c::Float64) where {T}
    ctx.split && return element_split!(ec, ctx, A, B, C, D, 0, J, a, b, c)
    key = (A, B, C, D, J, _bits(a), _bits(b), _bits(c))
    v = get(ec.elem, key, nothing)
    v === nothing || return v
    length(ec.elem) > ec.cap && empty!(ec.elem)
    v = _store_get(ctx.elems, key)
    v === nothing || return (ec.elem[key] = v)
    v = if isone(ctx.λ)
        _element_tilde(ec, ctx, A, B, C, D, J, a, b, c)
    else
        # ⟨σ^D (ln σ)^J⟩ = λ^D Σ_k C(J,k) (ln λ)^{J−k} ⟨σ̃^D (ln σ̃)^k⟩
        acc = zero(T)
        for k in 0:J
            acc += T(binomial(J, k)) * ctx.lnλ^(J - k) * _element_tilde(ec, ctx, A, B, C, D, k, a, b, c)
        end
        (D >= 0 ? ctx.λ^D : inv(ctx.λ^(-D))) * acc
    end
    ec.elem[key] = _store_put!(ctx.elems, key, v)
end

"""
    elements_upto!(ec, ctx, A,B,C,D, Jmax, a,b,c) -> (v₀, v₁, v₂)

Elements J = 0..Jmax of the same (A,B,C,D; a,b,c) together (entries J > Jmax are 0). The monomial sums S₀, S₁, S₂ and the
kernel lookups are shared across J (bit-identical to `element!`: same summation order and the same formula per J). The
memory (`ec.elem`, `ctx.elems`) is not consulted; all J ≤ Jmax are stored. Not for `split` contexts.
"""
function elements_upto!(ec::ElementCache{T}, ctx::IntegralContext{T}, A::Int, B::Int, C::Int, D::Int, Jmax::Int,
                        a::Float64, b::Float64, c::Float64) where {T}
    ctx.split && throw(ArgumentError("use element_split! in split contexts"))
    t = _element_tilde_upto(ec, ctx, A, B, C, D, Jmax, a, b, c)
    v = if isone(ctx.λ)
        t
    else
        sc = D >= 0 ? ctx.λ^D : inv(ctx.λ^(-D))
        ntuple(3) do J1
            J = J1 - 1
            J > Jmax && return zero(T)
            acc = zero(T)
            for k in 0:J
                acc += T(binomial(J, k)) * ctx.lnλ^(J - k) * t[k+1]
            end
            sc * acc
        end
    end
    for J in 0:Jmax
        key = (A, B, C, D, J, _bits(a), _bits(b), _bits(c))
        length(ec.elem) > ec.cap && empty!(ec.elem)
        ec.elem[key] = _store_put!(ctx.elems, key, v[J+1])
    end
    v
end

"`_element_tilde` for J = 0..Jmax together (shared sums)."
function _element_tilde_upto(ec::ElementCache{T}, ctx::IntegralContext{T}, A, B, C, D, Jmax,
                             a::Float64, b::Float64, c::Float64) where {T}
    (A >= 0 && B >= 0 && C >= 0) || throw(ArgumentError("negative powers ($A,$B,$C)"))
    0 <= Jmax <= 2 || throw(ArgumentError("J=$Jmax"))
    ta, tb_, tc = T(a), T(b), T(c)
    ρu, ρv, ρw = tb_ + tc, ta + tc, ta + tb_
    (ρu > 0 && ρv > 0 && ρw > 0) || throw(ArgumentError("divergent exponents (a,b,c)=($a,$b,$c)"))
    rkey = (_bits(b + c), _bits(a + c), _bits(a + b))
    tmp = zero(T)
    if D < 0
        R = -D
        R < A + B + C + 3 || throw(ArgumentError("not integrable: R=$R, min(p+q+r)=$(A + B + C)"))
        κ = R - 1
        S0 = zero(T); S1 = zero(T); S2 = zero(T)
        for (p, q, r, w) in _monomials!(ec, ctx, A, B, C, 0)
            K = _kernels_cached!(ec, ctx, κ, p + 1, q + 1, r + 1, ρu, ρv, ρw, rkey)
            S0 = _acc2!(S0, tmp, w, K[1])
            Jmax >= 1 && (S1 = _acc2!(S1, tmp, w, K[2]))
            Jmax == 2 && (S2 = _acc2!(S2, tmp, w, K[3]))
        end
        iG = ctx.invΓ[R]; ψ = ctx.ψ[R]
        t0 = 2 * iG * S0
        t1 = Jmax >= 1 ? -2 * iG * (S1 - ψ * S0) : zero(T)
        t2 = Jmax == 2 ? 2 * iG * (S2 - 2ψ * S1 + (ψ * ψ - ctx.ψ1[R]) * S0) : zero(T)
        return (t0, t1, t2)
    end
    SD = _smoment_half(ec, ctx, A, B, C, D, ρu, ρv, ρw, rkey)
    t0 = 2 * SD
    Jmax == 0 && return (t0, zero(T), zero(T))
    K1 = zero(T); K2 = zero(T)
    for (P, Q, R, w) in _monomials!(ec, ctx, A, B, C, D + 1)
        K = _kernels_cached!(ec, ctx, 0, P + 1, Q + 1, R + 1, ρu, ρv, ρw, rkey)
        K1 = _acc2!(K1, tmp, w, K[2])
        Jmax == 2 && (K2 = _acc2!(K2, tmp, w, K[3]))
    end
    γ = ctx.γE
    t1 = 2 * (-K1 - γ * SD)
    t2 = Jmax == 2 ? 2 * (K2 + 2γ * K1 + (γ * γ - ctx.π2o6) * SD) : zero(T)
    (t0, t1, t2)
end

# Element integrals with σ̃ = u + v + c_w w.
#
# (v+w)^A (u+w)^B (u+v)^C σ̃^m is expanded into monomials u^P v^Q w^R (integer coefficients times powers of c_w, exact in
# Int128) and equal monomials are combined before integrating. ∫ u^P v^Q w^R e^{−ρ·} = P!Q!R!/(ρu^{P+1} ρv^{Q+1} ρw^{R+1}), so
#     ⟨σ̃^m⟩ = 2 Σ c_PQR P!Q!R! ρu^{−P−1} ρv^{−Q−1} ρw^{−R−1}, and the kernel sum is Σ c_PQR P!Q!R! K(κ; P+1, Q+1, R+1).
# There are at most (A+B+C+m+1)(A+B+C+m+2)/2 monomials, far fewer than the (A+1)(B+1)(C+1)·(m+1)(m+2)/2 expansion terms.
# BigFloat runs the sums with in-place MPFR operations.

"""
Integer coefficients of (v+w)^A (u+w)^B (u+v)^C σ̃^m: (P, Q, R) → coefficient per power l of c_w (entry l+1), exact.
Checked Int128 arithmetic (OverflowError on overflow), redone in BigInt if it overflows. The coefficients sum to
2^{A+B+C}·3^m, so Int128 is exceeded when A+B+C + 1.585m > 127.
"""
function _monomials_int(A::Int, B::Int, C::Int, m::Int)
    try
        _monomials_int(Int128, A, B, C, m)
    catch e
        e isa OverflowError || rethrow()
        _monomials_int(BigInt, A, B, C, m)
    end
end

function _monomials_int(::Type{I}, A::Int, B::Int, C::Int, m::Int) where {I<:Integer}
    mul(x, y) = I === BigInt ? x * y : Base.checked_mul(x, y)
    add(x, y) = I === BigInt ? x + y : Base.checked_add(x, y)
    # binomials from a Pascal triangle (checked sums, so overflow raises OverflowError); binomial(::Int128, ::Int128) builds
    # BigInts internally and is slow
    nmax = max(A, B, C, m)
    pas = [zeros(I, n + 1) for n in 0:nmax]
    for n in 0:nmax
        pas[n+1][1] = I(1); pas[n+1][n+1] = I(1)
        for k in 1:n-1; pas[n+1][k+1] = add(pas[n][k], pas[n][k+1]); end
    end
    bin(n, k) = @inbounds pas[n+1][k+1]
    base = Dict{NTuple{3,Int},I}()
    for i in 0:A, k in 0:B, l in 0:C
        key = (k + l, (A - i) + (C - l), i + (B - k))
        base[key] = add(get(base, key, I(0)), mul(mul(bin(A, i), bin(B, k)), bin(C, l)))
    end
    # σ̃^m = Σ m!/(i!k!l!) u^i v^k (c_w w)^l, accumulated as integers per power of c_w
    acc = Dict{NTuple{3,Int},Vector{I}}()
    for ((p, q, r), c) in base, i in 0:m, k in 0:m-i
        l = m - i - k
        v = get!(() -> zeros(I, m + 1), acc, (p + i, q + k, r + l))
        v[l+1] = add(v[l+1], mul(c, mul(bin(m, i), bin(m - i, k))))
    end
    acc
end

"""
List of monomials (P, Q, R, c_PQR·P!Q!R!) of (v+w)^A (u+w)^B (u+v)^C σ̃^m, shared across tasks (`ctx.monos`) with the task-local `ec.mono` in front.
"""
_monomials!(ec::ElementCache{T}, ctx::IntegralContext{T}, A::Int, B::Int, C::Int, m::Int) where {T} =
    get!(ec.mono, (A, B, C, m)) do
        _store_get!(ctx.monos, (A, B, C, m)) do
            isone(ctx.cu) || return _monomials_cu(ctx, A, B, C, m)
            acc = _monomials_int(A, B, C, m)
            out = Tuple{Int,Int,Int,T}[]
            for (P, Q, R) in sort!(collect(keys(acc)))
                v = acc[(P, Q, R)]
                c = zero(T); cwl = one(T)
                for l in 0:m
                    iszero(v[l+1]) || (c += T(v[l+1]) * cwl)
                    cwl *= ctx.cw
                end
                push!(out, (P, Q, R, c * ctx.fact[P+1] * ctx.fact[Q+1] * ctx.fact[R+1]))
            end
            out
        end
    end
# dry runs use the real monomial lists too (`ctx.monos` is shared with the recording context and reused by the elements)

# monomials for w₂ ≠ 1 (c_u ≠ 1): σ̃^m = Σ m!/(i!k!l!) (c_u u)^i v^k (c_w w)^l, integer coefficients per (power i of c_u, power l of c_w) in BigInt.
function _monomials_cu(ctx::IntegralContext{T}, A::Int, B::Int, C::Int, m::Int) where {T}
    bin(n, k) = binomial(BigInt(n), BigInt(k))
    base = Dict{NTuple{3,Int},BigInt}()
    for i in 0:A, k in 0:B, l in 0:C
        key = (k + l, (A - i) + (C - l), i + (B - k))
        base[key] = get(base, key, BigInt(0)) + bin(A, i) * bin(B, k) * bin(C, l)
    end
    acc = Dict{NTuple{3,Int},Dict{NTuple{2,Int},BigInt}}()
    for ((p, q, r), c) in base, i in 0:m, k in 0:m-i
        l = m - i - k
        d = get!(() -> Dict{NTuple{2,Int},BigInt}(), acc, (p + i, q + k, r + l))
        d[(i, l)] = get(d, (i, l), BigInt(0)) + c * bin(m, i) * bin(m - i, k)
    end
    pu = [ctx.cu^i for i in 0:m]; pw = [ctx.cw^l for l in 0:m]
    out = Tuple{Int,Int,Int,T}[]
    for (P, Q, R) in sort!(collect(keys(acc)))
        c = zero(T)
        for ((i, l), v) in sort!(collect(acc[(P, Q, R)]))
            iszero(v) || (c += T(v) * pu[i+1] * pw[l+1])
        end
        push!(out, (P, Q, R, c * ctx.fact[P+1] * ctx.fact[Q+1] * ctx.fact[R+1]))
    end
    out
end

"Powers of 1/ρ (entry k+1 is ρ^{−k}) up to n (task-local memo, extended when needed)."
function _invpowers!(ec::ElementCache{T}, rkey, ρu::T, ρv::T, ρw::T, n::Int) where {T}
    pw = get(ec.ipow, rkey, nothing)
    if pw === nothing || length(pw[1]) < n + 1
        mk(ρ) = (x = inv(ρ); v = Vector{T}(undef, n + 1); v[1] = one(T); for k in 1:n; v[k+1] = v[k] * x; end; v)
        pw = ec.ipow[rkey] = (mk(ρu), mk(ρv), mk(ρw))
    end
    pw
end

# acc += x*y*z (BigFloat uses tmp, no allocation; two product roundings and one sum rounding, same order for both types)
@inline _acc3!(acc::BigFloat, tmp::BigFloat, x::BigFloat, y::BigFloat, z::BigFloat) =
    (mpfr_mul!(tmp, x, y); mpfr_mul!(tmp, tmp, z); mpfr_add!(acc, acc, tmp); acc)
@inline _acc3!(acc, tmp, x, y, z) = acc + x * y * z
@inline _acc2!(acc::BigFloat, tmp::BigFloat, x::BigFloat, y::BigFloat) = (mpfr_mul!(tmp, x, y); mpfr_add!(acc, acc, tmp); acc)
@inline _acc2!(acc, tmp, x, y) = acc + x * y

"⟨σ̃^m⟩/2 = Σ c_PQR P!Q!R! ρu^{−P−1} ρv^{−Q−1} ρw^{−R−1}."
function _smoment_half(ec::ElementCache{T}, ctx::IntegralContext{T}, A, B, C, m, ρu::T, ρv::T, ρw::T, rkey) where {T}
    mono = _monomials!(ec, ctx, A, B, C, m)
    iu, iv, iw = _invpowers!(ec, rkey, ρu, ρv, ρw, A + B + C + m + 1)
    s = zero(T); tmp = zero(T); x = zero(T)
    @inbounds for (P, Q, R, c) in mono
        x = _acc2!(zero!(x), tmp, c, iu[P+2])          # x = c ρu^{−P−1}
        s = _acc3!(s, tmp, x, iv[Q+2], iw[R+2])
    end
    s
end
zero!(x::BigFloat) = (ccall((:mpfr_set_zero, :libmpfr), Cvoid, (Ref{BigFloat}, Cint), x, 1); x)
zero!(x) = zero(x)

function _element_tilde(ec::ElementCache{T}, ctx::IntegralContext{T}, A, B, C, D, J,
                        a::Float64, b::Float64, c::Float64) where {T}
    (A >= 0 && B >= 0 && C >= 0) || throw(ArgumentError("negative powers ($A,$B,$C)"))
    0 <= J <= 2 || throw(ArgumentError("J=$J"))
    ta, tb_, tc = T(a), T(b), T(c)
    ρu, ρv, ρw = tb_ + tc, ta + tc, ta + tb_
    (ρu > 0 && ρv > 0 && ρw > 0) || throw(ArgumentError("divergent exponents (a,b,c)=($a,$b,$c)"))
    rkey = (_bits(b + c), _bits(a + c), _bits(a + b))

    J == 0 && D >= 0 && return 2 * _smoment_half(ec, ctx, A, B, C, D, ρu, ρv, ρw, rkey)
    tmp = zero(T)
    if D < 0
        R = -D
        R < A + B + C + 3 || throw(ArgumentError("not integrable: R=$R, min(p+q+r)=$(A + B + C)"))
        κ = R - 1
        S0 = zero(T); S1 = zero(T); S2 = zero(T)
        for (p, q, r, w) in _monomials!(ec, ctx, A, B, C, 0)
            K = _kernels_cached!(ec, ctx, κ, p + 1, q + 1, r + 1, ρu, ρv, ρw, rkey)
            S0 = _acc2!(S0, tmp, w, K[1])
            J >= 1 && (S1 = _acc2!(S1, tmp, w, K[2]))
            J == 2 && (S2 = _acc2!(S2, tmp, w, K[3]))
        end
        iG = ctx.invΓ[R]; ψ = ctx.ψ[R]
        J == 0 && return 2 * iG * S0
        J == 1 && return -2 * iG * (S1 - ψ * S0)
        ψ1 = ctx.ψ1[R]
        return 2 * iG * (S2 - 2ψ * S1 + (ψ * ψ - ψ1) * S0)
    end
    # D ≥ 0, J ∈ {1,2}:
    #   ⟨σ̃^D ln σ̃⟩  = −K₁ − γ S_D,   ⟨σ̃^D ln²σ̃⟩ = K₂ + 2γK₁ + (γ² − π²/6) S_D
    #   K_j = Σ ... ∫ (ln t)^j N_{D+1}(t) dt (N_m(t) = ⟨σ̃^m e^{−tσ̃}⟩ expanded in monomials)
    SD = _smoment_half(ec, ctx, A, B, C, D, ρu, ρv, ρw, rkey)
    K1 = zero(T); K2 = zero(T)
    for (P, Q, R, w) in _monomials!(ec, ctx, A, B, C, D + 1)
        K = _kernels_cached!(ec, ctx, 0, P + 1, Q + 1, R + 1, ρu, ρv, ρw, rkey)
        K1 = _acc2!(K1, tmp, w, K[2])
        J == 2 && (K2 = _acc2!(K2, tmp, w, K[3]))
    end
    γ = ctx.γE
    J == 1 && return 2 * (-K1 - γ * SD)
    2 * (K2 + 2γ * K1 + (γ * γ - ctx.π2o6) * SD)
end

# =============================================================================
# split: r₁^A r₂^B r₁₂^C s^D σ^{−R} (ln σ)^J,  s = r₁+r₂,  σ = s + βσ r₁₂
# =============================================================================
#
# In perimetric coordinates with w' = 2w: s = u+v+w', σ = s φ(x), exponent = s ε(x), with (u,v,w') = s(x₁,x₂,x),
# φ(x) = 1 + βσ(1−x). With x₁ = (1−x)y, x₂ = (1−x)(1−y) the exponent coefficient is
#     ε(x,y) = (1−x)[(b+c) y + (a+c)(1−y)] + (a+b) x/2
# (r₁ = v+w, r₂ = u+w, r₁₂ = u+v). With the radial integral in closed form, each term u^p v^q w^r gives
#     2^{−r} ∫₀¹ dx x^r (1−x)^{p+q+1} φ^{−R} ∫₀¹ dy y^p (1−y)^q Σ_k C(J,k) (ln φ)^{J−k} N! ε^{−N−1} L_k(ε),
#     N = p+q+r+D−R+2,  L₀ = 1,  L₁ = ψ(N+1) − ln ε,  L₂ = L₁² + ψ₁(N+1).
# For a = b, ε does not depend on y and the y integral is p!q!/(p+q+1)!. For a ≠ b, y is integrated by Gauss–Legendre too
# (ε is linear and positive in y, its zero outside the interval) on a separate (x, y) grid of nab nodes (default 80); the cost
# grows with the square of the node count. The accuracy is set by the exponent ratio κ = (b+c)/(a+c), so the grid is chosen
# by it (_abgrid: 80 for ratio ≤ 4, 128 for ≤ 10, 37√κ nodes above for 1e-63). The x integral is Gauss–Legendre on [0,1]
# (the logarithmic singularity of φ at x = 1 + 1/βσ is outside).

"""
    element_split!(ec, ctx, A,B,C,D,R,J, a,b,c) -> T

Split element integral (powers include the volume element). a ≠ b needs a two-dimensional (x, y) quadrature and costs more than a = b.
"""
function element_split!(ec::ElementCache{T}, ctx::IntegralContext{T}, A::Int, B::Int, C::Int, D::Int,
                        R::Int, J::Int, a::Float64, b::Float64, c::Float64) where {T}
    key = (A, B, C, D, J + 4R, _bits(a), _bits(b), _bits(c))
    v = get(ec.elem, key, nothing)
    v === nothing || return v
    length(ec.elem) > ec.cap && empty!(ec.elem)
    v = _store_get(ctx.elems, key)
    v === nothing || return (ec.elem[key] = v)
    (A >= 0 && B >= 0 && C >= 0) || throw(ArgumentError("negative powers ($A,$B,$C)"))
    (0 <= J <= 2 && 0 <= R <= 2) || throw(ArgumentError("(J,R)=($J,$R)"))
    (a > 0 && b > 0 && a + c > 0 && b + c > 0) || throw(ArgumentError("divergent exponents (a,b,c)=($a,$b,$c)"))
    a == b || return (ec.elem[key] = _store_put!(ctx.elems, key, _element_split_ab(ec, ctx, A, B, C, D, R, J, a, b, c)))
    iszero(c) && _STU_PATH[] && return (ec.elem[key] = _store_put!(ctx.elems, key, _element_split_stu(ec, ctx, A, B, C, D, R, J, a)))
    ta = T(a); tc = T(c)
    acc = zero(T)
    # c = 0 (He type): ln a and G_N = N!/a^{N+1} do not depend on the term, so log is taken once and G follows the recurrence
    # G_N = G_{N−1}·N/a (instead of log and exp per term of the perimetric expansion).
    lnε = iszero(c) ? log(ta) : zero(T); ia = iszero(c) ? inv(ta) : zero(T); Gs = T[]
    for (p, q, r, cf) in _expand_perimetric(ec, ctx, A, B, C)
        N = p + q + r + D - R + 2
        N >= 0 || throw(ArgumentError("not integrable: N=$N (A,B,C,D,R=$A,$B,$C,$D,$R)"))
        P = p + q + 1
        pref = ldexp(cf * ctx.fact[p+1] * ctx.fact[q+1] / ctx.fact[P+1], -r)
        if iszero(c)
            # ε = a is constant: angular part Ω(r,P,R,i) = ∫ x^r (1−x)^P φ^{−R} (ln φ)^i
            while length(Gs) <= N
                k = length(Gs); push!(Gs, k == 0 ? ia : Gs[end] * k * ia)
            end
            G = Gs[N+1]
            L1 = ctx.ψ[N+1] - lnε
            s = _ang!(ec, ctx, r, P, R, J)
            J >= 1 && (s += J * L1 * _ang!(ec, ctx, r, P, R, J - 1))
            J == 2 && (s += (L1 * L1 + ctx.ψ1[N+1]) * _ang!(ec, ctx, r, P, R, 0))
            acc += pref * G * s
        else
            acc += pref * _ang_eps!(ec, ctx, r, P, R, J, N, ta, tc, a, c)
        end
    end
    ec.elem[key] = _store_put!(ctx.elems, key, acc)
end

"Use the (s,t,u) path for a = b, c = 0 elements (default true); false uses the perimetric expansion (for checks)."
const _STU_PATH = Ref(true)

# a = b, c = 0 (He type, e^{−a s}): in (s,t,u) coordinates (s = r₁+r₂, t = r₁−r₂, u = r₁₂, dr₁dr₂dr₁₂ = ds dt du / 2,
# 0 ≤ |t| ≤ u ≤ s), with r₁^A r₂^B = 2^{−A−B} Σ_k c_k s^{A+B−k} t^k, ∫_{−u}^{u} t^k dt = 2u^{k+1}/(k+1) (even k), u = s y,
# σ = s φ(y), φ = 1 + βσ y, the element is 2^{−A−B} N!/a^{N+1} [S_J + J L₁ S_{J−1} + (for J = 2) (L₁² + ψ₁(N+1)) S₀],
# N = A+B+C+D+2−R, L₁ = ψ(N+1) − ln a, with
#   S_i(A,B,C,R) = Σ_{k even} c_k/(k+1) Ω(C+k+1, R, i), Ω(m,R,i) = ∫₀¹ y^m φ^{−R} (ln φ)^i dy.
# S_i is positive, but the c_k alternate in sign and the sum cancels (several digits lost for A = B = 20 at 256 bits), so S_i
# is computed at working precision + 64 bits and rounded. S_i does not depend on D, J, a and is tabulated: an element is one to
# three table lookups and a few multiplications (instead of (A+1)(B+1)(C+1) perimetric terms).
function _element_split_stu(ec::ElementCache{T}, ctx::IntegralContext{T}, A::Int, B::Int, C::Int, D::Int, R::Int, J::Int, a::Float64) where {T}
    N = A + B + C + D + 2 - R
    N >= 0 || throw(ArgumentError("not integrable: N=$N (A,B,C,D,R=$A,$B,$C,$D,$R)"))
    length(ec.stuS) > ec.cap && empty!(ec.stuS)
    Si(i) = get!(() -> _stu_S(ec, ctx, A, B, C, R, i), ec.stuS, (A, B, C, R, i))
    ta = T(a); lnε = log(ta)
    G = ldexp(exp(ctx.lnfact[N+1] - (N + 1) * lnε), -(A + B))
    L1 = ctx.ψ[N+1] - lnε
    s = Si(J)
    J >= 1 && (s += J * L1 * Si(J - 1))
    J == 2 && (s += (L1 * L1 + ctx.ψ1[N+1]) * Si(0))
    G * s
end

function _stu_S(ec::ElementCache{T}, ctx::IntegralContext{T}, A::Int, B::Int, C::Int, R::Int, i::Int) where {T}
    wb = T === BigFloat ? precision(BigFloat) : round(Int, -log2(Float64(eps(T))))
    hb = wb + 64
    v = setprecision(BigFloat, hb) do
        y, w, φ, lnφ = get!(ec.stuleg, hb) do
            βb = BigFloat(_tobig(ctx.βs))
            xs, ws = gauss_legendre01(BigFloat, length(ctx.xleg))
            (xs, ws, [1 + βb * x for x in xs], [log(1 + βb * x) for x in xs])
        end
        ck = get!(ec.stuc, (A, B)) do
            [BigFloat(sum(binomial(big(A), k - j) * binomial(big(B), j) * (isodd(j) ? -1 : 1) for j in max(0, k - A):min(k, B))) / (k + 1)
             for k in 0:2:(A + B)]
        end
        acc = BigFloat(0)
        for (q, c) in enumerate(ck)
            iszero(c) && continue
            m = C + 2(q - 1) + 1
            Ω = get!(ec.stuΩ, (m, R, i)) do
                t = BigFloat(0)
                for n in eachindex(y)
                    f = w[n] * y[n]^m
                    R > 0 && (f /= φ[n]^R)
                    i > 0 && (f *= lnφ[n]^i)
                    t += f
                end
                t
            end
            acc += c * Ω
        end
        acc
    end
    T === BigFloat ? BigFloat(v, RoundNearest; precision=wb) : T(v)
end

function _element_split_ab(ec::ElementCache{T}, ctx::IntegralContext{T}, A::Int, B::Int, C::Int, D::Int,
                           R::Int, J::Int, a::Float64, b::Float64, c::Float64) where {T}
    g = _abgrid(ctx, a, b, c)
    acc = zero(T)
    terms = Dict{NTuple{3,Int},T}()                 # combine equal (p,q,r) terms before integrating
    for (p, q, r, cf) in _expand_perimetric(ec, ctx, A, B, C)
        terms[(p, q, r)] = get(terms, (p, q, r), zero(T)) + cf
    end
    length(ec.term) > ec.cap && empty!(ec.term)    # per-task memo
    for ((p, q, r), cf) in terms
        N = p + q + r + D - R + 2
        N >= 0 || throw(ArgumentError("not integrable: N=$N (A,B,C,D,R=$A,$B,$C,$D,$R)"))
        s = get!(ec.term, (p, q, r, N, R, J, _bits(a), _bits(b), _bits(c))) do   # shared across elements
            P = p + q + 1
            Y = _angy!(ec, g, ctx, p, q, N, a, b, c)
            W0 = _wvec!(ec, g, r, P, R, 0)
            if J == 0
                _dot(W0, Y[1])
            elseif J == 1
                _dot(_wvec!(ec, g, r, P, R, 1), Y[1]) + _dot(W0, Y[2])
            else
                _dot(_wvec!(ec, g, r, P, R, 2), Y[1]) + 2 * _dot(_wvec!(ec, g, r, P, R, 1), Y[2]) + _dot(W0, Y[3])
            end
        end
        acc += cf * s / T(2)^r
    end
    acc
end

# The y integral ∫₀¹ y^p (1−y)^q ε^{−ν} (ψ − ln ε)^k dy (ε = E₀ + E₁ y > 0, ν = N+1) becomes, by the projective substitution
#     t = F y / (E₀ + E₁ y),  F = E₀ + E₁,  D(t) = F − E₁ t
# E₀^{p+1−ν} F^{q+1−ν} ∫₀¹ t^p (1−t)^q D^m (ψ − ln E₀ − ln F + ln D)^k dt (m = ν − p − q − 2).
# The pole of order ν of ε^{−ν} turns into D^m (m = r + D − R + 1, a small integer), a polynomial for k = 0 and m ≥ 0. The
# remaining singularity is the zero t = F/E₁ of D (outside the interval, at a distance of order (a+c)/(b−a)), closer for larger exponent ratios.
"Weights w x^r (1−x)^P φ^{−R} (ln φ)^e at the x nodes (powers are not recomputed per term)."
_wvec!(ec::ElementCache{T}, g::ABGrid{T}, r::Int, P::Int, R::Int, e::Int) where {T} =
    get!(ec.wvec, (r, P, R, e, g.n)) do
        [g.w[i] * g.x[i]^r * (1 - g.x[i])^P / g.φ[i]^R * g.lnφ[i]^e for i in eachindex(g.x)]
    end

_make_abgrid(::Type{T}, m::Int, βb) where {T} = let (xa, wa) = gauss_legendre01(T, m)
    ABGrid{T}(m, xa, wa, [T(_big(() -> 1 + βb * (1 - _tobig(x)))) for x in xa],
              [T(_big(() -> log(1 + βb * (1 - _tobig(x))))) for x in xa])
end

"""
Node count from the exponent ratio κ (`_split_kappa`: largest ÷ smallest corner value a+c, b+c, (a+b)/2 of ε(x,y)) for the
default nab = 80: 80 for κ ≤ 4, 128 for κ ≤ 10, otherwise 37√κ rounded up to a multiple of 16 (κ = 64: 304, 256: 592,
800: 1056). Compared with the σ path, the relative error has about 2 n/√κ − 10 digits (κ = 10–800, n = 192–768), so 37√κ
reaches 1e-63. The cost grows with the square of the node count; changing nab scales all counts by nab/80.
"""
# Ratios of the corner values of ε(x,y) = (1−x)[(b+c) y + (a+c)(1−y)] + (a+b) x/2 (linear in x and y, so extremes are at
# corners). The y direction ((b+c)/(a+c)) is reduced to a polynomial by the projective substitution, but the x direction
# (a+c, b+c at x = 0 and (a+b)/2 at x = 1) is integrated by Gauss–Legendre, so both ratios set the node count (the y ratio
# alone was not enough for γ < 0 with small α+γ). The a = b path (`_ang_eps!`) has the two corners a+c and a.
_split_kappa(a::Float64, b::Float64, c::Float64) = (e = (a + c, b + c, (a + b) / 2); maximum(e) / minimum(e))
# ratio in y (x fixed, ends in y, largest at x = 0) and ratio in x (y fixed, ends in x)
_kappa_y(a::Float64, b::Float64, c::Float64) = max(a + c, b + c) / min(a + c, b + c)
_kappa_x(a::Float64, b::Float64, c::Float64) = (h = (a + b) / 2; max(max(h, a + c) / min(h, a + c), max(h, b + c) / min(h, b + c)))

"""
Node count of the (x,y) grid for the y ratio κy, x ratio κx and working precision `digits` (`_abgrid` rule).
y: digits ≈ 1.74 n/√κy − 10; x: digits ≈ 1.50–1.62 n/√κx − 10 (estimated with 1.45). For c = 0, κx < 2 and x does not matter.
"""
function _split_nodes(κy::Float64, κx::Float64, nab::Int, digits::Float64)
    n = κy <= 4 ? nab : κy <= 10 ? round(Int, 1.6nab) :
        max(round(Int, 1.6nab), 16 * ceil(Int, nab / 80 * 37 * sqrt(κy) / 16))
    # nodes reaching the working precision: 37√κ above was measured at Float64x4 (63 digits); at 384 bits digits ≈ 1.74–2.0 n/√κ − 10.
    # The smaller 1.74 gives the lower bound on n for working precision + 10 digits.
    n = max(n, 16 * ceil(Int, (digits + 10) * sqrt(κy) / 1.74 / 16))
    max(n, 16 * ceil(Int, (digits + 10) * sqrt(κx) / 1.45 / 16))
end
_split_nodes(a::Float64, b::Float64, c::Float64, nab::Int, digits::Float64) =
    a == b ? _split_nodes(1.0, max(a + c, a) / min(a + c, a), nab, digits) : _split_nodes(_kappa_y(a, b, c), _kappa_x(a, b, c), nab, digits)

function _abgrid(ctx::IntegralContext{T}, a::Float64, b::Float64, c::Float64) where {T}
    _abgrid_n(ctx, _split_nodes(a, b, c, ctx.nab, -log10(Float64(eps(T)))), _split_kappa(a, b, c))
end

function _abgrid_n(ctx::IntegralContext{T}, n::Int, κ::Float64) where {T}
    g = lock(() -> get(ctx.abgrids, n, nothing), ctx.ablock)
    g === nothing || return g
    n > 1024 && @warn "split exponent ratio $(round(κ; digits=1)): grid of $(n) nodes ($(n)×$(n) for a ≠ b; expensive; accuracy from 37√κ nodes for 1e-63)" maxlog = 1
    g = _make_abgrid(T, n, _big(() -> BigFloat(ctx.βs)))   # built outside the lock (expensive); concurrent builds give the same content
    lock(() -> get!(ctx.abgrids, n, g), ctx.ablock)
end

function _dot(u::Vector{T}, v::Vector{T}) where {T}
    s = zero(T)
    @inbounds for i in eachindex(u)
        s += u[i] * v[i]
    end
    s
end

"E₀, F at the x nodes and D, ln D on the (x, t) grid (once per exponent set)."
function _epsgrid!(ec::ElementCache{T}, g::ABGrid{T}, ctx::IntegralContext{T}, a::Float64, b::Float64, c::Float64) where {T}
    _store_get!(ctx.epsgs, (_bits(a), _bits(b), _bits(c))) do
        ta, tb, tc = T(a), T(b), T(c)
        n = g.n
        lnE0 = Vector{T}(undef, n); lnF = Vector{T}(undef, n)
        Dm = Matrix{T}(undef, n, n); lnD = Matrix{T}(undef, n, n)
        for i in 1:n
            x = g.x[i]
            E0 = (1 - x) * (ta + tc) + (ta + tb) * x / 2
            E1 = (1 - x) * (tb - ta)
            F = E0 + E1
            lnE0[i] = log(E0); lnF[i] = log(F)
            for j in 1:n
                D = F - E1 * g.x[j]
                Dm[i, j] = D; lnD[i, j] = log(D)
            end
        end
        (lnE0, lnF, Dm, lnD)
    end
end

"Y_k(x) = ∫ dy y^p (1−y)^q N! ε^{−N−1} L_k(ε) at the x nodes (k = 0,1,2 together)."
function _angy!(ec::ElementCache{T}, g::ABGrid{T}, ctx::IntegralContext{T}, p::Int, q::Int, N::Int,
                a::Float64, b::Float64, c::Float64) where {T}
    _store_get!(ctx.angys, (p, q, N, _bits(a), _bits(b), _bits(c))) do
        lnE0, lnF, _, lnD = _epsgrid!(ec, g, ctx, a, b, c)
        n = g.n
        ν = N + 1; m = ν - p - q - 2
        G = _dpow!(ec, g, ctx, a, b, c, m)
        wt = get!(() -> [g.w[j] * g.x[j]^p * (1 - g.x[j])^q for j in 1:n], ec.wty, (p, q, n))
        # Σ_j wt D^m (ln D)^k for k = 0,1,2, recombined as a polynomial in L₁ = c₀ + ln D (column major, i innermost)
        S0 = zeros(T, n); S1 = zeros(T, n); S2 = zeros(T, n)
        @inbounds for j in 1:n
            w = wt[j]
            for i in 1:n
                f = w * G[i, j]; l = lnD[i, j]; fl = f * l
                S0[i] += f; S1[i] += fl; S2[i] += fl * l
            end
        end
        ψ = ctx.ψ[N+1]; ψ1 = ctx.ψ1[N+1]; lnf = ctx.lnfact[N+1]
        Y0 = Vector{T}(undef, n); Y1 = Vector{T}(undef, n); Y2 = Vector{T}(undef, n)
        @inbounds for i in 1:n
            c0 = ψ - lnE0[i] - lnF[i]
            pre = exp(lnf + (p + 1 - ν) * lnE0[i] + (q + 1 - ν) * lnF[i])
            Y0[i] = pre * S0[i]
            Y1[i] = pre * (c0 * S0[i] + S1[i])
            Y2[i] = pre * ((c0 * c0 + ψ1) * S0[i] + 2 * c0 * S1[i] + S2[i])
        end
        (Y0, Y1, Y2)
    end
end

"D^m on the (x, t) grid (once per exponent set and m)."
_dpow!(ec::ElementCache{T}, g::ABGrid{T}, ctx::IntegralContext{T}, a::Float64, b::Float64, c::Float64, m::Int) where {T} =
    _store_get!(ctx.dpows, (m, _bits(a), _bits(b), _bits(c))) do
        D = _epsgrid!(ec, g, ctx, a, b, c)[3]
        m >= 0 ? D .^ m : inv.(D) .^ (-m)
    end

function _ang!(ec::ElementCache{T}, ctx::IntegralContext{T}, r::Int, P::Int, R::Int, i::Int) where {T}
    key = (r, P, R, i, -1, 0, 0, 0)
    v = get(ec.ang, key, nothing)
    v === nothing || return v
    s = zero(T)
    @inbounds for n in eachindex(ctx.xleg)
        x = ctx.xleg[n]
        f = ctx.wleg[n] * x^r * (1 - x)^P
        R > 0 && (f /= ctx.φleg[n]^R)
        i > 0 && (f *= ctx.lnφleg[n]^i)
        s += f
    end
    ec.ang[key] = s
end

function _ang_eps!(ec::ElementCache{T}, ctx::IntegralContext{T}, r::Int, P::Int, R::Int, J::Int, N::Int,
                   ta::T, tc::T, a::Float64, c::Float64) where {T}
    key = (r, P, R, J, N, 1, _bits(a), _bits(c))
    v = get(ec.ang, key, nothing)
    v === nothing || return v
    # x nodes: if the corner ratio κ = max(a+c, a)/min(a+c, a) of ε = a + c(1−x) needs more than the default nleg, use the a ≠ b grid rule
    nx = _split_nodes(a, a, c, ctx.nab, -log10(Float64(eps(T))))
    g = nx <= length(ctx.xleg) ? nothing : _abgrid_n(ctx, nx, max(a + c, a) / min(a + c, a))
    xs, ws, φs, lnφs = g === nothing ? (ctx.xleg, ctx.wleg, ctx.φleg, ctx.lnφleg) : (g.x, g.w, g.φ, g.lnφ)
    s = zero(T); ψ = ctx.ψ[N+1]; ψ1 = ctx.ψ1[N+1]
    @inbounds for n in eachindex(xs)
        x = xs[n]
        ε = ta + tc * (1 - x); lnε = log(ε)
        f = ws[n] * x^r * (1 - x)^P * exp(ctx.lnfact[N+1] - (N + 1) * lnε)
        R > 0 && (f /= φs[n]^R)
        L1 = ψ - lnε; lφ = lnφs[n]
        g = J == 0 ? one(T) : J == 1 ? lφ + L1 : lφ * lφ + 2 * L1 * lφ + L1 * L1 + ψ1
        s += f * g
    end
    ec.ang[key] = s
end

include("gaussr.jl")          # element integrals with the r₁₂ Gaussian

end # module Integrals
