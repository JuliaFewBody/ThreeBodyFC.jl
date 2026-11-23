# Element integrals with the r₁₂ Gaussian e^{−η r₁₂²} (included inside Integrals).
#
#   I(A,B,C,D; a,c,η) = ∫ r₁^A r₂^B r₁₂^C s^D e^{−a(r₁ + r₂) − c r₁₂ − η r₁₂²} dr₁ dr₂ dr₁₂,   s = r₁ + r₂
#
# The K_{l,m,n} integrals of Hijikata–Nakashima–Nakatsuji, JCP 130, 024102 (2009), App. A, extended to the negative powers
# of s that the FC generation needs (from g = r₁r₂/s + r₁₂). a is common to r₁ and r₂ (α = β), no logarithm, σ = s.
# With s = r₁ + r₂, t = r₁ − r₂ (dr₁dr₂ = ds dt/2, |t| ≤ R ≤ s, R = r₁₂) the t integral is polynomial:
#   I = 2^{−A−B−1} Σ_{i,k} C(A,i) C(B,k) (−1)^k [m even] 2/(m+1) Q(C+m+1, D+A+B−m),   m = i + k
#   Q(p, q) = ∫₀^∞ R^p e^{−cR−ηR²} G_q(R) dR,   G_q(R) = ∫_R^∞ s^q e^{−as} ds
# G_q goes upward for q ≥ 0, G_q = (R^q e^{−aR} + q G_{q−1})/a, and for q < 0 downward from G_{−1} = E₁(aR) (MPFR eint),
# G_q = (R^{q+1} e^{−aR} − a G_{q+1})/(−q−1), with extra digits against the cancellation at large aR.
# The R integral uses the half-line DE rule (R = exp(τ − e^{−τ})), halving the step until the whole table is stable at
# the working precision. One table per (a, c, η) covers a range of p, q; η = 0 (c > 0) also works and checks `element!`.

"Table of Q(p, q) for one (a, c, η): Q[p − pmin + 1, q − qmin + 1]."
struct GaussTable{T}
    pmin::Int
    pmax::Int
    qmin::Int
    qmax::Int
    Q::Matrix{T}
    h::Float64        # converged step (diagnostic)
end

# E₁(x) = −Ei(−x)
_E1(x::BigFloat) = (r = BigFloat(); ccall((:mpfr_eint, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Int32), r, -x,
                                          Base.MPFR.ROUNDING_MODE[]); -r)

# add the contribution of one node to Q (w = DE weight × step)
function _gauss_node!(Q::Matrix{BigFloat}, R::BigFloat, w::BigFloat, a::BigFloat, c::BigFloat, η::BigFloat,
                      pmin::Int, pmax::Int, qmin::Int, qmax::Int, G::Vector{BigFloat}, Pw::Vector{BigFloat})
    ea = exp(-a * R)
    # G_q, q = qmin..qmax
    o = 1 - qmin                       # G[q + o]
    if qmax >= 0
        g = ea / a
        0 >= qmin && (G[o] = g)
        Rq = one(BigFloat)
        for q in 1:qmax
            Rq *= R
            g = (Rq * ea + q * g) / a
            q >= qmin && (G[q+o] = g)
        end
    end
    if qmin < 0
        g = _E1(a * R)
        -1 <= qmax && (G[-1+o] = g)
        Rq = one(BigFloat)               # R^{q+1}; R^{−1} at q = −2
        iR = inv(R)
        for q in -2:-1:qmin
            Rq *= iR
            g = (Rq * ea - a * g) / (-q - 1)
            q <= qmax && (G[q+o] = g)
        end
    end
    f = w * exp(-(c + η * R) * R)
    # R^p, p = pmin..pmax
    po = 1 - pmin
    x = f * (pmin >= 0 ? R^pmin : inv(R)^(-pmin))
    for p in pmin:pmax
        Pw[p+po] = x
        x *= R
    end
    @inbounds for j in qmin:qmax, i in pmin:pmax
        Q[i+po, j+o] += Pw[i+po] * G[j+o]
    end
    Q
end

"""
    gauss_table(T, a, c, η, (pmin, pmax), (qmin, qmax)) -> GaussTable{T}

Table Q(p, q) = ∫₀^∞ R^p e^{−cR−ηR²} ∫_R^∞ s^q e^{−as} ds dR (η > 0, or η = 0 with c > 0), at the precision of T plus margin.
"""
function gauss_table(::Type{T}, a::Float64, c::Float64, η::Float64, pr::NTuple{2,Int}, qr::NTuple{2,Int}) where {T}
    (pmin, pmax), (qmin, qmax) = pr, qr
    a > 0 || throw(ArgumentError("a > 0 required (a=$a)"))
    (η > 0 || c > 0) || throw(ArgumentError("η > 0 or c > 0 required (c=$c, η=$η)"))
    Pt = T === BigFloat ? precision(BigFloat) : 4 * 53
    prec = Pt + 64 + 6 * max(0, -qmin)            # margin for the downward recurrence (about log₂(aR) bits per step)
    setprecision(BigFloat, prec) do
        ab, cb, ηb = BigFloat(a), BigFloat(c), BigFloat(η)
        L = prec * log(2.0) + 60
        # upper end: R where the integrand R^{p+q} e^{−(a+c)R − ηR²} has fallen by e^{−L} from its peak
        Rmax = if η > 0
            Rpk = max(0.0, -(a + c) / (2η))
            Rpk + sqrt((L + 4 * (pmax + max(qmax, 0)) + 10) / η) + 1
        else
            (L + 4 * (pmax + max(qmax, 0)) * log(L)) / (a + c) + 10
        end
        τlo = -log(L) - 0.5; τhi = log(Rmax) + 0.5
        np = pmax - pmin + 1; nq = qmax - qmin + 1
        # integrable at R → 0? G_q(0) is finite for q ≥ 0, −ln R for q = −1, R^{q+1}/(−q−1) for q ≤ −2
        conv(i, j) = (p = pmin + i - 1; q = qmin + j - 1; q >= -1 ? p >= 0 : p + q >= -1)
        G = Vector{BigFloat}(undef, nq); Pw = Vector{BigFloat}(undef, np)
        node(τ) = (eτ = exp(-BigFloat(τ)); R = exp(BigFloat(τ) - eτ); (R, (1 + eτ) * R))
        # trapezoidal rule with step h; halving the step adds only the odd nodes
        h = 1 / 8
        S = zeros(BigFloat, np, nq)
        for τ in τlo:h:τhi
            R, dR = node(τ); _gauss_node!(S, R, dR, ab, cb, ηb, pmin, pmax, qmin, qmax, G, Pw)
        end
        Q = S * BigFloat(h)
        tol = BigFloat(2)^(-(Pt + 8))
        for _ in 1:12
            h /= 2
            Snew = zeros(BigFloat, np, nq)
            for τ in (τlo+h):2h:τhi
                R, dR = node(τ); _gauss_node!(Snew, R, dR, ab, cb, ηb, pmin, pmax, qmin, qmax, G, Pw)
            end
            Qn = Q / 2 + Snew * BigFloat(h)
            δ = maximum(abs(Qn[i, j] - Q[i, j]) / abs(Qn[i, j]) for i in 1:np, j in 1:nq if conv(i, j))
            Q = Qn
            if δ < tol
                Qt = T.(Q)
                for i in 1:np, j in 1:nq
                    conv(i, j) || (Qt[i, j] = T(NaN))      # divergent at R → 0 (NaN propagates if used)
                end
                return GaussTable{T}(pmin, pmax, qmin, qmax, Qt, h)
            end
        end
        error("gauss_table: no convergence down to step $h (a=$a, c=$c, η=$η, p=$pr, q=$qr)")
    end
end

"I(A,B,C,D) (A, B ≥ 0) from table `tab`; `binom` is the table of C(n,k) (`ctx.binom`)."
function element_gauss(tab::GaussTable{T}, binom::Matrix{T}, A::Int, B::Int, C::Int, D::Int) where {T}
    (A >= 0 && B >= 0) || throw(ArgumentError("element_gauss: A, B ≥ 0 required (A=$A, B=$B)"))
    s = zero(T)
    for i in 0:A, k in 0:B
        m = i + k
        isodd(m) && continue
        p = C + m + 1; q = D + A + B - m
        (tab.pmin <= p <= tab.pmax && tab.qmin <= q <= tab.qmax) ||
            throw(ArgumentError("element_gauss: outside the table (p=$p, q=$q; table p=$(tab.pmin):$(tab.pmax), q=$(tab.qmin):$(tab.qmax))"))
        t = binom[A+1, i+1] * binom[B+1, k+1] * tab.Q[p-tab.pmin+1, q-tab.qmin+1] * 2 / (m + 1)
        s += isodd(k) ? -t : t
    end
    s / T(2)^(A + B + 1)
end

"Range ((pmin, pmax), (qmin, qmax)) of the table needed for I(A,B,C,D)."
gauss_range(A::Int, B::Int, C::Int, D::Int) = ((C + 1, C + A + B + 1), (D, D + A + B))
