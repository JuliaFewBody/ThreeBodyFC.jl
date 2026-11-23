# Leading mα⁵ (R∞α³) QED for general three-body systems, assembled as in Pachucki–Patkóš–Yerokhin arXiv:2512.04623: two-body
# formulas for arbitrary masses per pair, the Bethe logarithm with the current of all particles, no further three-body terms.
# The Q term is 1810.07929 Eq. 14 (with γ; reproduces Salpeter's constant for hydrogen). No AMM terms (they cancel in the sum).

"""
    qed_leading(sys, wf; bethe, kinds=:auto, α=7.2973525693e-3, vp=:auto, N=50, logvar=LogVariable(), T=nothing) -> NamedTuple

Leading QED energy E (a.u., including α³) and its parts for three-body Coulomb systems of any masses, charges and L.
- `bethe`: Bethe logarithm of the current of all particles (R∞ units), a number or the result of `bethe_logarithm`; per particle as
  `[(β=β_μ, weights=(1, 0)), (β=β_e, weights=(0, 1))]` (D from the corresponding components of the full-current weights; current cross terms dropped).
- `kinds`: per particle `:lepton` (full self-energy) or `:nucleus` (logarithmic part only; the rest goes into the nuclear radius).
  `:auto` treats infinite masses or masses > 1000 as nuclei.
- Per pair (a,b) (q = zₐz_b, mₐ, m_b, a.u.):
  - identical leptons (same mass, charge, kind; e–e in He): (q²/mₐm_b)[(14/3)ln α + 164/15]⟨δ⟩ − (14/3)(q²/mₐm_b)Q (2512 Eqs. 11–12),
    with ⟨δ⟩ in Drachman's global form (the FC wave function does not satisfy the cusp of identical pairs exactly).
  - otherwise: two-photon exchange (q²/mₐm_b)[(2/3)(−ln α + 31/3) + c_ab]⟨δ⟩ − (14/3)(q²/mₐm_b)Q with
    c_ab = −2/(mₐ² − m_b²)[mₐ² ln m_b − m_b² ln mₐ] (point spin-1/2 leptons, 2512 Eq. 3 with reference mass 1); −2 ln m_lepton for
    lepton–nucleus (heavy-nucleus limit).
    Local self-energy per particle k: (z_k²/m_k²)(−q)(4/3)(ln(m_k/α²) + 5/6)⟨δ⟩ (lepton), (4/3)ln(m_k/α²) (nucleus).
    Vacuum polarization per lepton species l (mass m_l): point form (−q)(−4/15)/m_l²⟨δ⟩, but the exact `Uehling` (electron loop
    only) when μ_ab|q|α/m_l > 0.1 (muon–nucleus pairs with the electron loop). `vp = :point` or `:uehling` forces one form.
- Bethe logarithm: −(2/3π)·D·β, D = −2π Σ Δw² q⟨δ⟩ (current of all particles, as `:closed` of `bethe_log`).
Returns (E, E_over_α3, parts, D, β, kinds). For He (infinite mass) the same formula as `qed_correction`.
"""
function qed_leading(sys::ThreeBody, wf; bethe, kinds=:auto, α=big"7.2973525693e-3", vp::Symbol=:auto, N::Int=50,
                     logvar::LogVariable=LogVariable(), T::Union{Type,Nothing}=nothing)
    a = BigFloat(α); piB = BigFloat(Base.MathConstants.pi); lna = log(a)
    z = _charges_from_coulomb(sys, BigFloat)
    m = [isinf_mass(x) ? BigFloat(Inf) : exact(BigFloat, x) for x in sys.masses]
    kd = kinds === :auto ? [(isinf(mk) || mk > 1000) ? :nucleus : :lepton for mk in m] : collect(Symbol, kinds)
    pairs = ((1, 3), (2, 3), (1, 2))
    q = _coulomb_charges(sys, BigFloat)
    δ = expectation(sys, wf, [Delta(p...) for p in pairs]; logvar=logvar)
    inv(x) = isinf(x) ? zero(BigFloat) : 1 / x
    μp(i, j) = 1 / (inv(m[i]) + inv(m[j]))
    loops = unique([m[k] for k in 1:3 if kd[k] === :lepton])
    # quantities that need quadrature (Q and the exact Uehling), together
    need = Any[]; idxQ = Dict{Int,Int}(); idxU = Dict{Int,Int}(); idxD = Dict{Int,Int}()
    ident(p) = (i = pairs[p][1]; j = pairs[p][2]; kd[i] === :lepton && kd[j] === :lepton && m[i] == m[j] && z[i] == z[j])
    useU(p, ml) = vp === :uehling ? (ml == 1) : vp === :point ? false : (ml == 1 && !isinf(μp(pairs[p]...)) && μp(pairs[p]...) * abs(q[p]) * a / ml > big"0.1")
    for p in 1:3
        iszero(q[p]) && continue
        i, j = pairs[p]
        isinf(m[i]) && isinf(m[j]) && continue
        (iszero(inv(m[i]) * inv(m[j])) && !ident(p)) || (push!(need, QTerm(i, j)); idxQ[p] = length(need))
        ident(p) && (push!(need, DrachmanDelta(i, j)); idxD[p] = length(need); continue)   # identical pairs: global form (the cusp is not exact for FC wave functions)
        any(ml -> useU(p, ml), loops) && (push!(need, Uehling(i, j; inv_alpha=string(1 / a))); idxU[p] = length(need))
    end
    vals = isempty(need) ? BigFloat[] : expectation_quad(sys, wf, need; N=N, logvar=logvar, T=T)
    parts = Dict{String,BigFloat}()
    add!(k, v) = (parts[k] = get(parts, k, zero(BigFloat)) + v)
    for p in 1:3
        iszero(q[p]) && continue
        i, j = pairs[p]; lab = "($i,$j)"
        f = inv(m[i]) * inv(m[j]) * q[p]^2
        if ident(p)
            add!("two-photon exchange and self-energy $lab (identical)", f * (big"14" / 3 * lna + big"164" / 15) * vals[idxD[p]])
            add!("Q term $lab", -big"14" / 3 * f * vals[idxQ[p]])
            continue
        end
        if !iszero(f)
            # mass logarithms of 2512 Eq. 3 with reference mass 1 (electron, log scale m_e α²): −2/(mₐ² − m_b²)·[mₐ² ln m_b − m_b² ln mₐ];
            # lepton–nucleus in the heavy-nucleus limit: −2 ln m_lepton
            c = zero(BigFloat)
            if kd[i] === :lepton && kd[j] === :lepton
                c = m[i] == m[j] ? -2log(m[i]) + 1 : -2 / (m[i]^2 - m[j]^2) * (m[i]^2 * log(m[j]) - m[j]^2 * log(m[i]))
            elseif kd[i] === :lepton || kd[j] === :lepton
                c = -2log(kd[i] === :lepton ? m[i] : m[j])
            end
            add!("two-photon exchange $lab", f * (2 * (-lna + big"31" / 3) / 3 + c) * δ[p])
            add!("Q term $lab", -big"14" / 3 * f * vals[idxQ[p]])
        end
        for k in (i, j)
            isinf(m[k]) && continue
            g = z[k]^2 / m[k]^2 * (-q[p]) * δ[p]
            add!("self-energy particle $k $lab", kd[k] === :lepton ? big"4" / 3 * (log(m[k] / a^2) + big"5" / 6) * g : big"4" / 3 * log(m[k] / a^2) * g)
        end
        for ml in loops
            if useU(p, ml)
                parts["vacuum polarization (Uehling, exact) $lab"] = vals[idxU[p]] / a^3      # in units of E/α³
            else
                add!("vacuum polarization (point, loop m=$(Float64(ml))) $lab", (-q[p]) * (-big"4" / 15) / ml^2 * δ[p])
            end
        end
    end
    w = _full_current_weights(sys, BigFloat)
    Dof(sel) = (v = (w[1] * sel[1], w[2] * sel[2]); Δw = (v[1], v[2], v[1] - v[2]); -2piB * sum(Δw[p]^2 * q[p] * δ[p] for p in 1:3))
    if bethe isa AbstractVector              # Bethe logarithm per particle (dropping current cross terms; for muonic systems with separated scales)
        D = zero(BigFloat); β = zero(BigFloat)
        for b in bethe
            Dk = Dof(b.weights); D += Dk; β += Dk * BigFloat(b.β)
            parts["Bethe logarithm weights=$(b.weights)"] = -2 / (3piB) * Dk * BigFloat(b.β)
        end
        β /= D
    else
        β = bethe isa Number ? BigFloat(bethe) : BigFloat(bethe.β)
        D = Dof((1, 1))
        parts["Bethe logarithm"] = -2 / (3piB) * D * β
    end
    Eα3 = sum(values(parts))
    (E=Eα3 * a^3, E_over_α3=Eα3, parts=parts, D=D, β=β, kinds=kd, delta=δ,
     Q=Dict(pairs[p] => vals[i] for (p, i) in idxQ), delta_drachman=Dict(pairs[p] => vals[i] for (p, i) in idxD))
end
