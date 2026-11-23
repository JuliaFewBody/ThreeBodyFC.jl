"""
    ThreeBodyFC

High-precision variational calculations of three-body Coulomb systems by the free complement (FC) method.
"""
module ThreeBodyFC

using LinearAlgebra
using MultiFloats
using Printf
using HighPrecisionLinearAlgebra
using HighPrecisionLinearAlgebra: _el     # element pointers of packed storage (inner assembly loops)

include("integrals.jl")          # lowest layer: element integrals (submodule independent of the FC method)
using .Integrals
using .Integrals: _big, _tobig, _abgrid, _split_nodes, _recording_context, _record_element!, _store_get   # internal parts used by assembly and the driver
import .Integrals: describe      # each layer adds methods to this generic describe

include("angular.jl")          # angular weights for L ≥ 1 (submodule)
using .Angular
include("system.jl")
include("space.jl")
include("hamiltonian.jl")
include("generate.jl")
include("partialwave.jl")      # exchange-mirror pairs for L ≥ 1 by partial waves (used by assemble.jl)
include("assemble.jl")
include("symmetry.jl")
include("dependence.jl")
include("solvers.jl")
include("subsystems.jl")
include("order.jl")
include("driver.jl")
include("autoexp.jl")
include("output.jl")
include("observables.jl")
include("bethe.jl")
include("qed.jl")             # leading mα⁵ QED for general three-body systems (uses the current weights of bethe.jl)

export ThreeBody, PowerTerm, ExpTerm, coulomb, exact
export ExponentSets, gaussian_r12, LogVariable, NewLog, binary_exponents, binary_warning
export FCRule, KernelQuadrature
export NoSymmetry, Exchange
export CholeskyInverseIteration, CanonicalCut, PivotedSubset
export run_fc, count_basis, optimize_alpha, optimize_exponents, auto_exponents, auto_w2, w2_rule, set_rule, auto_form, auto_basis, auto_gaussr, late_greedy, thresholds, pair_scales, energy_error, hermiticity, default_exponents, default_symmetry, fmt, compare, load_results, load_wavefunction
export expectation, DistancePower, TotalEnergy, KineticEnergy, PotentialEnergy, Delta, Cusp, variance, temple, distribution
export expectation_quad, P4, LaplacianProduct, EnergyMinusVSquared, Retardation, InverseSquare, ArakiSucherPart, DrachmanDelta12, DrachmanDelta, KineticQuad, QTerm, PairAxis, LDot, LTensor, Uehling,
       relativistic_correction, araki_sucher, qed_correction, breit_pauli, fine_structure, spin_hamiltonian, quad_type
export short_range_sets, bethe_intermediate, bethe_spectrum, bethe_log, bethe_logarithm, qed_leading
export Float64x2, Float64x3, Float64x4

end
