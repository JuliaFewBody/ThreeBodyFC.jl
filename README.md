# ThreeBodyFC.jl

[![Julia 1.12+](https://badgen.net/static/Julia/1.12%2B/007ec6?icon=https%3A%2F%2Fraw.githubusercontent.com%2FJuliaLang%2Fjulia-logo-graphics%2Fmaster%2Fimages%2Fjulia-dots.svg)](https://julialang.org/downloads/)
[![CI](https://github.com/JuliaFewBody/ThreeBodyFC.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaFewBody/ThreeBodyFC.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24542861.svg)](https://doi.org/10.5281/zenodo.24542861)

High-precision variational energies of three-body Coulomb systems (He, H⁻, Ps⁻, muonic molecules, …) by the free complement (FC)
method. Given masses and charges, the basis is generated stage by stage and solved in extended precision (`Float64x4` or
`BigFloat`); the precision is raised automatically when the basis becomes nearly dependent. The dense linear algebra uses
[HighPrecisionLinearAlgebra.jl](https://github.com/JuliaFewBody/HighPrecisionLinearAlgebra.jl).

## Quick Start

Requires [Julia](https://julialang.org/install/) 1.12 or later.

```julia
import Pkg; Pkg.add(url="https://github.com/JuliaFewBody/ThreeBodyFC.jl.git")
using ThreeBodyFC
he = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))   # masses (m₁, m₂, m₃) and charges (q₁, q₂, q₃)
r = run_fc(he; stages=1:5, reference="-2.90372437703411959831115924519440444669690537")
display(r)
```

Start Julia with threads (`julia -t 16`).

## Main functions

| | |
|---|---|
| Systems | `ThreeBody`, `coulomb`, `PowerTerm`, `ExpTerm` |
| Bases | `ExponentSets`, `LogVariable`, `NewLog`, `gaussian_r12`, `FCRule`, `Exchange`, `NoSymmetry` |
| Solving | `run_fc` (also `L ≥ 1`, `nstates` for excited states), `count_basis` |
| Choosing exponents | `auto_basis`, `auto_exponents`, `auto_form`, `set_rule`, `late_greedy`, `optimize_exponents` |
| Results | `load_results`, `load_wavefunction`, `compare` |
| Expectation values | `expectation`, `expectation_quad`, `variance`, `temple`, `distribution` |

Relativistic, fine-structure and QED corrections and the Bethe logarithm are in `ThreeBodyFC.Experimental` (not yet optimized; the interface may change).

## Development

```shell
git clone https://github.com/JuliaFewBody/ThreeBodyFC.jl.git
cd ThreeBodyFC.jl
julia --project=. --startup-file=no -e "using Pkg; Pkg.test()"
```

## Acknowledgments

This package is built on an initial project template generated using [Julia Package Factory](https://github.com/JuliaPackageFactory).
