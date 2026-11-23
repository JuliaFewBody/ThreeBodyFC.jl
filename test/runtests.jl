using Test
using ThreeBodyFC
using ThreeBodyFC: IntegralContext, ElementCache, element!, KernelQuadrature, LogVariable, HamiltonianData, assemble, Key

# On CI (single thread, time limit) only the fast test sets run; set THREEBODYFC_FULL_TESTS=true to run everything.
const QUICK = get(ENV, "CI", "") == "true" && get(ENV, "THREEBODYFC_FULL_TESTS", "") != "true"
const QUICKSET = Set(["element integral identity σ = r₁ + r₂", "exact Float64 coefficients", "kernel quadrature (BigFloat step, range, refinement) and kernel prefetch", "assembly and solve with packed BigFloat (SymPacked)", "exchange symmetrization in packed storage (from representative rows only)", "normalization while filling (dscale) is bit-identical to diagonal_normalize", "angular weights for L ≥ 1 (Angular.angular_weight)", "run_fc for L ≥ 1 (He 2³P, 2¹P, infinite nuclear mass)", "mirror pairs for L ≥ 1 by partial waves give the same E as the monomial path", "coalescence expansion with non-binary exponents (exact κ)", "unnatural parity (He 2p² ³Pᵉ, unnatural=true)", "integer monomial coefficients (BigInt beyond Int128)", "excited states do not pick up lower states (pseudo-states above the Ps⁻ threshold)", "integral constants follow the working precision", "Gaussian in r₁₂ (η)", "quadrature working type from the coefficient magnitudes", "Bethe logarithm stops on non-binary exponents of ψ₀ or intermediate states", "rational dmin slope (lower bound −⌈q n⌉ at stage n)", "drop_exact keeps the space for βσ ≠ 0 (expansion in σ)", "σ = r₁ + w₂ r₂ (LogVariable(; w2)): symmetry of H and unchanged default", "auto_w2 (choose w₂ by the stage energy)", "w2_rule (rule of thumb for w₂)", "kernel quadrature peak (_kq_range bisection + window) equals a full scan", "local order _local_order", "localization _peak_power", "gauss_legendre01 terminates above 1086 bits"])
macro ts(name, body)
    esc(:(if !QUICK || $name in QUICKSET
              @testset $name $body
          end))
end


const T = Float64x4

@ts "element integral identity σ = r₁ + r₂" begin
    ctx = IntegralContext(T, KernelQuadrature(), LogVariable())
    ec = ElementCache{T}()
    for (A, B, C) in ((1, 1, 1), (2, 1, 3)), J in 0:2, D in (0, 1, -1), (a, b, c) in ((1.0, 0.5, 0.25), (2.0, 2.0, 0.0))
        A + B + C + D >= 0 || continue
        lhs = element!(ec, ctx, A + 1, B, C, D - 1, J, a, b, c) + element!(ec, ctx, A, B + 1, C, D - 1, J, a, b, c)
        rhs = element!(ec, ctx, A, B, C, D, J, a, b, c)
        @test abs(lhs - rhs) <= 1e-55 * abs(rhs)
    end
end

@ts "ppμ n=1 (21 digits against an independent implementation)" begin
    sys = ThreeBody(("9.8802444013385381166477", "9.8802444013385381166477", "1.1126095132977722858792"), coulomb(1, 1, -1))
    es = ExponentSets([0.5, 0.75, 1.0, 1.5], [0.5, 0.75, 1.0, 1.5], zeros(4))
    r = run_fc(sys, es; stages=1:1, io=devnull)
    @test r[1].M == 68
    @test fmt(r[1].E, 21) == "-0.541008056906455948767"
end

@ts "exchange symmetrization, output, restart, comparison" begin
    sys = ThreeBody(("9.8802444013385381166477", "9.8802444013385381166477", "1.1126095132977722858792"), coulomb(1, 1, -1))
    es = ExponentSets([0.5, 0.75, 1.0, 1.5], [0.5, 0.75, 1.0, 1.5], zeros(4))
    od = mktempdir()
    r = run_fc(sys, es; stages=1:1, symmetry=Exchange(+1), outdir=od, io=devnull)
    @test r[1].M == 68 && r[1].Msolved == 52
    @test fmt(r[1].E, 21) == "-0.541008056906455948767"
    r2 = run_fc(sys, es; stages=1:1, symmetry=Exchange(+1), outdir=od, io=devnull)
    @test fmt(r2[1].E, 30) == fmt(r[1].E, 30)
    c = compare(od, Dict(1 => "-0.541008056906455948767"); io=devnull)
    @test c[1].digits == 21
    wf = load_wavefunction(od, 1)
    @test length(wf.c) == length(wf.basis) == 68
    @test_throws ArgumentError run_fc(ThreeBody(("1", "2", "Inf"), coulomb(-1, -1, 2)), es; stages=1:1,
                                      symmetry=Exchange(+1), io=devnull)
    @test count_basis(sys, es; stages=1:3, io=devnull)[3].M == 844
end

@ts "exact Float64 coefficients" begin
    x = 20.391307867500078
    @test BigFloat(ThreeBodyFC.exact(BigFloat, ExpTerm(x, (0, 0, 0), (1.0, 0, 0)).c)) == BigFloat(x)
end

@ts "collect=:st (rule of JCP 127, 224104) reproduces M_n of Table IV log1" begin
    sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
    es = ExponentSets([1.7], [1.7], [0.0])
    rule = FCRule(collect=:st, gcoef=["1/2", "1"], dedup=false)
    c = count_basis(sys, es; stages=1:5, rule=rule, symmetry=Exchange(+1), io=devnull)
    @test [x.M for x in c] == [13, 52, 125, 246, 427]
    r = run_fc(sys, ExponentSets([1.55], [1.55], [0.0]); stages=1:2, rule=rule, symmetry=Exchange(+1), io=devnull)
    @test [x.Msolved for x in r] == [10, 34]
    r0 = run_fc(sys, ExponentSets([1.55], [1.55], [0.0]); stages=0:1, rule=rule, symmetry=Exchange(+1), io=devnull)
    @test [x.n for x in r0] == [0, 1] && [x.Msolved for x in r0] == [2, 10]
    @test r0[2].basis == r[1].basis && abs(r0[2].E - r[1].E) < 1e-50
    @test energy_error(r[1].E, "-2.9") == Float64(ThreeBodyFC._tobig(r[1].E) - parse(BigFloat, "-2.9"))
    @test_logs (:warn, r"16 digits") match_mode=:any run_fc(sys, ExponentSets([1.55], [1.55], [0.0]); stages=1:1, rule=rule,
                                                         symmetry=Exchange(+1), io=devnull, reference=-2.9)
    rp = run_fc(sys, ExponentSets([1.55], [1.55], [0.0]); stages=2:2, rule=rule, symmetry=Exchange(+1), io=devnull, progress=0.01)
    @test abs(rp[end].E - r[2].E) < 1e-50
    @test_throws ArgumentError run_fc(sys, ExponentSets([1.5], [1.5], [-1.6]); stages=1:1, rule=rule, io=devnull)
    @test hermiticity(sys, ExponentSets([1.4], [0.9], [0.2]); logvar=LogVariable(0.37; split=true)) < 1e-58
    @test default_symmetry(sys) isa Exchange && default_exponents(sys).α == fill(2.0, 2) && default_exponents(sys).β == [2.0, 0.5]
    @test default_symmetry(ThreeBody(("1", "2", "Inf"), coulomb(-1, -1, 2))) isa NoSymmetry
    rd = run_fc(sys; stages=1:1, io=devnull)
    @test count_basis(sys; stages=1:1, io=devnull)[1].M == rd[1].M
    @test rd[1].E < -2.89
    @test startswith(sprint(show, r[1]), "StageResult(n=1, M=") && occursin("solved", sprint(show, MIME"text/plain"(), r)) && occursin("ΔE", sprint(show, MIME"text/plain"(), r))
    @test_throws ArgumentError count_basis(sys, es; stages=1:1, rule=rule, logvar=LogVariable(1), io=devnull)
    a = run_fc(sys; stages=2:2, io=devnull); b = run_fc(sys; stages=2:2, io=devnull, drop_exact=false)
    @test a[1].nexact == 2 == b[1].solution.ndropped && a[1].solution.method == :cholesky && a[1].Msolved == b[1].Msolved
    @test abs(a[1].E - b[1].E) < 1e-55
    @test_logs (:warn, r"working precision") match_mode=:any run_fc(sys; stages=1:2, T=Float64, io=devnull)
end

@ts "new logarithmic form ln(s+βu) (LogVariable(split=true))" begin
    using ThreeBodyFC: element_split!
    for β in (1.0, 0.37)
        c0 = IntegralContext(T, KernelQuadrature(), LogVariable(β))
        c1 = IntegralContext(T, KernelQuadrature(), LogVariable(β; split=true))
        e0 = ElementCache{T}(); e1 = ElementCache{T}()
        for (A, B, C) in ((1, 1, 1), (5, 1, 4)), (a, c) in ((4.4, 0.0), (3.0, 0.7)), R in 0:2, J in 0:2
            x = element_split!(e1, c1, A, B, C, 0, R, J, a, a, c)
            y = element!(e0, c0, A, B, C, -R, J, a, a, c)
            @test abs(x - y) <= 1e-55 * abs(y)
        end
    end
    sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
    rule = FCRule(collect=:st, gcoef=["1/2", "1"], dedup=false)
    old = Dict(1 => ("1.475", "-2.9035368129845081753397218458810096020875379078796197226321"),
               2 => ("1.627", "-2.9037240073207848533297054592437102398122774635446758591651"),
               3 => ("1.679", "-2.9037243750941475527692353426463487689574712752928418695738"))
    Esplit(α, n) = run_fc(sys, ExponentSets([α], [α], [0.0]); stages=n:n, rule=rule,
                          logvar=LogVariable(1.0; split=true), symmetry=Exchange(+1), io=devnull)[end]
    big_(x) = BigFloat(ThreeBodyFC._tobig(x))
    for n in 1:3
        s, Eold = old[n]
        α = parse(Float64, s); h = 2.0^-20
        r = Esplit(α, n)
        @test r.Msolved == (10, 34, 77)[n]
        slope = (big_(Esplit(α + h, n).E) - big_(Esplit(α - h, n).E)) / 2h
        E10 = big_(r.E) + slope * (parse(BigFloat, s) - BigFloat(α))
        @test abs(E10 - parse(BigFloat, Eold)) < 1e-30
    end
    let sys = ThreeBody(("1", "1", "1"), coulomb(-1, -1, 1)), β = 0.37, es = ExponentSets([0.6], [0.6], [0.3])
        lvs = (LogVariable(β), LogVariable(β; split=true))
        HS(lv, basis; kw...) = assemble(HamiltonianData(sys, lv, T), IntegralContext(T, KernelQuadrature(), lv),
                                        copy(es), basis; kw...)
        b0 = Key[(A, B, C, 0, j, 1, 0) for A in 0:2 for B in 0:2 for C in 0:2 for j in 0:1 if A + B + C <= 3]
        (H0, S0), (H1, S1) = HS(lvs[1], b0), HS(lvs[2], b0)
        @test maximum(abs.(H0 - H1)) < 1e-58 * maximum(abs.(H0)) && maximum(abs.(S0 - S1)) < 1e-58 * maximum(abs.(S0))
        b1 = Key[(A, B, C, d, j, 1, 0) for A in 0:2 for B in 0:2 for C in 0:2 for d in -1:2 for j in 0:1
                 if A + B + C <= 3 && A + B + C + d >= 0]
        M = length(b1)
        H = vcat(HS(lvs[2], b1; rows=1:M-1)[1], HS(lvs[2], b1; rows=M:M)[1])
        @test maximum(abs.(H - transpose(H))) < 1e-58 * maximum(abs.(H))
    end
    o = optimize_alpha(sys, 1; bracket=(1.4, 1.55), rule=rule, logvar=LogVariable(1.0; split=true),
                       symmetry=Exchange(+1), io=devnull)
    @test round(o.α; digits=3) == 1.475 && o.E < parse(BigFloat, "-2.90353681228153")
    o2 = optimize_exponents(sys, 1, [1.5]; make_es=x -> ExponentSets([x[1]], [x[1]], [0.0]), tol=1e-5, io=devnull,
                            rule=rule, logvar=LogVariable(1.0; split=true), symmetry=Exchange(+1))
    @test abs(o2.x[1] - o.α) < 2e-4 && abs(o2.E - o.E) < 1e-11
    o3 = optimize_exponents(sys, 1, [1.5, 1.0]; make_es=x -> (ExponentSets([x[1]], [x[1]], [0.0]), (; logvar=LogVariable(x[2]; split=true))),
                            feasible=x -> x[2] <= 3, rule=rule, symmetry=Exchange(+1), io=devnull)
    @test o3.E < o.E + 1e-11 && all(e -> e[1][2] <= 3, o3.evals)
    @test NewLog() == LogVariable(1.0; split=true) && NewLog(0.5).βσ == 0.5
    a = run_fc(sys, ExponentSets([1.6], [1.6], [0.0]); stages=2:2, rule=rule, symmetry=Exchange(+1), io=devnull)
    b = run_fc(sys, ExponentSets([1.6], [1.6], [0.0]); stages=2:2, rule=rule, logvar=LogVariable(0.0; split=true),
               symmetry=Exchange(+1), io=devnull)
    @test abs(a[end].E - b[end].E) < 1e-55
    let c0 = IntegralContext(T, KernelQuadrature(), LogVariable(0.37)),
        c1 = IntegralContext(T, KernelQuadrature(), LogVariable(0.37; split=true))
        e0 = ElementCache{T}(); e1 = ElementCache{T}()
        for (A, B, C) in ((1, 1, 1), (5, 1, 4)), (a, b, c) in ((1.4, 0.6, 0.0), (1.2, 0.8, 0.3), (2.0, 0.25, 0.0), (5.0, 0.25, 0.0)), R in 0:2, J in 0:2
            x = element_split!(e1, c1, A, B, C, 0, R, J, a, b, c)
            y = element!(e0, c0, A, B, C, -R, J, a, b, c)
            @test abs(x - y) <= 1e-55 * abs(y)
        end
    end
    let c0 = IntegralContext(T, KernelQuadrature(), LogVariable(1.0)), c1 = IntegralContext(T, KernelQuadrature(), NewLog())
        e0 = ElementCache{T}(); e1 = ElementCache{T}()
        for (a, b, c) in ((1.8652, 2.0, -1.8339), (1.8652, 1.8652, -1.8339), (0.0156, 0.0156, 2.0), (0.0156, 0.03, 2.0)),
            (A, B, C) in ((0, 0, 0), (5, 1, 4)), R in 0:2, J in 0:1
            x = element_split!(e1, c1, A, B, C, 0, R, J, a, b, c)
            y = element!(e0, c0, A, B, C, -R, J, a, b, c)
            @test abs(x - y) <= 1e-55 * abs(y)
        end
        @test ThreeBodyFC._split_nodes(2 * 0.9325962066650391, 2.0, -2 * 0.9325942993164062, 80, 63.0) > 4096
        @test ThreeBodyFC._split_nodes(2 * 0.9325962066650391, 2.0, -2 * 0.4662971496582031, 80, 63.0) <= 128
        @test ThreeBodyFC._split_nodes(804.0, 2.0, -1.6, 80, 63.0) <= 4096
    end
    let I = ThreeBodyFC.Integrals
        vals(stu) = (I._STU_PATH[] = stu; e = ElementCache{T}(); c1 = IntegralContext(T, KernelQuadrature(), NewLog());
                     v = [element_split!(e, c1, A, B, C, D, R, J, 3.8, 3.8, 0.0) for (A, B) in ((3, 2), (20, 20), (40, 0), (30, 12))
                          for C in (0, 9) for D in (-1, 2) for R in 0:2 for J in 0:2]; I._STU_PATH[] = true; v)
        x = vals(true); y = vals(false)
        @test maximum(abs.((x .- y) ./ y)) < 1e-58
    end
    es2 = ExponentSets([1.9, 1.1], [1.1, 1.9], [0.0, 0.0])
    a = run_fc(sys, es2; stages=2:2, rule=rule, symmetry=Exchange(+1), io=devnull)
    b = run_fc(sys, es2; stages=2:2, rule=rule, logvar=LogVariable(0.0; split=true), symmetry=Exchange(+1), io=devnull)
    @test abs(a[end].E - b[end].E) < 1e-55
    let sys = ThreeBody(("1", "1", "1"), coulomb(-1, -1, 1)), lv = LogVariable(0.37; split=true)
        b1 = Key[(A, B, C, d, j, 1, 0) for A in 0:1 for B in 0:1 for C in 0:1 for d in -1:1 for j in 0:1 if A + B + C + d >= 0]
        M = length(b1)
        HS(rows) = assemble(HamiltonianData(sys, lv, T), IntegralContext(T, KernelQuadrature(), lv),
                            ExponentSets([0.9], [0.4], [0.2]), b1; rows=rows)[1]
        H = vcat(HS(1:M-1), HS(M:M))
        @test maximum(abs.(H - transpose(H))) < 1e-58 * maximum(abs.(H))
    end
    let μ = 0.125, lv = LogVariable(0.37; split=true)
        sys = ThreeBody(("1", "1", "Inf"), [ExpTerm("-2", (-1, 0, 0), (μ, 0, 0)), ExpTerm("-2", (0, -1, 0), (0, μ, 0)),
                                            ExpTerm("1", (0, 0, -1), (0, 0, μ))])
        b1 = Key[(A, B, C, d, j, 1, 0) for A in 0:1 for B in 0:1 for C in 0:1 for d in -1:1 for j in 0:1 if A + B + C + d >= 0]
        M = length(b1)
        HS(rows) = assemble(HamiltonianData(sys, lv, T), IntegralContext(T, KernelQuadrature(), lv),
                            ExponentSets([2.0], [1.5], [0.25]), b1; rows=rows)[1]
        H = vcat(HS(1:M-1), HS(M:M))
        @test maximum(abs.(H - transpose(H))) < 1e-58 * maximum(abs.(H))
    end
end

@ts "kernel quadrature (BigFloat step, range, refinement) and kernel prefetch" begin
    setprecision(BigFloat, 256) do
        e = KernelQuadrature(BigFloat)
        e50 = KernelQuadrature(BigFloat; h=2π^2 / (256log(2) + 50))
        ref = KernelQuadrature(BigFloat; h=e.h / 8, slo=e.slo - 60, shi=e.shi + 60, cut=e.cut + 100, maxlevel=0)
        for (κ, Nu, Nv, Nw) in ((0, 1, 1, 1), (0, 0, 2, 0), (5, 2, 3, 4), (1, 15, 15, 10), (20, 8, 9, 10)),
            (ρu, ρv, ρw) in ((1.0, 1.0, 2.0), (0.3, 402.6, 402.1))
            a = (BigFloat(ρu), BigFloat(ρv), BigFloat(ρw), big"2.0")
            r = ThreeBodyFC.kernels(ref, BigFloat, κ, Nu, Nv, Nw, a...)
            for eng in (e, e50)
                k = ThreeBodyFC.kernels(eng, BigFloat, κ, Nu, Nv, Nw, a...)
                @test all(abs(k[j] - r[j]) <= 1000 * eps(BigFloat) * (abs(r[j]) + r[1]) for j in 1:3)
            end
        end
        @test e50.nrefined[] > 0
    end
    setprecision(BigFloat, 256) do
        c1 = IntegralContext(BigFloat, KernelQuadrature(BigFloat); βσ=0.3, nmax=40)
        c2 = IntegralContext(BigFloat, KernelQuadrature(BigFloat); βσ=0.3, nmax=40)
        @test all(begin
                      v = ThreeBodyFC.elements_upto!(ElementCache{BigFloat}(), c1, A, B, 0, D, 2, 1.3, 0.7, 0.2)
                      all(v[J+1] == element!(ElementCache{BigFloat}(), c2, A, B, 0, D, J, 1.3, 0.7, 0.2) for J in 0:2)
                  end for A in 0:1, B in 0:1, D in (-2, 1))
    end
    sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2)); lv = LogVariable()
    hd = HamiltonianData(sys, lv, T); es = ExponentSets([1.8, 1.8], [1.8, 0.45], [0.0, 0.0])
    basis = Key[(A, B, C, 0, j, k, 0) for A in 0:1, B in 0:1, C in 0:1, j in 0:1, k in 1:2][:]
    H1, S1 = assemble(hd, IntegralContext(T, KernelQuadrature(), lv), es, basis; ntasks=3, prefetch=false)
    H2, S2 = assemble(hd, IntegralContext(T, KernelQuadrature(), lv), es, basis; ntasks=3, prefetch=true)
    @test H1 == H2 && S1 == S2
    cs = IntegralContext(T, KernelQuadrature(), lv; kernel_cap=50)
    H5, S5 = assemble(hd, cs, es, basis; ntasks=3, prefetch=true)
    @test H5 == H2 && S5 == S2 && cs.store.count[] > 50
    H6, S6 = assemble(hd, cs, es, basis; ntasks=3, prefetch=true)
    @test H6 == H2 && S6 == S2
    setprecision(BigFloat, 256) do
        hb = HamiltonianData(sys, lv, BigFloat)
        H3, S3 = assemble(hb, IntegralContext(BigFloat, KernelQuadrature(BigFloat), lv), es, basis; ntasks=3, prefetch=false)
        H4, S4 = assemble(hb, IntegralContext(BigFloat, KernelQuadrature(BigFloat), lv), es, basis; ntasks=3, prefetch=true)
        @test H3 == H4 && S3 == S4
    end
end

@ts "assembly and solve with packed BigFloat (SymPacked)" begin
    setprecision(BigFloat, 256) do
        U = BigFloat
        sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2)); lv = LogVariable()
        hd = HamiltonianData(sys, lv, U); es = ExponentSets([1.8, 1.8], [1.8, 0.45], [0.0, 0.0])
        basis = Key[(A, B, C, 0, j, k, 0) for A in 0:1, B in 0:1, C in 0:1, j in 0:1, k in 1:2][:]
        H1, S1 = assemble(hd, IntegralContext(U, KernelQuadrature(U), lv), es, basis; ntasks=3)
        H2, S2 = assemble(hd, IntegralContext(U, KernelQuadrature(U), lv), es, basis; ntasks=3, packed=true)
        @test H2 isa ThreeBodyFC.SymPacked && H1 == H2 && S1 == S2
        od = mktempdir()
        r1 = run_fc(sys, es; T=U, stages=1:2, packed=false, io=devnull)
        r2 = run_fc(sys, es; T=U, stages=1:2, packed=true, outdir=od, io=devnull)
        r3 = run_fc(sys, es; T=U, stages=1:2, packed=true, outdir=od, io=devnull)
        @test all(abs(a.E - b.E) < 1e-60 for (a, b) in zip(r1, r2))
        @test all(a.E == b.E for (a, b) in zip(r2, r3))
        @test all(r.solution.residual < 1e-60 for r in r2)
    end
end

@ts "exchange symmetrization in packed storage (from representative rows only)" begin
    setprecision(BigFloat, 256) do
        U = BigFloat
        sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 1)); lv = LogVariable()
        hd = HamiltonianData(sys, lv, U)
        es = ExponentSets([1.0, 0.25], [0.25, 1.0], [0.0, 0.0])
        basis = Key[(A, B, C, 0, j, k, 0) for A in 0:1, B in 0:1, C in 0:1, j in 0:1, k in 1:2][:]
        for sg in (+1, -1)
            ctx() = IntegralContext(U, KernelQuadrature(U), lv)
            H1, S1, C1 = ThreeBodyFC.assemble_projected(Exchange(sg), hd, ctx(), es, basis; ntasks=3)
            H2, S2, C2 = ThreeBodyFC.assemble_projected(Exchange(sg), hd, ctx(), es, basis; ntasks=3, packed=true)
            @test H2 isa ThreeBodyFC.SymPacked && size(H2, 1) == size(H1, 1)
            @test maximum(abs, Matrix(S2) - S1) < 1e-68 && maximum(abs, Matrix(H2) - H1) < 1e-68
            x = U.(1:size(H1, 1))
            @test C2 * x == Matrix(C2) * x && C1 * x == C2 * x
        end
        od = mktempdir()
        r1 = run_fc(sys, es; T=U, stages=1:2, symmetry=Exchange(+1), packed=false, io=devnull)
        r2 = run_fc(sys, es; T=U, stages=1:2, symmetry=Exchange(+1), packed=true, outdir=od, io=devnull)
        r3 = run_fc(sys, es; T=U, stages=1:2, symmetry=Exchange(+1), packed=true, outdir=od, io=devnull)
        @test all(abs(a.E - b.E) < 1e-60 for (a, b) in zip(r1, r2))
        @test all(a.E == b.E for (a, b) in zip(r2, r3))
        wf = load_wavefunction(od, 2)
        @test length(wf.c) == length(wf.basis)
    end
end

@ts "normalization while filling (dscale) is bit-identical to diagonal_normalize" begin
    setprecision(BigFloat, 256) do
        U = BigFloat
        sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 1)); lv = LogVariable()
        hd = HamiltonianData(sys, lv, U)
        es = ExponentSets([1.0, 0.25], [0.25, 1.0], [0.0, 0.0])
        basis = Key[(A, B, C, 0, j, k, 0) for A in 0:1, B in 0:1, C in 0:1, j in 0:1, k in 1:2][:]
        ctx() = IntegralContext(U, KernelQuadrature(U), lv)
        for sym in (NoSymmetry(), Exchange(+1), Exchange(-1))
            H1, S1, _ = ThreeBodyFC.assemble_projected(sym, hd, ctx(), es, basis; ntasks=3, packed=true)
            H1, S1, D1 = ThreeBodyFC.diagonal_normalize(H1, S1)
            d = Ref{Any}(nothing)
            H2, S2, _ = ThreeBodyFC.assemble_projected(sym, hd, ctx(), es, basis; ntasks=3, packed=true, dscale=d)
            @test d[] !== nothing && d[] == D1
            @test H2.buf == H1.buf && S2.buf == S1.buf
        end
        d = Ref{Any}(nothing)
        ThreeBodyFC.assemble_projected(Exchange(+1), hd, ctx(), es, basis; ntasks=3, dscale=d)
        @test d[] === nothing
        r1 = run_fc(sys, es; T=U, stages=1:2, symmetry=Exchange(+1), packed=true, outdir=mktempdir(), io=devnull)
        r2 = run_fc(sys, es; T=U, stages=1:2, symmetry=Exchange(+1), packed=true, io=devnull)
        @test all(a.E == b.E for (a, b) in zip(r1, r2))
    end
end

@ts "excited states (nstates, lifted inverse iteration)" begin
    sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
    es = ExponentSets([2.0, 2.0, 2.0, 1.0, 0.5], [2.0, 1.0, 0.5, 2.0, 2.0], zeros(5))
    r = run_fc(sys, es; stages=1:4, symmetry=Exchange(+1), nstates=2, io=devnull)
    @test fmt(r[end].states[2].E, 21) == "-2.145974045715965759824" && r[end].states[2].method === :cholesky_lift
    t = run_fc(sys, es; stages=4:4, symmetry=Exchange(-1), io=devnull)
    @test fmt(t[end].E, 21) == "-2.175229378157015767725"
    setprecision(BigFloat, 256) do
        kw = (; stages=1:3, symmetry=Exchange(+1), nstates=3, T=BigFloat, io=devnull,
              reference=["-2.90372437703411959831115924519440444669690537", "-2.14597404605441741580502897546192", nothing])
        a = run_fc(sys, es; packed=true, kw...)
        b = run_fc(sys, es; solver=CanonicalCut(), kw...)
        @test all(abs(x.E - y.E) < 1e-55 for (x, y) in zip(a[end].states, b[end].states))
        @test a[end].states[3].method === :cholesky_lift && length(a[end].errors) == 3 && isnan(a[end].errors[3])
        @test abs(a[end].errors[2] - Float64(a[end].states[2].E - parse(BigFloat, "-2.14597404605441741580502897546192"))) < 1e-30
    end
end

@ts "automatic exponent sets (auto_exponents), thresholds, error estimates" begin
    he = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
    @test thresholds(he).E == -2.0 && thresholds(ThreeBody(("1", "1", "1"), coulomb(-1, -1, 1))).E == -0.25
    @test isinf(thresholds(ThreeBody(("5", "5", "1"), [PowerTerm("-1", (-1, 0, 0)), PowerTerm("1/10", (1, 0, 0)),
                                                     PowerTerm("-1", (0, -1, 0)), PowerTerm("1/10", (0, 1, 0))])).E)
    @test pair_scales(he)[1:2] == (2.0, 2.0) && isnan(pair_scales(he)[3])
    @test ThreeBodyFC.tail_estimate(1e-3, 1e-2) ≈ 1e-3 * 0.1 / 0.9
    @test isnan(ThreeBodyFC.tail_estimate(1e-2, 1e-3))
    r = run_fc(he, ExponentSets([2.0], [2.0], [0.0]); stages=1:4, symmetry=Exchange(+1), io=devnull)
    @test isnan(r[2].estimates[1]) && 0.3 < r[4].estimates[1] / Float64(r[4].E - parse(BigFloat, "-2.903724377034119598311")) < 3
    @test all(x -> x.bound[1], r)
    sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 1))
    a = auto_exponents(sys; n=2, maxsets=3, io=devnull)
    h = a.history
    @test a.es.α[1] == 1.0 && a.es.β[1] == 1.0 && length(a.es.α) >= 2
    @test all(x -> x.chosen === nothing || x.gain > x.stage_gain, h)
    Es = [x.Es[1] for x in h if x.chosen !== nothing]
    @test issorted(Es; rev=true)
    @test all(x -> !isempty(x.ranking) && issorted([r.gain for r in x.ranking]; rev=true), h)
    @test all(x -> !(x.chosen isa Vector) || x.ranking[1].cand == x.chosen, h)
    last_ = h[findlast(x -> x.chosen !== nothing, h)]
    E = run_fc(sys, ExponentSets(getindex.(last_.sets, 1), getindex.(last_.sets, 2), getindex.(last_.sets, 3));
               stages=2:2, symmetry=Exchange(+1), io=devnull)[end].E
    @test Float64(E) == last_.Es[1]
end

@ts "expectation values (expectation)" begin
    setprecision(BigFloat, 256) do
        he = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
        d = mktempdir()
        run_fc(he, ExponentSets([2.0, 1.0], [1.0, 1.0], [0.0, 0.0]); stages=1:4, symmetry=Exchange(+1), outdir=d, io=devnull)
        wf = load_wavefunction(d, 4)
        v = expectation(he, wf, [TotalEnergy(), KineticEnergy(), PotentialEnergy(), DistancePower(0, 0, -1), DistancePower(1, 0, 0)])
        @test abs(v[1] - wf.E) < 1e-40
        @test abs(v[2] + v[3] - v[1]) < 1e-60
        @test abs(v[3] / v[1] - 2) < 1e-9
        @test abs(v[4] - parse(BigFloat, "0.945818448800")) < 1e-6 && abs(v[5] - parse(BigFloat, "0.929472294874")) < 1e-6
        @test_throws ArgumentError DistancePower(-2, 0, 0)
        w = expectation(he, wf, [Delta(1, 3), Delta(2, 3), Cusp(1, 3), Cusp(1, 2)])
        @test abs(w[1] - parse(BigFloat, "1.8104293185")) < 1e-6 && abs(w[1] - w[2]) < 1e-40
        @test abs(w[3] + 2) < 1e-4 && abs(w[4] - 0.5) < 1e-2
        q = variance(he, wf; N=40)
        @test abs(q.σ2 / parse(BigFloat, "1.2675992952e-6") - 1) < 1e-4 && abs(q.dE) < 1e-9
        @test temple(q.E, q.σ2, parse(BigFloat, "-2.14597404605")) < parse(BigFloat, "-2.903724377034")
        P = distribution(he, wf, (1, 2), [1e-4]; N=40)
        δ12, ν12 = expectation(he, wf, [Delta(1, 2), Cusp(1, 2)])
        @test abs(P[1] / (4 * BigFloat(π) * big"1e-8") / (δ12 * (1 + 2ν12 * big"1e-4")) - 1) < 1e-7
        d2 = mktempdir()
        run_fc(he, ExponentSets([2.0], [2.0], [0.0]); stages=1:4, symmetry=Exchange(+1), outdir=d2, io=devnull, logvar=NewLog(), T=BigFloat)
        ws = load_wavefunction(d2, 4)
        u = expectation(he, ws, [TotalEnergy(), Delta(1, 3), Cusp(1, 3)]; logvar=NewLog())
        Ps = distribution(he, ws, (1, 3), [1e-4]; logvar=NewLog(), N=40)
        @test abs(u[1] - ws.E) < 1e-40 && abs(u[2] - parse(BigFloat, "1.8104293185")) < 1e-4
        @test abs(Ps[1] / (4 * BigFloat(π) * big"1e-8") / (u[2] * (1 + 2u[3] * big"1e-4")) - 1) < 1e-6
    end
end

@ts "angular weights for L ≥ 1 (Angular.angular_weight)" begin
    W = ThreeBodyFC.Angular.angular_weight
    @test W(1, 0, 0) == [((0, 2, 0), 2 // 1)] && W(1, 1, 1) == [((2, 0, 0), 2 // 1)]
    @test Set(W(1, 0, 1)) == Set([((2, 0, 0), 1 // 1), ((0, 2, 0), 1 // 1), ((0, 0, 2), -1 // 1)])
    for L in 1:4, a in 0:L, c in 0:L
        @test Set(((q, p, s), w) for ((p, q, s), w) in W(L, a, c)) == Set(W(L, L - a, L - c)) && W(L, a, c) == W(L, c, a)
    end
end

@ts "run_fc for L ≥ 1 (He 2³P, 2¹P, infinite nuclear mass)" begin
    setprecision(BigFloat, 256) do
        he = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
        es = ExponentSets([2.0, 0.5, 2.0], [0.5, 2.0, 2.0], [0.0, 0.0, 0.0])
        kw = (stages=1:2, L=1, io=devnull, T=BigFloat, auto_precision=false)
        t = run_fc(he, es; symmetry=Exchange(-1), kw...)
        s = run_fc(he, es; symmetry=Exchange(+1), kw...)
        u = run_fc(he, es; symmetry=NoSymmetry(), kw...)
        @test [x.Msolved for x in t] == [30, 122] && abs(t[end].E - big"-2.133163785649") < 1e-11
        @test [x.Msolved for x in s] == [30, 122] && abs(s[end].E - big"-2.123842488173") < 1e-11
        @test t[end].E > big"-2.133164190779" && s[end].E > big"-2.123843086498"
        @test [x.Msolved for x in u] == [60, 244] && abs(u[end].E - t[end].E) < 1e-30
    end
end

@ts "mirror pairs for L ≥ 1 by partial waves give the same E as the monomial path" begin
    he = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
    @test ThreeBodyFC.Angular.angular_lmin(4, 0, 4) == 4 && ThreeBodyFC.Angular.angular_lmin(4, 1, 3) == 2
    @test [l for (l, _) in ThreeBodyFC.Angular.legendre_components(6, 1, 5)] == [4, 6]
    for (L, es, sym, lv, rl) in ((3, ExponentSets([2.0, 2.0], [0.25, 0.5], [0.0, 0.0]), Exchange(+1), LogVariable(), FCRule()),
                                 (2, ExponentSets([2.0, 2.0], [0.5, 1.0], [0.0, 0.0]), NoSymmetry(), LogVariable(), FCRule()))
        E = Dict{Bool,Any}()
        for pw in (false, true)
            ThreeBodyFC._PW_ENABLED[] = pw
            E[pw] = [x.E for x in run_fc(he, es; stages=1:2, L=L, symmetry=sym, logvar=lv, rule=rl, io=devnull)]
        end
        ThreeBodyFC._PW_ENABLED[] = true
        @test maximum(abs.(big.(E[true]) .- big.(E[false]))) < 1e-45
    end
end

@ts "coalescence expansion with non-binary exponents (exact κ)" begin
    setprecision(BigFloat, 256) do
        he = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
        es = ExponentSets([1.7, 1.7, 0.3], [1.7, 0.3, 1.7], [-0.1, -0.1, -0.1])
        d = mktempdir()
        run_fc(he, es; stages=3:3, symmetry=Exchange(+1), io=devnull, T=BigFloat, auto_precision=false, outdir=d)
        wf = load_wavefunction(d, 3)
        e2 = ExponentSets(copy(wf.α), copy(wf.β), copy(wf.γ)); basis = ThreeBodyFC._key7.(wf.basis)
        ψ, _ = ThreeBodyFC._coalescence(basis, e2, wf.c, 2, 0.0, false)
        r = big(3) // 7
        g = sum(a * r^p * log(r)^q * exp(-κ * r) for ((p, q, κ, _), a) in ψ)
        direct = sum(BigFloat(wf.c[i]) * (t[2] == 0 ? r^t[1] * r^t[3] * r^t[4] * log(r)^t[5] * exp(-(BigFloat(e2.α[t[6]]) + BigFloat(e2.γ[t[6]])) * r) : big(0))
                     for (i, t) in enumerate(basis))
        @test abs(g / direct - 1) < 1e-60
    end
end

@ts "saving excited-state wave functions (nstates, load_wavefunction(; state))" begin
    setprecision(BigFloat, 256) do
        sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
        es = ExponentSets([2.0, 2.0, 1.0, 0.5], [2.0, 1.0, 2.0, 2.0], zeros(4))
        d = mktempdir()
        r = run_fc(sys, es; stages=1:2, symmetry=Exchange(+1), nstates=3, io=devnull, T=BigFloat, auto_precision=false, outdir=d)
        for k in 1:3
            wf = load_wavefunction(d, 2; state=k)
            @test wf.E == r[end].states[k].E && abs(expectation(sys, wf, [TotalEnergy()])[1] - wf.E) < 1e-45
        end
        @test_throws SystemError load_wavefunction(d, 2; state=4)
    end
end

@ts "δ with Gaussian bases and L ≥ 1 (H₂⁺, 1810.07929 Table II)" begin
    setprecision(BigFloat, 256) do
        sys = ThreeBody(("1836.15267389", "1836.15267389", "1"), coulomb(1, 1, -1))
        es = binary_exponents(gaussian_r12(0.7519, 4.2305, 2.0))
        for (L, ref) in ((0, big"0.20673647629"), (1, big"0.20649132016"))
            d = mktempdir()
            run_fc(sys, es; stages=3:4, rule=FCRule(seeds=[(0, 0, 0, 0, 0)], jmax=0, dmin=-2), symmetry=Exchange(isodd(L) ? -1 : 1), L=L,
                   io=devnull, T=BigFloat, auto_precision=false, outdir=d)
            v = expectation(sys, load_wavefunction(d, 4), [Delta(1, 3), Delta(2, 3), Delta(1, 2)])
            @test abs(v[1] - ref) < 2e-6 && abs(v[1] - v[2]) < 1e-40 && abs(v[3]) < 1e-10
            ν = expectation(sys, load_wavefunction(d, 4), [Cusp(1, 3)])[1]
            @test abs(ν + big"1836.15267389" / big"1837.15267389") < 2e-2
        end
    end
end

@ts "unnatural parity (He 2p² ³Pᵉ, unnatural=true)" begin
    w = ThreeBodyFC.Angular.angular_weight(1, 0, 0; unnatural=true)
    r1, r2, r12 = big(3) // 2, big(5) // 7, big(9) // 5
    @test sum(c * r1^p * r2^q * r12^t for ((p, q, t), c) in w) == 2r1^2 * r2^2 - (r1^2 + r2^2 - r12^2)^2 / 2
    @test_throws ArgumentError ThreeBodyFC.Angular.angular_weight(2, 2, 0; unnatural=true)
    he = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
    es = ExponentSets([1.0, 1.0, 0.5, 1.5], [1.0, 0.5, 1.0, 1.5], zeros(4))
    kw = (stages=1:2, L=1, unnatural=true, io=devnull, T=Float64x4)
    t = run_fc(he, es; symmetry=Exchange(-1), kw...)
    u = run_fc(he, es; symmetry=NoSymmetry(), kw...)
    s = run_fc(he, es; symmetry=Exchange(+1), kw...)
    @test [x.Msolved for x in t] == [26, 98] && abs(t[end].E - big"-0.710500102886172") < 1e-13
    @test abs(u[end].E - t[end].E) < 1e-25 && s[end].E > t[end].E
    @test_throws ArgumentError run_fc(he, es; stages=1:1, unnatural=true, io=devnull)
    d = mktempdir(); run_fc(he, es; stages=2:2, symmetry=Exchange(-1), L=1, unnatural=true, io=devnull, T=Float64x4, outdir=d)
    @test all(iszero, expectation(he, load_wavefunction(d, 2), [Delta(1, 3), Delta(2, 3), Delta(1, 2)]))
end

@ts "inclusion check: hand-chosen sets lie in the general candidate space" begin
    include(joinpath(@__DIR__, "systems.jl"))
    cases = [(k, v[1], v[2]) for (k, v) in SYSTEMS]
    for (name, sys, es) in cases
        gs = [ThreeBodyFC.exponent_grid(sys, d) for d in 1:3]; w = ThreeBodyFC.exponent_window(sys)
        trip = [(es.α[i], es.β[i], es.γ[i]) for i in eachindex(es.α)]
        for t in trip, d in 1:3
            x = t[d]; x == 0 && continue
            @test gs[d] !== nothing
            @test (x < 0) == (gs[d].kind === :repulsive)
            @test w[d][1] * (1 - 1e-12) <= abs(x) <= w[d][2] * (1 + 1e-12)
        end
        for d in 1:3
            groups = Dict{Any,Vector{Float64}}()
            for t in trip
                push!(get!(groups, Tuple(t[i] for i in 1:3 if i != d), Float64[]), t[d])
            end
            for v in values(groups)
                v = sort(unique(abs.(v))); length(v) >= 3 || continue
                @test length(v) <= 8 && all(v[i+1] / v[i] == 2 for i in 1:length(v)-1)
            end
        end
    end
end

@ts "integer monomial coefficients (BigInt beyond Int128)" begin
    I = ThreeBodyFC.Integrals
    for (A, B, C, m) in [(3, 4, 5, 2), (20, 20, 20, 10), (67, 30, 10, 12)]
        a = I._monomials_int(A, B, C, m); b = I._monomials_int(BigInt, A, B, C, m)
        @test keys(a) == keys(b) && all(BigInt.(a[k]) == b[k] for k in keys(a))
    end
    @test_throws OverflowError I._monomials_int(Int128, 60, 60, 60, 10)
    a = I._monomials_int(60, 60, 60, 10)
    @test eltype(valtype(a)) == BigInt
    @test sum(sum, values(a)) == big(2)^180 * big(3)^10
end

@ts "excited states do not pick up lower states (pseudo-states above the Ps⁻ threshold)" begin
    sys = ThreeBody(("1", "1", "1"), coulomb(-1, -1, 1))
    r = run_fc(sys, ExponentSets([0.5], [0.5], [0.0]); stages=1:4, symmetry=Exchange(+1), nstates=3, io=devnull)
    for x in r
        Es = [Float64(s.E) for s in x.states]
        @test issorted(Es) && minimum(diff(Es)) > 1e-8
    end
    for k in 1:3, i in 2:length(r)
        @test r[i].states[k].E <= r[i-1].states[k].E + 1e-25
    end
end

@ts "integral constants follow the working precision" begin
    I = ThreeBodyFC.Integrals
    for bits in (640, 1024)
        setprecision(BigFloat, bits) do
            tol = BigFloat(2)^(-bits + 8)
            γ = BigFloat(Base.MathConstants.eulergamma)
            @test abs(I.euler_gamma(BigFloat) - γ) <= tol
            @test abs(I.digamma_int(BigFloat, 5) - (1 + BigFloat(1) / 2 + BigFloat(1) / 3 + BigFloat(1) / 4 - γ)) <= tol
            @test abs(I.pi2over6(BigFloat) - BigFloat(π)^2 / 6) <= tol
            @test abs(I.gamma_inv(BigFloat, 30) * factorial(BigInt(29)) - 1) <= tol
        end
    end
end

@ts "factor_S = false gives bit-identical E; smin is NaN at stages with a previous E" begin
    he = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
    es = ExponentSets([2.0, 2.0], [2.0, 1.0], [0.0, 0.5])
    setprecision(BigFloat, 256) do
        kw = (; stages=1:4, symmetry=Exchange(+1), T=BigFloat, engine=KernelQuadrature(BigFloat), auto_precision=false, io=devnull)
        a = run_fc(he, es; kw..., solver=CholeskyInverseIteration(ozaki_min=32))
        for sp in (:none,)
            b = run_fc(he, es; kw..., solver=CholeskyInverseIteration(ozaki_min=32, factor_S=false))
            @test [s.E for s in a] == [s.E for s in b]
            @test b[1].solution.smin == a[1].solution.smin
            @test all(isnan(s.solution.smin) for s in b[2:end])
            @test all(s.solution.method === :cholesky for s in b)
        end
    end
end

@ts "etol_rel (stop inverse iteration at an energy tolerance)" begin
    he = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
    es = ExponentSets([2.0, 2.0], [2.0, 1.0], [0.0, 0.5])
    setprecision(BigFloat, 256) do
        kw = (; stages=1:5, symmetry=Exchange(+1), T=BigFloat, engine=KernelQuadrature(BigFloat), auto_precision=false, io=devnull)
        a = run_fc(he, es; kw..., solver=CholeskyInverseIteration(ozaki_min=32))
        b = run_fc(he, es; kw..., solver=CholeskyInverseIteration(ozaki_min=32, etol_rel=1e-6))
        @test all(s.solution.method === :cholesky for s in b)
        @test a[1].E == b[1].E && a[2].E == b[2].E
        @test all(b[n].solution.iterations <= a[n].solution.iterations for n in 3:5)
        @test any(b[n].solution.iterations < a[n].solution.iterations for n in 3:5)
        @test all(abs(b[n].E - a[n].E) <= 1e-6 * (a[n-2].E - a[n-1].E) for n in 3:5)
    end
end

@ts "basis order (previous stage first) and reuse of the previous factor (reuse_factor)" begin
    he = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
    es = ExponentSets([2.0, 2.0], [2.0, 1.0], [0.0, 0.5])
    setprecision(BigFloat, 256) do
        kw0 = (; stages=1:4, symmetry=Exchange(+1), T=BigFloat, engine=KernelQuadrature(BigFloat), auto_precision=false, io=devnull,
               solver=CholeskyInverseIteration(ozaki_min=32))
        a0 = run_fc(he, es; kw0..., reuse_factor=false); b0 = run_fc(he, es; kw0...)
        @test [s.E for s in a0] == [s.E for s in b0] && !any(occursin("factor_reused", s.solution.note) for s in b0)
        buf = IOBuffer()
        b1 = run_fc(he, es; kw0..., symmetry=NoSymmetry(), io=buf)
        @test count("FCRule(dedup=false) lets the next stage reuse", String(take!(buf))) == 1 &&
              !any(occursin("factor_reused", s.solution.note) for s in b1)
        c = run_fc(he, es; stages=1:4, symmetry=NoSymmetry(), T=BigFloat, engine=KernelQuadrature(BigFloat), auto_precision=false, io=devnull,
                   rule=FCRule(dedup=false), solver=CholeskyInverseIteration(ozaki_min=32))
        @test all(c[n].basis[1:length(c[n-1].basis)] == c[n-1].basis for n in 2:4)
    end
    a0 = 402.1373167037964
    mu = ThreeBody(("206.768262", "1.0", "7294.2996"), coulomb(-1, -1, 2))
    esm = ExponentSets([a0, a0], [1.9997262954711914, 0.9998626708984375], [0.0, 0.0])
    setprecision(BigFloat, 384) do
        kw = (; stages=1:5, T=BigFloat, engine=KernelQuadrature(BigFloat), logvar=LogVariable(0.0; w2=0.25), rule=FCRule(dedup=false),
              auto_precision=false, io=devnull)
        M5 = 0
        for fS in (false, true)
            sv = CholeskyInverseIteration(fallback=PivotedSubset(), ozaki_min=64, factor_S=fS)
            a = run_fc(mu, esm; kw..., solver=sv, reuse_factor=false); b = run_fc(mu, esm; kw..., solver=sv)
            @test all(s.solution.method === :cholesky for s in b)
            @test all(b[n].basis[1:length(b[n-1].basis)] == b[n-1].basis for n in 2:5)
            @test [s.basis for s in a] == [s.basis for s in b]
            @test a[1].E == b[1].E && a[2].E == b[2].E
            @test occursin("factor_reused=$(length(b[2].basis))", b[3].solution.note)
            @test !occursin("factor_reused", b[4].solution.note)
            @test occursin("factor_reused=$(length(b[4].basis))", b[5].solution.note)
            @test all(abs(b[n].E - a[n].E) <= 100 * max(a[n].solution.residual, b[n].solution.residual) * abs(a[n].E) for n in 3:5)
            @test all(b[n].solution.residual < 1e-30 for n in 1:5)
            M5 = length(b[5].basis)
        end
        io = IOBuffer()
        r = run_fc(mu, esm; kw..., io=io, solver=CholeskyInverseIteration(fallback=PivotedSubset(), ozaki_min=64),
                   max_mem_gb=max(ThreeBodyFC._matrix_gb(BigFloat, M5), ThreeBodyFC._build_gb(BigFloat, M5)) * 1.0001)
        @test occursin("not reused", String(take!(io))) && !occursin("factor_reused", r[5].solution.note)
    end
end

include("test_order.jl")

@ts "Gaussian in r₁₂ (η)" begin
    setprecision(BigFloat, 256) do
        ctx = ThreeBodyFC.IntegralContext(BigFloat, KernelQuadrature(BigFloat), LogVariable())
        ec = ThreeBodyFC.ElementCache{BigFloat}()
        tab = ThreeBodyFC.gauss_table(BigFloat, 1.5, 0.75, 0.0, (0, 9), (-4, 8))
        for (A, B, C, D) in [(0, 0, 0, 0), (2, 1, 0, 0), (1, 1, 3, -2), (3, 0, 1, -3), (0, 4, 2, 1)]
            v = ThreeBodyFC.element_gauss(tab, ctx.binom, A, B, C, D)
            @test abs(v / ThreeBodyFC.element!(ec, ctx, A, B, C, D, 0, 1.5, 1.5, 0.75) - 1) < 1e-70
        end
        sys = ThreeBody(("1836.152701", "1836.152701", "1"), coulomb(1, 1, -1))
        es = binary_exponents(gaussian_r12(0.6695, 5.9054, 2.0))
        kw = (stages=0:2, rule=FCRule(seeds=[(0, 0, 0, 0, 0)], jmax=0), io=devnull, T=BigFloat, auto_precision=false)
        s = run_fc(sys, es; symmetry=Exchange(+1), kw...)
        @test abs(s[1].E - big"-0.5732504092") < 1e-9
        @test [x.Msolved for x in s] == [1, 12, 75] && abs(s[end].E - big"-0.597135582326486") < 1e-14
        @test s[end].E > big"-0.59713906312340"
        p = run_fc(sys, es; symmetry=Exchange(-1), L=1, kw...)
        @test [x.Msolved for x in p] == [1, 21, 131] && abs(p[end].E - big"-0.596871538152990") < 1e-14
        @test p[end].E > big"-0.596873738784279"
    end
end

@ts "expectation values for L ≥ 1 and the r₁₂ Gaussian (H₂⁺ ³P)" begin
    setprecision(BigFloat, 256) do
        sys = ThreeBody(("1836.15267247", "1836.15267247", "1"), coulomb(1, 1, -1))
        es = binary_exponents(gaussian_r12(0.9221, 4.5, 2.0))
        d = mktempdir()
        run_fc(sys, es; stages=2:2, rule=FCRule(seeds=[(0, 0, 0, 0, 0)], jmax=0, dmin=-2), symmetry=Exchange(-1), L=1,
               io=devnull, T=BigFloat, auto_precision=false, outdir=d)
        wf = load_wavefunction(d, 2)
        v = expectation(sys, wf, [TotalEnergy(), DistancePower(1, 0, 0), DistancePower(0, 1, 0), DistancePower(0, 0, 1)])
        @test abs(v[1] - wf.E) < 1e-60
        @test abs(v[2] - v[3]) < 1e-60
        @test abs(v[4] - 2.0666) < 2e-3
        @test abs(expectation(sys, wf, [Cusp(1, 3)])[1] + 1) < 5e-2
        δ = expectation(sys, wf, [Delta(1, 3), Delta(2, 3), Delta(1, 2)])
        @test abs(δ[1] - δ[2]) < 1e-60 && abs(δ[1] - 0.2065) < 5e-3 && iszero(δ[3])
    end
end

@ts "quadrature operators and Breit–Pauli (He 1¹S)" begin
    setprecision(BigFloat, 256) do
        sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
        es = ExponentSets([2.0, 2.0, 1.0], [2.0, 1.0, 2.0], zeros(3))
        d = mktempdir()
        run_fc(sys, es; stages=3:3, symmetry=Exchange(+1), io=devnull, T=BigFloat, auto_precision=false, outdir=d)
        wf = load_wavefunction(d, 3)
        v = expectation_quad(sys, wf, [InverseSquare(1, 3), InverseSquare(2, 3), InverseSquare(1, 2), P4(1), P4(2)]; N=30)
        @test abs(v[1] - v[2]) < 1e-20 && abs(v[4] - v[5]) < 1e-20
        @test abs(v[1] / big"6.017408867" - 1) < 1e-4 && abs(v[3] / big"1.464770923" - 1) < 1e-3 && abs(v[4] / big"54.08806723" - 1) < 1e-3
        r = relativistic_correction(sys, wf; N=30)
        @test abs(r.Erel - big"-1.951754767") < 2e-3 && r.Z == 2
        p3 = expectation_quad(sys, wf, [P4(3)]; N=30)[1]; p3x2 = expectation_quad(sys, wf, [P4(3)]; N=30, T=Float64x2)[1]
        @test p3 > 0 && isfinite(p3) && abs(p3x2 / p3 - 1) < 1e-15
        qd = qed_correction(sys, wf; lnk0=big"4.370160223070", N=30)
        @test abs(qd.EQED - big"57.288165") < 0.05 && abs(qd.araki_sucher - big"0.989273545") < 2e-2 && abs(qd.delta12 - big"0.1063453706") < 1e-4
    end
end

@ts "quadrature operators and σ² for L ≥ 1 (angular derivatives)" begin
    setprecision(BigFloat, 256) do
        he = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
        for (es, kw) in ((ExponentSets([2.0, 0.5, 2.0], [0.5, 2.0, 2.0], zeros(3)), (L=1, symmetry=Exchange(-1))),
                         (ExponentSets([1.0, 1.0, 0.5, 1.5], [1.0, 0.5, 1.0, 1.5], zeros(4)), (L=1, unnatural=true, symmetry=Exchange(-1))))
            d = mktempdir()
            run_fc(he, es; stages=2:2, io=devnull, T=BigFloat, auto_precision=false, outdir=d, kw...)
            wf = load_wavefunction(d, 2)
            cl = expectation(he, wf, [DistancePower(1, 0, 0), DistancePower(0, 0, -1), KineticEnergy()])
            q = expectation_quad(he, wf, [DistancePower(1, 0, 0), DistancePower(0, 0, -1), KineticQuad(:laplacian), KineticQuad(:gradient)]; N=60)
            @test all(abs.(q .- [cl[1], cl[2], cl[3], cl[3]]) .< 1e-9)
            v = variance(he, wf; N=60)
            @test 0 < v.σ2 < 1e-3 && abs(v.dE) < 1e-8
        end
        d = mktempdir()
        run_fc(he, ExponentSets([2.0, 2.0, 1.0], [2.0, 1.0, 2.0], zeros(3)); stages=2:2, symmetry=Exchange(+1), io=devnull, T=BigFloat, auto_precision=false, outdir=d)
        wf = load_wavefunction(d, 2)
        a = variance(he, wf; N=30); b = expectation_quad(he, wf, [ThreeBodyFC._Residual(2)]; N=30, T=Float64x4)[1]
        @test abs(b / a.σ2 - 1) < 1e-25
    end
end

@ts "quadrature: cyclic reordering (input (nucleus, e, e)) keeps the pairs; pair-axis grid (PairAxis)" begin
    setprecision(BigFloat, 256) do
        sys = ThreeBody(("Inf", "1", "1"), coulomb(2, -1, -1))
        d = mktempdir()
        run_fc(sys, ExponentSets([2.0, 2.0, 1.0], [2.0, 1.0, 2.0], zeros(3)); stages=2:2, symmetry=Exchange(+1), io=devnull, T=BigFloat, auto_precision=false, outdir=d)
        wf = load_wavefunction(d, 2)
        @test wf.order == (2, 3, 1)
        ops = [DistancePower(1, 0, 0), DistancePower(0, 1, 0), DistancePower(0, -1, 0)]
        cl = expectation(sys, wf, ops)
        @test all(abs.(expectation_quad(sys, wf, ops; N=40) .- cl) .< 1e-9)
        @test all(abs.(expectation_quad(sys, wf, ops; N=80, coords=PairAxis(1, 2)) .- cl) .< 1e-8)
        @test all(abs.(expectation_quad(sys, wf, ops; N=80, coords=PairAxis(2, 3)) .- cl) .< 1e-8)
        he = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
        d = mktempdir()
        run_fc(he, ExponentSets([2.0, 2.0, 1.0], [2.0, 1.0, 2.0], zeros(3)); stages=2:2, symmetry=Exchange(+1), io=devnull, T=BigFloat, auto_precision=false, outdir=d)
        w2 = load_wavefunction(d, 2)
        r = expectation_quad(he, w2, [Retardation(), Retardation(1, 2), QTerm(1, 2)]; N=30)
        @test r[1] == r[2]
        @test abs(4 * BigFloat(π) * r[3] - araki_sucher(he, w2; N=30)) < 1e-20
    end
end

@ts "no silent spurious eigenvalues (He 3¹D, non-binary exponent 0.33)" begin
    setprecision(BigFloat, 256) do
        he = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
        es = ExponentSets([2.0, 0.33, 2.0], [0.33, 2.0, 2.0], zeros(3))
        r = run_fc(he, es; stages=1:4, L=2, symmetry=Exchange(+1), io=devnull, T=BigFloat, auto_precision=false)
        @test length(r) == 3 && all(x -> x.E > -2.06, r)
    end
end

@ts "leading fine structure (He 2³P, Pachucki–Yerokhin arXiv:1011.2467 Table 2)" begin
    setprecision(BigFloat, 256) do
        M = string(1 / big"1.37093355570e-4")
        he = ThreeBody(("1", "1", M), coulomb(-1, -1, 2))
        d = mktempdir()
        run_fc(he, ExponentSets([2.0, 0.5, 2.0], [0.5, 2.0, 2.0], zeros(3)); stages=2:3, L=1, symmetry=Exchange(-1), io=devnull, T=BigFloat, outdir=d)
        f = fine_structure(he, load_wavefunction(d, 3); N=50)
        khz(E) = E * (1 / big"137.035999679")^2 * 2 * big"3289841960361"
        @test abs(f.L2 - 2) < 1e-30
        @test abs(khz(f.E[0] - f.E[1]) / big"29563765.45" - 1) < 1e-6
        @test abs(khz(f.E[1] - f.E[2]) / big"2320241.43" - 1) < 3e-6
    end
end

@ts "quadrature working type from the coefficient magnitudes" begin
    setprecision(BigFloat, 320) do
        @test quad_type((c=[big"1e10", big"-3e9"],)) == Float64x3
        @test quad_type((c=[big"1e40"],)) == Float64x4
        @test quad_type((c=[big"1e45"],)) == BigFloat
        @test quad_type((c=[big"1e10"],); floor=Float64x4) == Float64x4
        @test_logs (:warn, r"not enough") ThreeBodyFC._resolve_quad_type(Float64x3, (c=[big"1e40"],))
        @test_logs ThreeBodyFC._resolve_quad_type(Float64x4, (c=[big"1e40"],))
        @test_logs (:warn, r"setprecision") quad_type((c=[big"1e80"],))   # 266 + 64 > 320
    end
end

@ts "vacuum polarization Uehling (reference integrals; leading Zα term −(4/15)α³Z⟨δ⟩ for He)" begin
    setprecision(BigFloat, 256) do
        ref = [(big"1e-3", big"1.2383062367366131660908545625555402465425e+01"), (big"0.5", big"9.6865945577363702097598371490577985618925e-01"),
               (big"3", big"1.7736300177270858116300665880941518456695e-02"), (big"50", big"1.9152291156648741388384930662470855080380e-24")]
        pt = (V = [((-1, 0, 0), big"-2")],)
        Iof(T) = (o = ThreeBodyFC._prepare_quad(Uehling(1, 3), pt, (1, 2, 3), T); [BigFloat(ThreeBodyFC._uehling_integral(o, T(x / o.x0))) for (x, _) in ref])
        I4, Ib = Iof(Float64x4), Iof(BigFloat)
        @test all(abs(Ib[k] / ref[k][2] - 1) < 1e-39 for k in eachindex(ref))
        @test all(abs(I4[k] - Ib[k]) < 1e-60 for k in eachindex(ref))
        sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
        es = ExponentSets([2.0, 2.0, 1.0], [2.0, 1.0, 2.0], zeros(3))
        d = mktempdir()
        run_fc(sys, es; stages=3:3, symmetry=Exchange(+1), io=devnull, T=BigFloat, auto_precision=false, outdir=d)
        wf = load_wavefunction(d, 3)
        v = expectation_quad(sys, wf, [Uehling(1, 3), Uehling(2, 3)]; N=30)
        @test abs(v[1] / v[2] - 1) < 1e-20
        α = 1 / big"137.035999177"
        lead = -(big(4) / 15) * α^3 * 2 * expectation(sys, wf, [Delta(1, 3)])[1]
        @test 0.975 < v[1] / lead < 0.99
    end
end

@ts "Bethe logarithm (velocity-form transitions, closed-form D, asymptotics, He 1¹S)" begin
    setprecision(BigFloat, 256) do
        sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
        d0 = mktempdir()
        run_fc(sys, ExponentSets([2.0, 2.0, 1.0], [2.0, 1.0, 2.0], zeros(3)); stages=1:2, symmetry=Exchange(+1), T=BigFloat, auto_precision=false, outdir=d0, io=devnull)
        wf0 = load_wavefunction(d0, 2)
        sr = [p for n in 0:7 for p in ((4.0 * 3.0^n, 2.0, 0.0), (2.0, 4.0 * 3.0^n, 0.0))]
        @test short_range_sets((2.0, 2.0, 0.0), 1; ζ0=4.0, τ=3.0, n=8) == [sr[i] for i in 1:2:16]
        inter = bethe_intermediate(sys, ExponentSets([2.0, 0.5, 2.0], [0.5, 2.0, 2.0], zeros(3)); n=2, symmetry=Exchange(+1), short_range=sr, pow=1)
        spec = bethe_spectrum(sys, wf0, inter; weights=(1, 1))
        @test spec.M == 368 && abs(spec.trk - 3) < 1e-4
        @test abs(spec.trk - big"2.99991661215753441") < 1e-15
        r = bethe_log(spec, sys, wf0; window=(10, 200))
        @test abs(r.D / spec.D_basis - 1) < 2e-3
        @test abs(r.coef.a / big"-257.39788565" - 1) < 1e-8
        @test abs(r.β - big"4.37016022306") < 5e-4
    end
end

@ts "general three-body mα⁵ QED (qed_leading, assembly of 2512.04623)" begin
    setprecision(BigFloat, 256) do
        α = 1 / big"137.035999177"; kHz = big"6.5796839204999e12"
        es = ExponentSets([2.0, 2.0, 1.0], [2.0, 1.0, 2.0], zeros(3))
        he = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
        d0 = mktempdir(); run_fc(he, es; stages=1:3, symmetry=Exchange(+1), T=BigFloat, auto_precision=false, outdir=d0, io=devnull)
        w = load_wavefunction(d0, 3)
        @test abs(qed_leading(he, w; bethe=big"4.370160223070", N=30).E_over_α3 - qed_correction(he, w; lnk0=big"4.370160223070", N=30).EQED) < 1e-30
        M = big"7294.29954171"
        he4 = ThreeBody(("1", "1", "7294.29954171"), coulomb(-1, -1, 2))
        d1 = mktempdir(); run_fc(he4, es; stages=1:3, symmetry=Exchange(+1), T=BigFloat, auto_precision=false, outdir=d1, io=devnull)
        w4 = load_wavefunction(d1, 3)
        βM = big"4.37016022306" + log(M / (M + 1)) + big"0.09438944" / M - big"0.16501" / M^2
        r4 = qed_leading(he4, w4; bethe=βM, α=α, N=30); r0 = qed_leading(he, w; bethe=big"4.37016022306", α=α, N=30)
        μ = M / (M + 1); κ = 2μ; δ = κ^3 / big(π); β1 = big"2.98412855576549761" + log(4μ)
        hp(M) = (isinf(M) ? (-2 / (3big(π)) * 2big(π) * 2 * 8 / big(π) * (big"2.98412855576549761" + log(big(4))) + 2 * big"4" / 3 * (log(1 / α^2) + big"5" / 6) * 8 / big(π) - 2 * big"4" / 15 * 8 / big(π)) :
                 (-2 / (3big(π)) * 2big(π) * 2 * (1 + 2 / M)^2 * δ * β1 + 4 / M * (2 * (-log(α) + big"31" / 3) / 3) * δ + big"14" / 3 * 4 / M * δ * log(2κ) +
                  2 * big"4" / 3 * (log(1 / α^2) + big"5" / 6) * δ + 8 / M^2 * big"4" / 3 * log(M / α^2) * δ - 2 * big"4" / 15 * δ))
        rec = ((r4.E_over_α3 - hp(M)) - (r0.E_over_α3 - hp(big(Inf)))) * α^3 * kHz
        @test abs(rec - big"-10411.907046949") < 1e-6 && abs(rec - big"-10335.035") < 100
        @test r4.kinds == [:lepton, :lepton, :nucleus] && haskey(r4.parts, "Q term (1,3)")
    end
end

@ts "Bethe logarithm for L ≥ 1 (CG-coupled invariants, three channels, defaults, He 2¹P)" begin
    setprecision(BigFloat, 256) do
        w = ThreeBodyFC.Angular.coupled_weight(0, 0, 1, 1, 1)
        @test length(w) == 1 && w[1][1] == (2, 0, 0) && abs(w[1][2] - sqrt(big(2))) < 1e-70
        for L0 in 1:2, (Lp, un) in ((L0 - 1, false), (L0, true), (L0 + 1, false)), a in 0:L0, c in 0:Lp-un, k in 1:4
            ThreeBodyFC.Angular.coupled_weight(L0, a, Lp, c, k; unnaturalp=un)
        end
        @test true
        sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
        d0 = mktempdir()
        run_fc(sys, ExponentSets([2.0, 0.5, 2.0], [0.5, 2.0, 2.0], zeros(3)); stages=1:1, L=1, symmetry=Exchange(1), T=BigFloat, auto_precision=false, outdir=d0, io=devnull)
        wf0 = load_wavefunction(d0, 1)
        r = bethe_logarithm(sys, wf0; n1=1, nsr=3, τ=6.0, pow=1, cpow=0)
        @test [(c.L, c.unnatural, c.M) for c in r.channels] == [(0, false, 52), (1, true, 54), (2, false, 132)]
        @test r.below == 3
        @test abs(r.channels[1].trk - big"-0.083400651527752521") < 1e-13 && abs(r.channels[3].trk - big"2.423611782275077797") < 1e-13
        @test abs(r.β - big"4.5587602331755945080563") < 1e-12
    end
end
@ts "Bethe logarithm stops on non-binary exponents of ψ₀ or intermediate states" begin
    setprecision(BigFloat, 256) do
        sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
        d0 = mktempdir()
        run_fc(sys, ExponentSets([2.1], [2.1], [0.0]); stages=1:1, symmetry=Exchange(+1), T=BigFloat, auto_precision=false, outdir=d0, io=devnull)
        wf0 = load_wavefunction(d0, 1)
        inter = bethe_intermediate(sys, ExponentSets([2.0], [2.0], [0.0]); n=1, symmetry=Exchange(+1))
        @test_throws ArgumentError bethe_spectrum(sys, wf0, inter)
        @test ThreeBodyFC._check_binary_sums(merge(wf0, (α=[2.0], β=[2.0], γ=[0.0])), inter) === nothing
        d1 = mktempdir()
        run_fc(sys, ExponentSets([2.0], [2.0], [0.0]); stages=1:1, symmetry=Exchange(+1), T=BigFloat, auto_precision=false, outdir=d1, io=devnull)
        w1 = load_wavefunction(d1, 1)
        @test_logs (:warn, r"cut=60") bethe_spectrum(sys, w1, inter; cut=60)
        @test_logs bethe_spectrum(sys, w1, inter; cut=40)
    end
end

@ts "auto_exponents excludes sets nearly equal to the start (ratio within 2^{1/8}, H⁻)" begin
    sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 1))
    a = 0.5002803802490234
    r = auto_exponents(sys; n=1, start=ExponentSets([a], [a], [0.0]), maxsets=3, logvar=NewLog(), io=devnull)
    ps = collect(zip(r.es.α, r.es.β, r.es.γ))
    near1(x, y) = x == y || (x != 0 && y != 0 && sign(x) == sign(y) && abs(log(abs(x / y))) < log(2) / 8)
    @test !any(near1(p[1], q[1]) && near1(p[2], q[2]) && near1(p[3], q[3]) ||
               near1(p[2], q[1]) && near1(p[1], q[2]) && near1(p[3], q[3]) for (i, p) in enumerate(ps) for q in ps[i+1:end])
end

@ts "late_greedy adds the best candidate below the threshold ratio at the same Meff (He)" begin
    sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
    es = ExponentSets([2.0], [2.0], [0.0])
    cs = [(2.0, 1.0, 0.0), (1.0, 1.0, -0.5), (2.0, 2.0, 0.125), (1.0, 1.0, 0.0)]
    r0 = late_greedy(sys, es; n=1, rounds=1, thr=0.0, cands=cs, io=devnull)
    @test length(r0.es) == 1 && r0.history[1].chosen === nothing
    @test length(r0.history[1].rows) == 4 && all(x -> isfinite(x.ratio) && x.ratio >= 0, r0.history[1].rows)
    r1 = late_greedy(sys, es; n=1, rounds=1, thr=Inf, cands=cs, io=devnull)
    b = argmin(x -> x.ratio, r1.history[1].rows)
    @test length(r1.es) == 2 && (r1.es.α[2], r1.es.β[2], r1.es.γ[2]) == b.cand == r1.history[1].chosen
    r2 = late_greedy(sys, es; n=1, rounds=1, thr=0.0, cands=[(2.0, 2.0, 0.0), (2.0 * 2^(1 / 16), 2.0, 0.0), (1.0, 1.0, 0.0)], io=devnull)
    @test [x.cand for x in r2.history[1].rows] == [(1.0, 1.0, 0.0)]
    r3 = late_greedy(sys, es; n=2, rounds=1, thr=0.0, cands=cs, screen=1, topk=2, io=devnull)
    h = r3.history[1]
    @test length(h.screened) == 4 && [x.cand for x in h.rows] == [x.cand for x in sort(h.screened; by=x -> x.ratio)[1:2]]
end

@ts "auto_exponents adds widened sets when the default start cannot resolve high roots (He 3³S)" begin
    sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
    a = auto_exponents(sys; n=3, nstates=2, symmetry=Exchange(-1), maxsets=1, io=devnull)
    @test collect(zip(a.es.α, a.es.β, a.es.γ)) == [(2.0, 2.0, 0.0), (2.0, 0.5, 0.0)]
    @test_throws ErrorException auto_exponents(sys; n=3, nstates=2, symmetry=Exchange(-1), maxsets=1, io=devnull,
                                               start=ExponentSets([2.0], [2.0], [0.0]))
end

@ts "rational dmin slope (lower bound −⌈q n⌉ at stage n)" begin
    @test [ThreeBodyFC._dmin(FCRule(dmin=2//1), n) for n in 1:4] == [-2, -4, -6, -8]
    @test [ThreeBodyFC._dmin(FCRule(dmin=1//2), n) for n in 1:4] == [-1, -1, -2, -2]
    sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
    es = ExponentSets([2.0], [2.0], [0.0])
    a = run_fc(sys, es; stages=1:3, symmetry=Exchange(+1), io=devnull)
    b = run_fc(sys, es; stages=1:3, rule=FCRule(dmin=2//1), symmetry=Exchange(+1), io=devnull)
    @test [s.M for s in a] == [s.M for s in b] && [s.E for s in a] == [s.E for s in b]
end

@ts "drop_exact keeps the space for βσ ≠ 0 (expansion in σ)" begin
    sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
    es = ExponentSets([2.0, 1.0], [1.0, 2.0], [0.0, 0.0])
    for b in (1.0, -0.5)
        a = run_fc(sys, es; stages=1:2, logvar=LogVariable(b), rule=FCRule(dedup=false), drop_exact=true, io=devnull)
        c = run_fc(sys, es; stages=1:2, logvar=LogVariable(b), rule=FCRule(dedup=false), drop_exact=false, io=devnull)
        @test all(abs(x.E - y.E) < 1e-40 for (x, y) in zip(a, c))
        @test a[end].nexact > 0
    end
end

@ts "σ = r₁ + w₂ r₂ (LogVariable(; w2)): symmetry of H and unchanged default" begin
    mu = ThreeBody(("206.768262", "1.0", "7294.2996"), coulomb(-1, -1, 2))
    es = ExponentSets([402.125, 402.125], [1.0, 4.0], [0.0, -0.5])
    for (b, w) in ((0.0, 0.25), (0.5, 0.25), (-0.5, 0.75))
        @test hermiticity(mu, es; logvar=LogVariable(b; w2=w)) < 1e-55
    end
    @test_throws ArgumentError run_fc(ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2)), ExponentSets([2.0], [2.0], [0.0]);
                                      stages=1:1, symmetry=Exchange(+1), logvar=LogVariable(0.0; w2=0.5), rule=FCRule(dedup=false), io=devnull)
    @test LogVariable().w2 == 1.0
end

@ts "auto_w2 (choose w₂ by the stage energy)" begin
    sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))
    r = auto_w2(sys, ExponentSets([2.0, 1.0], [1.0, 2.0], [0.0, 0.0]); n=1, kmax=2, refine=false, io=devnull)
    @test length(r.table) == 3 && Set(t.w2 for t in r.table) == Set([1.0, 0.5, 0.25])
    @test r.table[findfirst(t -> t.w2 == r.w2, r.table)].E == minimum(t.E for t in r.table)
    @test_throws ArgumentError auto_w2(sys, ExponentSets([2.0], [2.0], [0.0]); n=1, symmetry=Exchange(+1), io=devnull)
end

@ts "w2_rule (rule of thumb for w₂)" begin
    r = w2_rule(ThreeBody(("206.768262", "1.0", "7294.2996"), coulomb(-1, -1, 2)))
    @test isapprox(r.w2, 0.85 * 206.768262^(-0.25); rtol=1e-12) && r.inrange
    @test !w2_rule(ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))).inrange
    @test_throws ArgumentError w2_rule(ThreeBody(("Inf", "1", "1"), coulomb(-1, -1, 2)))
end

@ts "set_rule (sets without search)" begin
    D = ThreeBodyFC._dyadic
    r = set_rule(ThreeBody(("1", "1", "100"), coulomb(-1, -1, 2)); nsets=8)
    c = r.c
    @test r.Z == 2 && r.inrange && length(r.es.α) == 8 && r.es.α[1] == r.es.β[1] == c && r.es.γ[1] == 0
    @test r.es.α[6] == D(2√2 * c) && r.es.β[6] == D(√2 * c) && r.es.γ[6] == D(-c / √2)
    r1 = set_rule(ThreeBody(("1", "1", "100"), coulomb(-1, -1, 1)))
    @test r1.Z == 1 && r1.inrange && length(r1.es.α) == 4 && r1.es.γ[2] == D(-r1.c / 8) && r1.es.α[2] == D(r1.c / 2)
    @test set_rule(ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 2))).inrange
    D = ThreeBodyFC._dyadic
    rl = set_rule(ThreeBody(("1", "1", "0.025"), coulomb(-1, -1, 1)))
    @test rl.Z == 1 && rl.inrange && rl.family == :v7 && rl.es.α[1] == rl.c && all(rl.es.β[1:2] .== rl.c) && rl.es.γ[1] == 0 && rl.es.γ[3] == rl.c && isapprox(rl.es.γ[4], 2rl.c; atol=1e-5) && isapprox(rl.es.γ[2], rl.es.α[2]; atol=1e-5) && isapprox(rl.es.α[2], 0.59 * (40 / 18)^0.6 * rl.c; atol=1e-4)
    r5 = set_rule(ThreeBody(("1", "1", "0.3"), coulomb(-1, -1, 1)))                  # v5
    @test r5.inrange && r5.family == :v5 && r5.es.α[2] == D(√2 * r5.c) && r5.es.β[3] == D(r5.c / √2)
    r6 = set_rule(ThreeBody(("1", "1", "0.85"), coulomb(-1, -1, 1)))
    @test r6.inrange && r6.family == :v6 && r6.es.α[2] == D(r6.c / 2) && r6.es.γ[4] == D(-r6.c / (16√2))
    @test !set_rule(ThreeBody(("1", "1", "0.6"), coulomb(-1, -1, 1))).inrange
    @test !set_rule(ThreeBody(("1", "1", "1.25"), coulomb(-1, -1, 1))).inrange
    @test set_rule(ThreeBody(("1", "1", "2.5"), coulomb(-1, -1, 1))).family == :v2 && set_rule(ThreeBody(("1", "1", "2.5"), coulomb(-1, -1, 1))).inrange
    @test !set_rule(ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 4))).inrange
    @test !set_rule(ThreeBody(("1", "1", "2"), coulomb(-1, -1, 2))).inrange
    @test_throws ArgumentError set_rule(ThreeBody(("1", "1", "100"), coulomb(-1, -1, 1)); nsets=8)
    ra = set_rule(ThreeBody(("24", "1.0", "7294.2996"), coulomb(-1, -1, 2)))
    @test ra.family == :A_atomic && ra.Z == 2 && ra.inrange && all(ra.es.α .== ra.es.α[1]) && ra.es.β == [1.0, 2.0, 4.0] && isapprox(ra.w2, 0.384; atol=1e-3)
    @test set_rule(ThreeBody(("24", "1.0", "7294.2996"), coulomb(-1, -1, 3))).inrange && set_rule(ThreeBody(("24", "1.0", "250"), coulomb(-1, -1, 2))).inrange
    @test !set_rule(ThreeBody(("24", "1.0", "7294.2996"), coulomb(-1, -1, 5))).inrange
    @test !set_rule(ThreeBody(("24", "1.0", "100"), coulomb(-1, -1, 2))).inrange
    @test !set_rule(ThreeBody(("2", "1.0", "7294.2996"), coulomb(-1, -1, 2))).inrange
    @test !set_rule(ThreeBody(("206.768262", "1.0", "7294.2996"), coulomb(-1, -1, 2))).inrange
    rh = set_rule(ThreeBody(("22.0", "44.0", "1.0"), coulomb(1, 1, -1)))
    xh = 0.59 * (22 / 18)^0.6; c1h = 22 / 23
    @test rh.family == :v7 && rh.inrange && length(rh.es.α) == 4 && isapprox(rh.es.γ[2], xh * c1h; atol=1e-5) && isapprox(rh.es.γ[4], 2c1h; atol=1e-5) && rh.es.γ[1] == 0
    rn = set_rule(ThreeBody(("9.0", "13.5", "1.0"), coulomb(1, 1, -1)))
    @test rn.family == :v7 && rn.inrange && isapprox(rn.es.γ[2], 0.58 * rn.c; atol=1e-5) && rn.es.γ[1] == 0 && all(rn.es.γ[2:4] .> 0)
    @test !set_rule(ThreeBody(("6.0", "9.0", "1.0"), coulomb(1, 1, -1))).inrange
    @test !set_rule(ThreeBody(("9.0", "63.0", "1.0"), coulomb(1, 1, -1))).inrange
    @test set_rule(ThreeBody(("14.0", "84.0", "1.0"), coulomb(1, 1, -1))).family == :v7 && set_rule(ThreeBody(("14.0", "84.0", "1.0"), coulomb(1, 1, -1))).inrange   # μ_h = 12
    r20 = set_rule(ThreeBody(("20.0", "20.0", "1.0"), coulomb(1, 1, -1)); symmetry=NoSymmetry())     # μ_h = 10
    @test r20.family == :v7 && r20.inrange
    @test !set_rule(ThreeBody(("22.0", "440.0", "1.0"), coulomb(1, 1, -1))).inrange
    @test set_rule(ThreeBody(("27.58", "27.58", "1.0"), coulomb(1, 1, -1)); symmetry=NoSymmetry()).family == :v7
    @test set_rule(ThreeBody(("27.58", "27.58", "1.0"), coulomb(1, 1, -1))).family == :v7
end

@ts "auto_form (compare function forms at the same Meff)" begin
    sys = ThreeBody(("1", "1", "Inf"), coulomb(-1, -1, 1))
    r = auto_form(sys, default_exponents(sys); n=2, io=devnull)
    @test length(r.table) == 3 && r.table[3].rule.collect == :st && r.best.logvar.split && r.best.rule.collect != :st
    @test [s[2] for s in r.table[2].stages] == [s[2] for s in r.table[1].stages]
    @test r.table[2].stages[end][3] < r.table[1].stages[end][3]
    r2 = auto_form(sys, ExponentSets([1.0, 1.0], [1.0, 0.25], [0.0, 0.0]); n=1, symmetry=NoSymmetry(), io=devnull,
                   forms=[("log1", LogVariable(), FCRule()), ("w₂=1/2", LogVariable(0.0; w2=0.5), FCRule(dedup=false))])
    E1 = r2.table[1].stages[end][3]; E2 = r2.table[2].stages[end][3]
    @test length(r2.table) == 2 || E2 < E1
    for k in 3:length(r2.table)
        @test r2.table[k].logvar.w2 == r2.table[k-1].logvar.w2 / 2
        Ek = [t.stages[end][3] for t in r2.table[2:k-1]]
        @test Ek[end] == minimum(Ek) && Ek[end] < E1
    end
    Ew = [t.stages[end][3] for t in r2.table[2:end]]
    @test r2.best.logvar.w2 == (minimum(Ew) < E1 ? r2.table[1 + argmin(Ew)].logvar.w2 : 1.0)
end

@ts "kernel quadrature peak (_kq_range bisection + window) equals a full scan" begin
    I = ThreeBodyFC.Integrals
    function brute(tb, k1, Nu, Nv, Nw, cut)
        Emax = -Inf; imax = 1
        for i in eachindex(tb.s64)
            E = I._kq_E(tb, i, k1, Nu, Nv, Nw); E > Emax && (Emax = E; imax = i)
        end
        lo = imax; hi = imax
        while lo > 1 && I._kq_E(tb, lo - 1, k1, Nu, Nv, Nw) >= Emax - cut; lo -= 1; end
        while hi < length(tb.s64) && I._kq_E(tb, hi + 1, k1, Nu, Nv, Nw) >= Emax - cut; hi += 1; end
        (Emax, lo, hi, imax)
    end
    nbad = 0; ntot = 0
    for (eng, T) in ((KernelQuadrature(), Float64), (KernelQuadrature(BigFloat), BigFloat)), (ρu, ρv, ρw, cw) in ((1.0, 2.0, 3.0, 1.0), (0.05, 7.5, 0.3, 0.5), (40.0, 1e-3, 2.0, 2.0)), level in (0, 4)
        tb = I._kq_table(eng, T(ρu), T(ρv), T(ρw), T(cw), level)
        for k1 in (1, 2, 5, 17), Nu in (0, 1, 4, 13), Nv in (0, 2, 9), Nw in (0, 1, 6)
            ntot += 1
            I._kq_range(tb, k1, Nu, Nv, Nw, eng.cut) === brute(tb, k1, Nu, Nv, Nw, eng.cut) || (nbad += 1)
        end
    end
    @test ntot == 2 * 3 * 2 * 4 * 4 * 3 * 3 && nbad == 0
end

@ts "local order _local_order" begin
    st = [(k, M, big(-1) + big(M)^-5) for (k, M) in enumerate((100, 200, 400))]
    @test ThreeBodyFC._local_order(st) ≈ 5 atol = 1e-12
    @test isnan(ThreeBodyFC._local_order(st[1:2]))
    @test isnan(ThreeBodyFC._local_order([(1, 10, big(-1.0)), (2, 20, big(-1.0)), (3, 40, big(-1.0))]))
end

@ts "localization _peak_power" begin
    @test ThreeBodyFC._peak_power(3.9, 1.0, 0.0) == 0
    @test ThreeBodyFC._peak_power(6.0, 1.0, 0.0) == 1 == ThreeBodyFC._peak_power(6.0, 2.5, 0.0)
    @test ThreeBodyFC._peak_power(2.0739^2 / 0.07296, 1.0, 0.0) == 28
    @test ThreeBodyFC._peak_power(4.09, 2.0, 0.0) == 1
    @test ThreeBodyFC._peak_power(5.0, 1.0, -1.5) === nothing                        # 2γ + 2α ≤ 0
end

@ts "gauss_legendre01 terminates above 1086 bits" begin
    for bits in (256, 1088)
        x, w = setprecision(BigFloat, bits) do
            ThreeBodyFC.Integrals.gauss_legendre01(BigFloat, 24)
        end
        e = setprecision(BigFloat, bits) do
            maximum(abs(sum(w .* x .^ k) - 1 / BigFloat(k + 1)) for k in 0:47)
        end
        @test exponent(e + big(2.0)^-100000) < 8 - bits
    end
end
