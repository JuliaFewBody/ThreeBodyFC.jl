using ThreeBodyFC: _rat, permute, exchange_order, _inverse, _exchange_invariant

@testset "particle reordering (identical pair to internal 1, 2)" begin
    @test _rat("1") == _rat("1.0") == _rat("10/10") == _rat("1e0")
    @test _rat("9.88e1") == 988 // 10
    @test _rat("-0.25") == -1 // 4
    @test _rat("Inf") === nothing

    psA = ThreeBody(("1", "1.0", "1"), coulomb(1, -1, -1))
    π = exchange_order(psA)
    @test π !== nothing && sort([π[1], π[2]]) == [2, 3] && π[3] == 1
    @test _exchange_invariant(permute(psA, π))
    @test !_exchange_invariant(psA)
    es = ExponentSets([0.5, 0.25], [0.5, 0.125], [0.0, 0.1])
    @test permute(permute(es, π), _inverse(π)).α == es.α
    @test permute(permute(es, π), _inverse(π)).γ == es.γ
    back = permute(permute(psA, π), _inverse(π))
    @test back.masses == psA.masses && [t.pow for t in back.terms] == [t.pow for t in psA.terms]

    psB = ThreeBody(("1", "1", "1"), coulomb(-1, -1, 1))
    esB = ExponentSets([0.5, 0.5], [0.5, 0.25], [0.0, 0.0])
    rB = run_fc(psB, esB; stages=1:2, symmetry=Exchange(+1), io=devnull, threshold=nothing)
    rA = run_fc(psA, permute(esB, _inverse(π)); stages=1:2, symmetry=Exchange(+1), io=devnull, threshold=nothing)
    @test rA[end].Msolved == rB[end].Msolved
    @test abs(Float64(rA[end].E - rB[end].E)) < 1e-40

    @test [x.M for x in count_basis(psA; stages=1:3, io=devnull)] == [x.M for x in count_basis(psB; stages=1:3, io=devnull)]
    dA = run_fc(psA; stages=1:2, io=devnull); dB = run_fc(psB; stages=1:2, io=devnull)
    @test dA[end].order == π && dB[end].order == (1, 2, 3)
    @test abs(Float64(dA[end].E - dB[end].E)) < 1e-40

    muhe = ThreeBody(("206.768262", "1.0", "7294.2996"), coulomb(-1, -1, 2))
    @test exchange_order(muhe) === nothing
    @test_throws ArgumentError run_fc(muhe, ExponentSets([402.0], [2.0], [0.0]); stages=1:1, symmetry=Exchange(+1), io=devnull)
end
