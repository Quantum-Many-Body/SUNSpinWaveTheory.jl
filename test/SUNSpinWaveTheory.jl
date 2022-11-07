using SUNSpinWaveTheory
using Plots: plot, plot!, savefig
using QuantumLattices: Lattice, Hilbert, Algorithm, ReciprocalPath, @rectangle_str, atol, ReciprocalZone, Segment, reciprocals
using TightBindingApproximation: EnergyBands
using QuantumLattices: Fock, @σ_str, Onsite, expand, bonds, MatrixCoupling, FID
using StaticArrays: @SVector
using Test
using Optim: ConjugateGradient

@time @testset "SquareFM" begin
    lattice = Lattice(
        [0.0, 0.0];
        vectors=[[1.0, 0.0], [0.0, 1.0]]
    )
    σx = [0 1; 1 0]
    σy = [0 -im; im 0]
    σz = [1 0; 0 -1]
    Jmat = -1/4*(kron(σx, σx) + kron(σy, σy) + kron(σz, σz))
    hmat = -0.1*(σx)/2
    J = SUNTerm(:J, Jmat, 2, 2, 1)
    h = SUNTerm(:h, hmat, 2, 2, 0)
    hilbert = Hilbert(pid=>Fock{:b}(2, 1) for pid in 1:length(lattice))

    magneticstructure = MagneticStructure(lattice,
        Dict(pid=>[pi/4,pi/2] for pid in 1:length(lattice))
        )
    eng = SUNLSWT(lattice, hilbert, (J, h), magneticstructure);
    #optimize the ground state
    op = optimorder(eng; method = ConjugateGradient(), numrand = 2);
    @test isapprox(op[2].minimum , -0.55, atol=atol)
    # order parameters
    px = Dict(pid => σx for pid in 1:length(lattice))
    py = Dict(pid => σy for pid in 1:length(lattice))
    pz = Dict(pid => σz for pid in 1:length(lattice))
    @test isapprox(real(localorder(op[1], σx)[1]), 1.0, atol=atol) 
    x, y, z = localorder(op[1], px)[1], localorder(op[1], py)[1], localorder(op[1], pz)[1]
    @test isapprox(real(x), 1.0, atol=atol)
    @test isapprox(abs2(x) + abs2(y) + abs2(z), 1.0, atol=atol)
    
    sunlswt = Algorithm(:SquareFM, op[1]);
    path = ReciprocalPath(reciprocals(lattice), rectangle"Γ-X-M-Γ", length=8)
    data = sunlswt(:EBS, EnergyBands(path))[2].data[2]
    A(; k) = 2-cos(k[1])-cos(k[2]) + 0.1
    for (i, params) in enumerate(pairs(path))
        @test isapprox(A(; params...), data[i, 1], atol=atol)
        @test isapprox(A(; params...), data[i, 2], atol=atol)
    end

    path = ReciprocalPath(reciprocals(lattice), rectangle"Γ-X-M-Γ", length=100)
    ebs = sunlswt(:EBS, EnergyBands(path))
    mx, my, mz = MatrixCoupling(:, FID, σ"x", :, :), MatrixCoupling(:, FID, σ"y", :, :), MatrixCoupling(:, FID, σ"z", :, :)
    sx = expand(Onsite(:mx, 0.5+0.0im, mx, amplitude=nothing, modulate=true), bonds(lattice, 0), hilbert, half=false)
    sy = expand(Onsite(:my, 0.5+0.0im, my, amplitude=nothing, modulate=true), bonds(lattice, 0), hilbert, half=false)
    sz = expand(Onsite(:mz, 0.5+0.0im, mz, amplitude=nothing, modulate=true), bonds(lattice, 0), hilbert, half=false)
    ss = @SVector [sx, sy, sz]
    ins = sunlswt(
        :INS, 
        Spectra{InelasticNeutron}(
            path, 
            range(0.0, 5.0, length=501),
            (ss, ss); 
            fwhm=0.1, 
            scale=log
        )
    )
    plt = plot()
    plot!(plt, ins)
    plot!(plt, ebs, color=:red, linestyle=:dash)
    display(plt)
    savefig("squareFM_INS_Spectra.png")
    #spin-spin correlation
    insmultipole = sunlswt(:Multipole,
                    Spectra{Multipole}(path, range(0.0, 5.0, length=501), (ss,ss); fwhm=0.1, scale=log)
                    )
    plt1 = plot()
    plot!(plt1, insmultipole, [-1.0, -0.1])
    plot!(plt1, ebs, color=:red, linestyle=:dash)
    display(plt1)
    savefig("squareFM_Multipole_spectra.png")
    #INS of fixed energy 
    nx, ny = 16, 16
    zone = ReciprocalZone(reciprocals(lattice), Segment(0, +1, nx), Segment(-0//2, +2//2, ny))
    inszone = sunlswt(
                :INSZ,
                Spectra{InelasticNeutron}(
                    zone, 
                    range(0.0, 5.0, length=501), 
                    (ss, ss); 
                    fwhm=0.1,
                    gauss=false,
                    scale=log
                )
            )
    ecut, dE = 1.0, 0.1
    plt2 = plot()
    plot!(plt2, inszone, ecut, dE)
    display(plt2)
    savefig("squareFM_Ecut.png")
end
