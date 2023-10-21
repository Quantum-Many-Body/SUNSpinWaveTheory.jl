```@meta
CurrentModule = SUNSpinWaveTheory
```

# SU(2) antiferromagnetic Heisenberg modle on square lattice  

## Magnon bands, inelastic neutron spectra and multipole spctra by SU(N) linear spin wave theory

The following codes could compute the spin wave dispersions of the antiferromagnetic Heisenberg model on the square lattice.

```@example SU(2)AFM
using SUNSpinWaveTheory
using Plots: plot, plot!
using QuantumLattices: Lattice, Point, Hilbert, Algorithm, ReciprocalPath, @rectangle_str, atol,ReciprocalZone
using TightBindingApproximation: EnergyBands
using QuantumLattices: Fock, @σ_str, Onsite, expand, bonds, reciprocals, MatrixCoupling, FID
using StaticArrays: @SVector
using Optim:  ConjugateGradient

#define square lattice and magnetic unit cell
lattice = Lattice(
     [0.0, 0.0];
    vectors=[[1.0, 0.0], [0.0, 1.0]]
    )

cell = Lattice(
    [0.0, 0.0],
    [1.0, 0.0];
    vectors=[[1.0, 1.0], [1.0, -1.0]]
    )

#define antiferromagnetic Heisenberg model with magnetic field
σx = [0 1; 1 0]
σy = [0 -im; im 0]
σz = [1 0; 0 -1]
j, B = 1.0, -0.01
Jmat = 1/4*(kron(σx,σx) + kron(σy,σy) + kron(σz,σz))
hmat = σy/2
J = SUNTerm(:J, Jmat, 2, 2, 1; value=j)
h = SUNTerm(:h, hmat, 2, 2, 0; amplitude=( x-> iseven(x[1].site) ? -1 : 1) , value=B )

hilbert = Hilbert(pid=>Fock{:b}(2, 1) for pid in 1:length(cell))

magneticstructure = MagneticStructure(
    cell,
    Dict(pid=>(iseven(pid) ?  Float64[pi/2, pi/2] : Float64[0., 0.] ) for pid in 1:length(cell))
    )

#initialization of SUNLSWT engine
eng = SUNLSWT(lattice, hilbert, (J, h), magneticstructure)

#optimize the ground state with four random steps
op = optimorder2(eng; method = ConjugateGradient(), numrand = 4);

#the classical energy
println("The classical energy ≈ -1.01 :",isapprox(op[2].minimum, -1.01, atol=atol))

antiferromagnet = Algorithm(:SquareAFM, op[1] )

#calculate the order parameters
px = Dict(pid => σx for pid in 1:length(cell))
py = Dict(pid => σy for pid in 1:length(cell))
pz = Dict(pid => σz for pid in 1:length(cell))

println("σx on each site: ", localorder(op[1], px))
println("σy on each site: ", localorder(op[1], py))
println("σz on each site: ", localorder(op[1], pz))
```

### Magnon band structure
```@example SU(2)AFM
path = ReciprocalPath(reciprocals(lattice), rectangle"Γ-X-M-Γ", length=100)
mx, my, mz = MatrixCoupling(:, FID, σ"x", :, :), MatrixCoupling(:, FID, σ"y", :, :), MatrixCoupling(:, FID, σ"z", :, :)
sx = expand(Onsite(:mx, 0.5+0.0im, mx,amplitude=nothing, modulate=true), bonds(cell,0), hilbert, half=false)
sy = expand(Onsite(:my, 0.5+0.0im, my,amplitude=nothing, modulate=true), bonds(cell,0), hilbert, half=false)
sz = expand(Onsite(:mz, 0.5+0.0im, mz,amplitude=nothing, modulate=true), bonds(cell,0), hilbert, half=false)
ss = @SVector [sx,sy,sz]

ins = antiferromagnet(:INS,
    Spectra{InelasticNeutron}(path, range(0.0, 2.5, length=251), (ss,ss); fwhm=0.1, scale=log, gauss=false)
    )
energybands = antiferromagnet(:EB, EnergyBands(path))
plt = plot()
plot!(plt, ins)
plot!(plt, energybands, color=:red, linestyle=:dash)
#display(plt)
```
### Inelastic neutron spectra
```@example SU(2)AFM
insmultipole = antiferromagnet(:Multipole,
    Spectra{Multipole}(path, range(0.0, 2.5, length=251), (ss,ss); fwhm=0.1, scale=log)
    )
plt1 = plot()
plot!(plt1, energybands, color=:red, linestyle=:dash)
plot!(plt1, insmultipole)
#display(plt1)
```
### Inelastic neutron spectra with E=1.0
```@example SU(2)AFM
nx, ny = 32, 32
zone = ReciprocalZone(reciprocals(lattice), 0=>1, 0=>1; length=(nx, ny), ends=((true, true), (true, true)))
inszone = antiferromagnet(:INSZ,
    Spectra{InelasticNeutron}(zone, range(0.0, 2.5, length=251), (ss,ss); fwhm=0.1, scale=log)
    )
ecut,dE = 1.0, 0.1
# data = spectraEcut(inszone[2], ecut, dE)
plt2 = plot()
plot!(plt2, inszone, ecut, dE)
#display(plt2)
```