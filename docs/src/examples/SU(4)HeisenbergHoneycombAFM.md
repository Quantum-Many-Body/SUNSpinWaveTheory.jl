```@meta
CurrentModule = SUNSpinWaveTheory
```

# SU(4) antiferromagnetic Heisenberg modle on honeycomb lattice 

## Magnon bands and inelastic neutron spectra by SU(4) linear spin wave theory

The following codes could compute the flavor wave dispersions of the SU(4) antiferromagnetic Heisenberg model, i.e. ``H=J∑\\_{⟨ij⟩}[∑\\_{1<=a<b<=5} Γᵢᵃᵇ Γⱼᵃᵇ - ∑ₐΓᵢᵃΓⱼᵃ]`` :


```@example SU(4)AFM
using SUNSpinWaveTheory
using Plots: plot, plot!
using QuantumLattices: Lattice, Point, Hilbert, Algorithm, ReciprocalPath, atol,ReciprocalZone, Segment
using TightBindingApproximation: EnergyBands
using QuantumLattices: Fock, @σ_str, Onsite, expand, bonds, MatrixCoupling, FID, reciprocals, @hexagon_str
using StaticArrays: @SVector
using QuantumLattices: @indexes, Index, Coupling, Coulomb

lattice = Lattice(
    [0.0, 0.0], [1/(sqrt(3)),0.0];
    vectors = [[sqrt(3)/2, -0.50], [0.0, 1.0]],
    )
cell = lattice

#construct the Hamiltonian
σx = [0 1; 1 0]
σy = [0 -im; im 0]
σz = [1 0; 0 -1]
σ0 = [1 0; 0  1]
Γ¹ = kron(σz, σy); Γ² = kron(σz, σx); Γ³ = kron(σy, σ0);Γ⁴ = kron(σx, σ0);Γ⁵ = kron(σz, σz);

#H=J∑\_{⟨ij⟩}[∑\_{1<=a<b<=5}Γᵢᵃᵇ Γⱼᵃᵇ-∑ₐΓᵢᵃΓⱼᵃ] 
Γ(x,y) = (x*y-y*x)/2im
Γ¹²=(Γ¹*Γ²-Γ²*Γ¹)/2im; Γ¹³=Γ(Γ¹,Γ³);Γ¹⁴=Γ(Γ¹,Γ⁴);Γ¹⁵=Γ(Γ¹,Γ⁵)
Γ²³=Γ(Γ²,Γ³);Γ²⁴=Γ(Γ²,Γ⁴); Γ²⁵=Γ(Γ²,Γ⁵)
Γ³⁴=Γ(Γ³,Γ⁴);Γ³⁵=Γ(Γ³,Γ⁵); Γ⁴⁵=Γ(Γ⁴,Γ⁵)
Γab = [Γ¹²,Γ¹³,Γ¹⁴,Γ¹⁵,Γ²³,Γ²⁴,Γ²⁵,Γ³⁴,Γ³⁵,Γ⁴⁵];
Γa = [Γ¹, Γ² , Γ³ ,Γ⁴ ,Γ⁵ ]

j, h = 1.0, 0.001
Jmat = zeros(ComplexF64,16,16)

for m in Γab
    Jmat[:,:] += kron(m, m)
end
for m in Γa
    Jmat[:,:] -= kron(m, m)
end
hmat = Γ¹²
f1(bond) = iseven(bond[1].site) ? 1 : -1

J = SUNTerm(:J, Jmat, 4, 4, 1; value=j)
J1 = SUNTerm(:B, hmat, 4, 4, 0; value=h, amplitude=f1)

hilbert = Hilbert(pid=>Fock{:b}(4, 1) for pid in 1:length(cell))

magneticstructure = MagneticStructure(
    cell,
    Dict(1 => [1.5707960439947803, 1.5707963267641, 1.570796326798081, 3.1432634733226568, 3.1430008310022868, 4.2168412327339055],
         2 => [2.8285525202602516e-7, 1.5708000153983595, 1.5707999995656055, 3.1429367971485704, 3.1430659136725807, 1.075295154664448],
    )
)
eng = SUNLSWT(lattice, hilbert, (J, J1), magneticstructure)

#start optimize:
op = optimorder(eng; numrand = 1);

#the classical energy
println(expand(op[1].Ω).contents[()].value)

antiferromagnet = Algorithm(:SquareAFM, op[1] )

# order parameters
px = Dict(pid => Γ¹² for pid in 1:length(cell))
py = Dict(pid => Γ³⁴ for pid in 1:length(cell))
pz = Dict(pid => Γ⁵  for pid in 1:length(cell))
orderpara = [localorder(antiferromagnet.frontend, px),localorder(antiferromagnet.frontend, py),localorder(antiferromagnet.frontend, pz)]
println(orderpara)
```

### Band structure
```@example SU(4)AFM
sx = [0       sqrt(3)/2      0      0;
     sqrt(3)/2   0            1.0    0;
       0        1.0           0     sqrt(3)/2;
       0         0           sqrt(3)/2  0
]
mx = MatrixCoupling(:, FID, sx, :, :)
sy = [0       -im*sqrt(3)/2      0      0;
     im*sqrt(3)/2   0            -1im    0;
       0        1im           0     -im*sqrt(3)/2;
       0         0           im*sqrt(3)/2  0
]
my = MatrixCoupling(:, FID, sy, :, :)
sz = [3/2   0       0      0;
       0    1/2     0      0;
       0    0       -1/2   0;
       0    0       0     -3/2
]
mz = MatrixCoupling(:, FID, sz, :, :)

path = ReciprocalPath(reciprocals(lattice), hexagon"Γ-K-M-Γ,60°", length=100)
sx = expand(Onsite(:mu, 1.0+0.0im, mx,amplitude=nothing, modulate=true),bonds(cell,0),hilbert,half=false)
sy = expand(Onsite(:mu, 1.0+0.0im, my,amplitude=nothing, modulate=true),bonds(cell,0),hilbert,half=false)
sz = expand(Onsite(:mu, 1.0+0.0im, mz,amplitude=nothing, modulate=true),bonds(cell,0),hilbert,half=false)
ss = @SVector [sx,sy,sz]

ins = antiferromagnet(:INS,
    Spectra{InelasticNeutron}(path, range(0.0, 20, length=501), (ss,ss); fwhm=0.2, scale=identity, gauss=false, atol=1e-9)
    )
energybands = antiferromagnet(:EB, EnergyBands(path; atol=1e-9))

plt = plot()
plot!(plt, ins)
plot!(plt, energybands, color=:white, linestyle=:dash, linealpha=0.1, xticks=([0,100,200,300],["Γ","K","M","Γ"]))
#display(plt)
```
### Spectra of multipole operators 
```@example SU(4)AFM
#multipole
quadrupolar = [multipoleOperator(3//2,3//2,2,m) for m=-2:2]
Mcouplings = [MatrixCoupling(:, FID, i, :, :) for i in quadrupolar]   
M₋₂ = expand(Onsite(:mu1, 1.0+0.0im, Mcouplings[1],amplitude=nothing, modulate=true), bonds(cell,0), hilbert, half=false)
M₋₁ = expand(Onsite(:mu2, 1.0+0.0im, Mcouplings[2],amplitude=nothing, modulate=true), bonds(cell,0), hilbert, half=false)
M₀ = expand(Onsite(:mu3,  1.0+0.0im, Mcouplings[3],amplitude=nothing, modulate=true), bonds(cell,0), hilbert, half=false)
M₁ = expand(Onsite(:mu4,  1.0+0.0im, Mcouplings[4],amplitude=nothing, modulate=true), bonds(cell,0), hilbert, half=false)
M₂ = expand(Onsite(:mu5,  1.0+0.0im, Mcouplings[5],amplitude=nothing, modulate=true), bonds(cell,0), hilbert, half=false)
Mss = [M₋₂, M₋₁, M₀, M₁, M₂]
Mssd = [M₋₂', M₋₁', M₀', M₁', M₂']

insmultipole = antiferromagnet(:Multipole,
    Spectra{Multipole}(path, range(0.0, 20, length=501), (Mss, Mssd); fwhm=0.2, gauss=true, scale=x->x, atol=1e-9)
    )
plt1 = plot()
plot!(plt1, insmultipole, clims=(0, 100))
plot!(plt1, energybands, color=:white, linestyle=:dash, linealpha=0.1,xticks=([0,100,200,300],["Γ","K","M","Γ"]))
#display(plt1)
```
### Spectra of multipole operators with E=1.0
```@example SU(4)AFM
#Ecut
nx, ny = 16, 16
zone = ReciprocalZone(reciprocals(lattice),Segment(0, +1, nx),Segment(-0//2, +2//2, ny))
inszone = antiferromagnet(:MultipoleZone,
    Spectra{Multipole}(zone, range(0.0, 20, length=501), (Mss, Mssd); fwhm=0.2, gauss=true, scale=identity, atol=1e-9)
    )
ecut,dE = 3.0, 0.1
# data = spectraEcut(inszone[2], ecut, dE, nx, ny)
plt2 = plot()
plot!(plt2, inszone, ecut, dE)
#display(plt2)
```