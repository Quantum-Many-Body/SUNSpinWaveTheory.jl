module SUNSpinWaveTheory
using Printf: @sprintf
using RecipesBase: RecipesBase, @recipe, @series
using TimerOutputs: @timeit
using StaticArrays: SVector, SMatrix, @SMatrix
using LinearAlgebra: Hermitian, Diagonal, dot, eigen, norm, I, kron, diag
using QuantumLattices: ID, CompositeIndex, Operator, Operators, UnitSubstitution, RankFilter,  OperatorGenerator, Image, Action, Algorithm, Assignment
using QuantumLattices: AbstractLattice, bonds, Index, FID, Table, Hilbert, Fock, Term, Boundary, ReciprocalPath, ReciprocalZone, ReciprocalSpace
using QuantumLattices: atol, rtol, dtype, indextype, fulltype, idtype, reparameter, sub!, mul!, expand, plain, rcoordinate, icoordinate, delta, decimaltostr
using QuantumLattices: Coulomb, Onsite, matrix, dimension, Neighbors, Coupling, ishermitian, MatrixCoupling
using QuantumLattices: OperatorUnitToTuple
using TightBindingApproximation: TBAKind, AbstractTBA, TBAMatrixRepresentation
using Optim: LBFGS, optimize, Fminbox, Options
using WignerSymbols: clebschgordan

import QuantumLattices: optype, contentnames, update!, add!, run!, Metric, initialize
import TightBindingApproximation: commutator

export MagneticStructure, HPTransformation, suncouplings, SUNTerm, localorder, matrixcoef, spectraEcut, multipoleOperator
export SUNLSWT, Spectra, rotation_gen, rotation, @rotation_gen, optimorder, optimorder2, fluctuation
export SpectraKind, InelasticNeutron, Multipole, @f2couplings, kdosOperator

macro f2couplings(ex, ex1, ey1)
    res = :(+())
    @assert ex.head == :vcat "f2couplings error: the head of first Expr $(ex.head) == :vcat"
    for i in 1:length(ex.args)
        expr1 = (Expr(:ref, (ex1), i))
        for (j, arg1) in enumerate(ex.args[i].args) 
            expr2 = Expr(:ref, ey1, j)
            expr3 = @eval(Main, Expr(:call, :*, $expr1, $expr2 ))
            push!(res.args, Expr(:call, :*, arg1, expr3 ) )
        end
    end
    res1 = res
        return @eval(Main, $res1)
end
"""
    suncouplings(J₁₂₃₄::Array{<:Number,4}) -> OperatorSum

Obtain the Couplings for exchange term in Hamiltionian. Like ``∑(J_{i1,i2;j3,j4}b†_{i1}b_{i2}b†_{j3}b_{j4}) ``
"""
function suncouplings(Jᵤᵥᵢⱼ::Array{<:Number, 4})
    fc = []
    n, m, l, k = size(Jᵤᵥᵢⱼ)    
    for j = 1 : k
        for i = 1 : l
            for v = 1 : m
                for u = 1 : n
                    cp = Coupling(Jᵤᵥᵢⱼ[u, v, i, j], (1, 1, 2, 2), FID, (u, v, i, j), (0, 0, 0, 0), (2, 1, 2, 1))
                    push!(fc,  cp)
                end
            end 
        end
    end
    return sum(fc)
end
"""
    suncouplings(Bᵤᵥ::Array{<:Number, 2})   -> MatrixCoupling
    
Obtain the Couplings for onsite term in Hamiltionian. Like ∑(Bᵢᵤᵥb†ᵢᵤbᵢᵥ )
"""
function suncouplings(Jmat::Array{<:Number, 2})  
    return MatrixCoupling(:, FID, Jmat, :, :)
end

"""
    SUNTerm(
        id::Symbol, 
        J₁₂₃₄::Array{<:Number, 2}, 
        dimᵢ::Int, 
        dimⱼ::Int, 
        bondkind::Int=1;
        value=1, 
        amplitude::Union{Function, Nothing}=nothing, 
        modulate::Union{Function, Bool}=false
    ) -> Term

Return the `Term` type. `J₁₂₃₄` is the coefficience of Hamiltionian including the exchange term and onsite term, i.e. ``∑(J_{i1,j2;i3,j4}b†_{i1}b†_{j2}b_{i3}b_{j4}) + ∑(Bᵢᵤᵥb†ᵢᵤbᵢᵥ )``
`dimᵢ` and `dimⱼ` are the dimensions of local Hilbert space on sites i and j, respectively.
"""
function SUNTerm(id::Symbol, J₁₂₃₄::Array{<:Number, 2}, dimᵢ::Int, dimⱼ::Int, bondkind::Int=1;
        value=1, amplitude::Union{Function, Nothing}=nothing, modulate::Union{Function, Bool}=false)    
    if bondkind == 0
        return SUNTerm(id, J₁₂₃₄, 0; value=value, amplitude=amplitude, modulate=modulate)
    else
        J = permutedims(reshape(J₁₂₃₄, (dimᵢ, dimⱼ, dimᵢ, dimⱼ)), [1, 3, 2, 4]  )   
        return  SUNTerm(id, J, bondkind; value=value, amplitude=amplitude, modulate=modulate)
    end
end
"""
    SUNTerm(
        id::Symbol, 
        J₁₂₃₄::Array{<:Number, 4}, 
        bondkind::Int=1;
        value = 1,
        amplitude::Union{Function, Nothing}=nothing, 
        modulate::Union{Function, Bool}=false
    ) -> Coulomb

Hamiltionian: ``∑(J_{i1,i2;j3,j4}b†_{i1}b_{i2}b†_{j3}b_{j4}) ``
"""
function SUNTerm(id::Symbol, J₁₂₃₄::Array{<:Number, 4}, bondkind::Int=1;
                value = 1,
                amplitude::Union{Function, Nothing}=nothing, 
                modulate::Union{Function, Bool}=false)  
    couplings = suncouplings(J₁₂₃₄)
    datatype = eltype(J₁₂₃₄)
    return  Coulomb(id, value*one(datatype), bondkind, couplings; ishermitian=true, amplitude=amplitude, modulate=modulate)
end

"""
    SUNTerm(
        id::Symbol, 
        Jᵤᵥ::Array{<:Number, 2}, 
        bondkind::Int=0;
        value = 1, 
        amplitude::Union{Function, Nothing}=nothing,
        modulate::Union{Function, Bool}=false
        ) -> Onsite
"""
function SUNTerm(id::Symbol, Jᵤᵥ::Array{<:Number, 2}, bondkind::Int=0;
                value = 1, 
                amplitude::Union{Function, Nothing}=nothing,
                modulate::Union{Function, Bool}=false)  
    @assert bondkind == 0 "sunterms error: bondkind ($bondkind) is 0"  
    @assert ishermitian(Jᵤᵥ) == true "sunterms error: the onsite matrix should be hermitian."
    couplings = MatrixCoupling((1, 1), FID, Jᵤᵥ, :, :)
    datatype = eltype(Jᵤᵥ)
    return  Onsite(id, value*one(datatype), couplings; ishermitian=true, amplitude=amplitude, modulate=modulate)
end

###
"""
    rotation(sitapsi::T, generators::Vector{Matrix{Complex{Int}}}}) where T<:AbstractVector{<:Number} -> Matrix{ComplexF64}

Get the rotation matrix which rotates the `|0⟩` state to the target state |T⟩=U(θ₁,θ₂,...,ψ₁,ψ₂,...)b⁰† |0⟩; T†ᵥ = ∑ᵤUᵥᵤ*bᵘ†. The detail is given in doi:10.1103/PhysRevB.97.205106.
"""   
function rotation(sitapsi::AbstractVector{T}, generators::Vector{Matrix{Complex{Int}}}) where T<:Number
    @assert iseven(length(sitapsi)) "rotation error: sitapsi vector must be even-dimensional."
    n = length(sitapsi)÷2
    @assert 2*n == length(generators) "rotation error: the number of generators vector ($(length(generators))) must be $(2*n)"
    sita = sitapsi[1 : n]
    psi = sitapsi[n + 1 : end]
    X = [ sita[1] for i = 1 : 2*n ]
    for i = 1:n-1
        X[ 2*i - 1] = X[ 2*i - 1 ]*cos(sita[i + 1])
        X[ 2*i ] = X[2*i]*cos(sita[i + 1])
    end
    for i = 0 : n - 1
        X[2*i + 1] = X[2*i + 1]*cos(psi[i + 1])
        X[2*i + 2] = X[2*i + 2]*sin(psi[i + 1])
        for j = 1 + i : n - 1
            X[2*j + 1] = X[2*j + 1]*sin(sita[i + 2])
            X[2*j + 2] = X[2*j + 2]*sin(sita[i + 2])
        end
    end
    datatype = promote_type(T, eltype(eltype(generators)))
    M = zeros(datatype, (n + 1, n + 1))  
    for i = 1 : 2*n
        M .+= generators[i]*X[i]
    end
    return exp(1im*Hermitian(M))
end
"""
    rotation_gen(n::Int) -> Vector{Matrix{Complex{Int}}}

Return the generator matrices of SU(n) group.
"""
function rotation_gen(n::Int)
    A = Matrix{Complex{Int}}[]
    for i = 1 : n-1
        m = Complex{Int}[0 for i1 = 1 : n, j1 = 1 : n]
        m[1, 1 + i] = 1
        m[i + 1, 1] = 1
        push!(A,m)
        m = Complex{Int}[0 for i1 = 1 : n, j1 = 1 : n]
        m[1, 1 + i] = -1im
        m[i + 1, 1] = 1im
        push!(A,m)
    end
    return A
end
"""
    @rotation_gen ::Int -> Vector{Matrix{Complex{Int}}}

Return the generator matrices of SU(n) group.
"""
macro rotation_gen(n0)
    n=eval(n0)
    @assert isa(n,Int) "@rotation_gen error: input must be a integer"
    A = Matrix{Complex{Int}}[]
    for i = 1 : n-1
        m = Complex{Int}[0 for i1 = 1 : n, j1 = 1 : n]
        m[1, 1 + i] = 1
        m[i + 1, 1] = 1
        push!(A,m)
        m = Complex{Int}[0 for i1 = 1:n, j1 = 1:n]
        m[1, 1 + i] = -1im
        m[i + 1, 1] = 1im
        push!(A, m)
    end
    return A
end 
           
"""
    MagneticStructure{L<:AbstractLattice, P<:Int, D<:Number, T<:Complex}

The magnetic structure of an ordered quantum lattice system. 
"""
struct MagneticStructure{L<:AbstractLattice, P<:Int, D<:Number, T<:Complex}
    cell::L
    moments::Dict{P, Vector{D}}
    rotations::Dict{P, Matrix{T}}
end
"""
    MagneticStructure(cell::AbstractLattice, moments::Dict{<:Int, <:AbstractVector})
    (ms::MagneticStructure)(orders::Dict{<:Int, <:AbstractMatrix}) -> Dict{<:Int, <:Number} 

1).Construct the magnetic structure on a given lattice with the given moments. The moment is given by the angles (θ₁,θ₂,...,ψ₁,ψ₂,...). 
2).Get order parameters for every site. e.g. |gs⟩ = ms.rotations[i][1,:] where gs = classical ground state; ⟨σᵢˣ⟩ = ⟨gs|σᵢˣ|gs⟩.
"""
function MagneticStructure(cell::AbstractLattice, moments::Dict{<:Int, <:AbstractVector})
    @assert length(cell)==length(moments) "MagneticStructure error: mismatched magnetic cell and moments."
    datatype = promote_type(dtype(cell), eltype(valtype(moments)))
    moments = convert(Dict{keytype(moments), Vector{datatype}}, moments)
    datatype2 = promote_type(eltype(valtype(moments)), Complex{Int})
    rotations = Dict{keytype(moments), Matrix{datatype2}}()
    for pid in 1:length(cell)
        n = length(moments[pid])÷2 + 1
        gen_mat = rotation_gen(n)
        u = rotation(moments[pid], gen_mat)
        u[norm.(u) .< atol] .= 0.0 # omit the small quantities.
        rotations[pid] = u
    end
    return MagneticStructure(cell, moments, rotations)
end

function (ms::MagneticStructure)(orders::Dict{<:Int, <:AbstractMatrix})
    @assert length(ms.cell)==length(orders) "MagneticStructure error: mismatched magnetic cell and orders."
    datatype = promote_type(eltype(valtype(ms.rotations)), eltype(valtype(orders)))
    orders = convert(Dict{keytype(orders), Matrix{datatype}}, orders)
    datatype2 = promote_type(datatype, Complex{Int})
    res =  Dict{keytype(orders), datatype2}()
    for pid in 1:length(ms.cell)
        gs = ms.rotations[pid][1, :]
        res[pid] = gs'*orders[pid]*gs
    end
    return res
end
function (ms::MagneticStructure)(order::Matrix{T}) where T<:Number
    datatype = promote_type(eltype(valtype(ms.rotations)), eltype(order))
    res =  Dict{Int, Matrix{datatype}}()
    for pid in 1:length(ms.cell)
        res[pid] = order
    end
    return ms(res)
end

"""
    HPTransformation{S<:Operators, U<:CompositeIndex, M<:MagneticStructure} <: UnitSubstitution{U, S}

Holstein-Primakoff transformation.
"""
struct HPTransformation{S<:Operators, U<:CompositeIndex, M<:MagneticStructure} <: UnitSubstitution{U, S}
    magneticstructure::M
    function HPTransformation{S}(magneticstructure::MagneticStructure) where {S<:Operators}
        O = optype(HPTransformation, S)
        new{Operators{O, idtype(O)}, eltype(eltype(S)), typeof(magneticstructure)}(magneticstructure)
    end
end
@inline Base.valtype(hp::HPTransformation) = valtype(typeof(hp))
@inline Base.valtype(::Type{<:HPTransformation{S}}) where {S<:Operators} = S
@inline Base.valtype(::Type{<:HPTransformation{S}}, ::Type{<:Operator}) where {S<:Operators} = S
@inline Base.valtype(::Type{<:HPTransformation{S}}, ::Type{<:Operators}) where {S<:Operators} = S
@inline function optype(::Type{<:HPTransformation}, ::Type{S}) where {S<:Operators}
    V = promote_type(valtype(eltype(S)), Complex{Int})
    Iₒ = indextype(eltype(eltype(S)))
    Iₜ = reparameter(Iₒ, :iid, FID{:b, Int, Rational{Int}, Int})
    I = Iₜ<:Iₒ ? Iₒ : Iₜ
    U = reparameter(eltype(eltype(S)), :index, I)
    M = fulltype(eltype(S), NamedTuple{(:value, :id), Tuple{V, ID{U}}})
end
@inline (hp::HPTransformation)(oid::CompositeIndex; kwargs...) = Operator(1, oid)
function (hp::HPTransformation)(oid::CompositeIndex{<:Index{Int, <:FID{:b}}})
    # datatype = valtype(eltype(valtype(hp)))
    factor = 1/2
    n = length(hp.magneticstructure.moments[oid.index.site])÷2 + 1
    zₗ = zero(valtype(hp))
    add!(zₗ, 1)
    op = []
    opd = []
    for i = 1:n-1
        op₁ = Operator(1, replace(oid, index=replace(oid.index, iid=FID{:b}(i, 0, 1 ))))
        push!(op, op₁)
        push!(opd, op₁')
    end
    sub!(zₗ, factor*sum(opd.*op))
    sv = [zₗ, op...]
    svd = [zₗ, opd...]
    if oid.index.iid.nambu == 2
        uu = hp.magneticstructure.rotations[oid.index.site]
        u = (uu)'
        res = u*SVector(svd...)
    elseif oid.index.iid.nambu == 1
        uu = (hp.magneticstructure.rotations[oid.index.site])
        u = transpose(uu)
        res = u*SVector(sv...)
    end 
    return  res[oid.index.iid.orbital]
end

"""
    Hilbert(hilbert::Hilbert{<:Fock{:b}}, magneticstructure::MagneticStructure)

Get the corresponding Hilbert space of the original one after the Holstein-Primakoff transformation. 
"""
@inline function Hilbert(hilbert::Hilbert{<:Fock{:b}}, magneticstructure::MagneticStructure)
    return Hilbert(site=>Fock{:b}(hilbert[site].norbital-1, 1) for site in 1:length(magneticstructure.cell))
end
"""
    SUNMagnonic <: TBAKind{:BdG}

Magnonic quantum lattice system.
"""
struct SUNMagnonic <: TBAKind{:BdG} end

"""
    Metric(::SUNMagnonic, hilbert::Hilbert{<:Fock{:b}}) -> OperatorUnitToTuple

Get the index-to-tuple metric for a quantum spin system after the Holstein-Primakoff transformation.
"""
@inline @generated Metric(::SUNMagnonic, hilbert::Hilbert{<:Fock{:b}}) = OperatorUnitToTuple(:nambu, :site, :orbital)

"""
    commutator(::SUNMagnonic, hilbert::Hilbert{<:Fock{:b}}) -> Diagonal

Get the commutation relation of the Holstein-Primakoff bosons.
"""
@inline commutator(::SUNMagnonic, hilbert::Hilbert{<:Fock{:b}}) = Diagonal(kron([1, -1], ones(Int64, sum(length, values(hilbert))÷2)))
"""
    SUNLSWT{K<:TBAKind{:BdG}, L<:AbstractLattice, Hₛ<:OperatorGenerator, HP<:HPTransformation, Ω<:Image, H<:Image} <: AbstractTBA{K, H, AbstractMatrix}

SU(N) Linear spin wave theory for magnetically ordered quantum lattice systems.
"""
struct SUNLSWT{K<:TBAKind{:BdG}, L<:AbstractLattice, Hₛ<:OperatorGenerator, HP<:HPTransformation, Ω<:Image, H<:Image} <: AbstractTBA{K, H, AbstractMatrix}
    lattice::L
    Hₛ::Hₛ
    hp::HP
    Ω::Ω
    H::H
    commutator::AbstractMatrix
    function SUNLSWT{K}(lattice::AbstractLattice, Hₛ::OperatorGenerator, hp::HPTransformation) where {K<:TBAKind{:BdG}}
        temp = hp(Hₛ)
        hilbert = Hilbert(Hₛ.hilbert, hp.magneticstructure)
        table = Table(hilbert, Metric(K(), hilbert))
        H₀ = RankFilter(0)(temp, table=table)
        H₂ = RankFilter(2)(temp, table=table)
        commt = commutator(K(), hilbert)
        new{K, typeof(lattice), typeof(Hₛ), typeof(hp), typeof(H₀), typeof(H₂)}(lattice, Hₛ, hp, H₀, H₂, commt)
    end
end
@inline contentnames(::Type{<:SUNLSWT}) = (:lattice, :Hₛ, :hp, :Ω, :H, :commutator)
@inline function update!(sunlswt::SUNLSWT; k=nothing, kwargs...)
    if length(kwargs)>0
        update!(sunlswt.Hₛ; kwargs...)
        update!(sunlswt.Ω; kwargs...)
        update!(sunlswt.H; kwargs...)
    end
    return sunlswt
end

"""
    SUNLSWT(
        lattice::AbstractLattice, 
        hilbert::Hilbert, 
        terms::Tuple{Vararg{Term}}, 
        magneticstructure::MagneticStructure; 
        neighbors::Union{Nothing, Int, Neighbors}=nothing, 
        boundary::Boundary=plain
    )

Construct a SUNLSWT. `lattice` is the original lattice.
"""
@inline function SUNLSWT(lattice::AbstractLattice, hilbert::Hilbert, terms::Tuple{Vararg{Term}}, magneticstructure::MagneticStructure; neighbors::Union{Nothing, Int, Neighbors}=nothing, boundary::Boundary=plain)
    isnothing(neighbors) && (neighbors=maximum(term->term.bondkind, terms))
    Hₛ = OperatorGenerator(terms, bonds(magneticstructure.cell, neighbors), hilbert; half=false, boundary=boundary)
    hp = HPTransformation{valtype(Hₛ)}(magneticstructure)
    return SUNLSWT{SUNMagnonic}(lattice, Hₛ, hp)
end

"""
    add!(dest::Matrix,
        mr::TBAMatrixRepresentation{SUNMagnonic},
        m::Operator{<:Number, <:ID{CompositeIndex{<:Index{Int, <:FID{:b}}}, 2}};
        atol=atol/5, 
        kwargs...
    ) -> typeof(dest)

Get the matrix representation of an operator and add it to destination.
"""
function add!(dest::Matrix, mr::TBAMatrixRepresentation{SUNMagnonic}, m::Operator{<:Number, <:ID{CompositeIndex{<:Index{Int, <:FID{:b}}}, 2}}; atol=atol/5, kwargs...)
    if m[1]==m[2]'
        seq₁ = mr.table[m[1].index]
        seq₂ = mr.table[m[2].index]
        dest[seq₁, seq₁] += m.value + atol
        dest[seq₂, seq₂] += m.value + atol
    else
        coordinate = mr.gauge==:rcoordinate ? rcoordinate(m) : icoordinate(m)
        phase = isnothing(mr.k) ? one(eltype(dest)) : convert(eltype(dest), exp(1im*dot(mr.k, coordinate))) #c^†ᵢ = 1/√N*∑ₖc†ₖ exp(-i*k*rᵢ) ; c^†_k c_k exp(i*k*(rj-ri))
        seq₁ = mr.table[m[1].index']
        seq₂ = mr.table[m[2].index]
        dest[seq₁, seq₂] += m.value*phase
        if m[1].index.site == m[2].index.site && isapprox(norm(rcoordinate(m)), 0, atol=atol, rtol=rtol)
            seq₃ = mr.table[m[2].index']
            seq₄ = mr.table[m[1].index]
            dest[seq₃, seq₄] += m.value 
        else
            dest[seq₂, seq₁] += m.value'*phase' 
        end
        
    end
    return dest
end
"""
    matrixcoef(sunlswt::SUNLSWT) -> Hermitian

Return the coefficience of exchange interactions, i.e. ∑(Jᵢ₁ⱼ₂ⱼ₃ᵢ₄b†ᵢ₁b†ⱼ₂bⱼ₃bᵢ₄) + ∑(Bᵢᵤᵥb†ᵢᵤbᵢᵥ )
"""
function matrixcoef(sunlswt::SUNLSWT)
    table = Table(sunlswt.Hₛ.hilbert, OperatorUnitToTuple(:site, :orbital, :spin))
    H4 = RankFilter(4)(sunlswt.Hₛ, table=table)
    H2 = RankFilter(2)(sunlswt.Hₛ, table=table)
    hcoef = zeros(valtype(eltype(valtype(sunlswt.Hₛ))), length(table), length(table), length(table), length(table))
    for opt4 in H4
        seq₁ = table[opt4[1].index]
        seq₂ = table[opt4[3].index]
        seq₃ = table[opt4[2].index]
        seq₄ = table[opt4[4].index]
        hcoef[seq₁, seq₂, seq₃, seq₄] += opt4.value
    end
    hcoef2 = zeros(valtype(eltype(valtype(sunlswt.Hₛ))), length(table), length(table))
    for opt2 in H2
        seq₁ = table[opt2[1].index]
        seq₂ = table[opt2[2].index]
        hcoef2[seq₁, seq₂] += opt2.value
    end
    eye = zeros(valtype(eltype(valtype(sunlswt.Hₛ))), length(table), length(table))
    nf = size(sunlswt.hp.magneticstructure.rotations[1], 1)
    for i = 1:nf
        eye[i, i] = 1
    end
    hcoef4 = Hermitian(reshape(hcoef, length(table)^2, length(table)^2))
    res = hcoef4 + kron(eye, hcoef2)   
    return Hermitian((res + res')/2)
end
"""
    optimorder(sunlswt::SUNLSWT; numrand::Int = 0, method = LBFGS(), g_tol = 1e-12, optionskwargs... ) -> Tuple{SUNLSWT,Union{Optim.MultivariateOptimizationResults,Nothing}}

Optimize the ground state of Hamiltionian, i.e. ⟨T|H|T⟩, see DOI: 10.1103/PhysRevB.97.205106 
`numrand` is the number of the optimization. `optionskwargs` is the keywords of Optim.Options(g_tol=g_tol,optionskwargs...)
## References
 - Zhao-Yang Dong, Wei Wang, and Jian-Xin Li, SU(N) spin-wave theory: Application to spin-orbital Mott insulators, Physical Review B 97, 205106 (2018)
"""
function optimorder(sunlswt::SUNLSWT; numrand::Int = 0, method = LBFGS(), g_tol = 1e-12, optionskwargs... ) 
    numrand == 0 && return sunlswt, nothing
    hcoef = matrixcoef(sunlswt)
    ms₀ = sunlswt.hp.magneticstructure
    gen_mat = []
    ndims = Int[]
    x₀ = Float64[]
    for pid in 1:length(ms₀.cell)
        n = length(ms₀.moments[pid])÷2 + 1
        push!(gen_mat, rotation_gen(n))
        push!(ndims, n)
        append!(x₀, ms₀.moments[pid])
    end
    @assert  sum(ndims)^2 == size(hcoef,1) "optimorder error: the dimension of matrix of exchange coefficience $(sqrt(size(hcoef,1))) is equal to the degrees of freedom $(sum(ndims)) in unit cell. "
    function fmin(x::Vector)
        n0, n1 = 0, 0
        m = sum(ndims)
        rotations = zeros(ComplexF64, m)
        for (i, n) in  enumerate(ndims)   
            θs = x[n0 + 1 : 2*(n - 1) + n0 ]
            u1 = rotation(θs, gen_mat[i])
            rotations[1 + n1 : n + n1] = u1[1, :] 
            n0 += 2*n - 2
            n1 += n  
        end
        u = kron(rotations, rotations)
        res = u'*hcoef*u
        return real(res)
    end
    nx₀ = length(x₀)
    lb = zeros(Float64, nx₀) .- 1e-16
    ub = zeros(Float64, nx₀) 
    n0 = 0
    for n in ndims   
        ub[n0 + 1 : (n - 1) + n0] .= pi
        ub[n0 + 1 + n - 1 : 2*(n - 1) + n0] .= 2*pi
        n0 += 2*n - 2 
    end
    inner_optimizer = method #GradientDescent() #ConjugateGradient()
    op = optimize(fmin, lb, ub, x₀, Fminbox(inner_optimizer), Options(g_tol = g_tol, optionskwargs...))
    op₀ = op 
    for _ = 1:1:numrand-1
        x₀ = rand(nx₀) .* ub 
        op = optimize(fmin, lb, ub, x₀, Fminbox(inner_optimizer), Options(g_tol = g_tol, optionskwargs...))
        op₀ = op.minimum <= op₀.minimum ? op : op₀
    end
    #obtain new SUNLSWT 
    x₁ = []
    n0 = 0
    for n in ndims
        push!(x₁, op₀.minimizer[n0 + 1 : 2*(n - 1) + n0] )
        n0 += 2*n - 2
    end
    moments = Dict{keytype(ms₀.moments), valtype(ms₀.moments)}()
    for i in 1:length(ms₀.cell)
        moments[i] = x₁[i]
    end 
    ms₁ = MagneticStructure(ms₀.cell, moments)
    hp₁ =  HPTransformation{valtype(sunlswt.Hₛ)}(ms₁)
    newsunlswt = SUNLSWT{SUNMagnonic}(sunlswt.lattice, sunlswt.Hₛ, hp₁)
    E₀ = (newsunlswt.Ω.operators|>expand).contents[()].value
    @assert imag(E₀) < 1e-12 "optimorder error: the imaginary part of H₀ (classical energy $(E₀)) is larger than 1e-12. "
    return newsunlswt, op₀
end

"""
    localorder(sunlswt::SUNLSWT, orders::Dict{Int, <:Matrix{T}}) where T<:Number -> Dict{Int, <:Number} 

Return order parameters of each site, <classical gs|o|classical gs>. `orders[pid]` is the matrix of physical observable.
"""
@inline function localorder(sunlswt::SUNLSWT, orders::Dict{Int, <:Matrix{<:Number}}) 
    return (sunlswt.hp.magneticstructure)(orders)
end
@inline localorder(sunlswt::SUNLSWT, order::Matrix{<:Number}) = (sunlswt.hp.magneticstructure)(order)

#Spectra
"""
    SpectraKind{K}
    The kind of a spectrum calculation.
"""
abstract type SpectraKind{K} end
"""
    InelasticNeutron <: SpectraKind{:INS}

Inelastic Neutron Scattering Spectra of quantum lattice system.
"""
struct InelasticNeutron <: SpectraKind{:INS} end
"""
    Multipole <: SpectraKind{:Multipole}

Multipole Scattering Spectra of quantum lattice system. (∑_{α}s_{α}*s_{α})
"""
struct Multipole <: SpectraKind{:Multipole} end

"""
    Spectra{K<:SpectraKind, P<:ReciprocalSpace, E<:AbstractVector, S<:Operators, O} <: Action

Spectra of 'magnetically' ordered quantum lattice systems by SU(N) linear spin wave theory.
"""
struct Spectra{K<:SpectraKind, P<:ReciprocalSpace, E<:AbstractVector, S<:Operators, O} <: Action
    reciprocalspace::P
    energies::E
    operators::Tuple{AbstractVector{S}, AbstractVector{S}}
    options::O
    function Spectra{K}(reciprocalspace::ReciprocalSpace, energies::AbstractVector, operators::Tuple{AbstractVector{<:Operators}, AbstractVector{<:Operators}}, options) where {K<:SpectraKind}
        @assert names(reciprocalspace)==(:k,) "Spectra error: the name of the momenta in the reciprocalspace must be :k."
        datatype = eltype(eltype(operators))
        new{K, typeof(reciprocalspace), typeof(energies), datatype, typeof(options)}(reciprocalspace, energies, operators, options)
    end
end
@inline Spectra{K}(reciprocalspace::ReciprocalSpace, energies::AbstractVector, operators::Tuple{AbstractVector{<:Operators}, AbstractVector{<:Operators}}; options...) where {K <: SpectraKind} = Spectra{K}(reciprocalspace, energies, operators, options)

@inline function initialize(ins::Spectra, sunlswt::SUNLSWT)
    x = ins.reciprocalspace#collect(Float64, 0:(length(ins.reciprocalspace)-1))
    y = collect(Float64, ins.energies)
    z = zeros(Float64, length(y), length(x))
    return (x, y, z)
end
function run!(sunlswt::Algorithm{<:SUNLSWT{SUNMagnonic}}, ins::Assignment{<:Spectra{InelasticNeutron}}) 
    operators = spinoperators(ins.action.operators, sunlswt.frontend.hp)
    m = zeros(promote_type(valtype(sunlswt.frontend), Complex{Int}), dimension(sunlswt.frontend), dimension(sunlswt.frontend))
    data = zeros(Complex{Float64}, size(ins.data[3]))
    gauss = get(ins.action.options, :gauss, true)
    kT = get(ins.action.options, :kT, 0.0) # k = 8.617333262145e-5 eV/K
    σ = gauss ? get(ins.action.options, :fwhm, 0.1)/2/√(2*log(2)) : get(ins.action.options, :fwhm, 0.1)
    for (i, q) in enumerate(ins.action.reciprocalspace)
        (eigenvalues, eigenvectors) = eigen(sunlswt; k=q, ins.action.options...)
        @timeit sunlswt.timer "spectra" for α=1:3, β=1:3
            factor = delta(α, β) - ((norm(q)==0 || α>length(q) || β>length(q)) ? 0 : q[α]*q[β]/dot(q, q))
            if !isapprox(abs(factor), 0, atol=atol, rtol=rtol)
                matrix!(m, operators, α, β, sunlswt.frontend.H.table, q)
                diag = Diagonal(eigenvectors'*m*eigenvectors)
                for (nₑ, e) in enumerate(ins.action.energies)
                    for j = (dimension(sunlswt.frontend)÷2 + 1):dimension(sunlswt.frontend)
                        temp = gauss ? 1/√(2pi)/σ*exp(-(e-eigenvalues[j])^2/2/σ^2) : σ^2/(σ^2 + (e-eigenvalues[j])^2)/pi
                        bosonfactor = (kT ≈ 0.0) ? 1.0 : 1 + 1/(exp(eigenvalues[j]/kT) - 1 )
                        data[nₑ, i] += factor*diag[j, j]*temp*bosonfactor 
                    end
                end
            end
        end
    end
    isapprox(norm(imag(data)), 0, atol=atol, rtol=rtol) || @warn "run! warning: not negligible imaginary part ($(norm(imag(data))))."
    ins.data[3][:, :] .= real(data)[:, :]
    ins.data[3][:, :] = get(ins.action.options, :scale, identity).(ins.data[3].+1)
end
function spinoperators(opt::Tuple{AbstractVector{<:Operators}, AbstractVector{<:Operators}}, hp::HPTransformation{S, U}) where { S<:Operators, U<:CompositeIndex{<:Index{Int, <:FID}}}
    @assert length(opt[1]) == length(opt[2]) == 3 "spinoperators error: the operators are not three spin operators."
    opt₁ = RankFilter(1).(hp.(opt[1]))
    opt₂ = RankFilter(1).(hp.(opt[2]))
    x₁, y₁, z₁ = opt₁[1], opt₁[2], opt₁[3]
    x₂, y₂, z₂ = opt₂[1], opt₂[2], opt₂[3]
    return @SMatrix [x₁*x₂ x₁*y₂ x₁*z₂; y₁*x₂ y₁*y₂ y₁*z₂; z₁*x₂ z₁*y₂ z₁*z₂]
end
function matrix!(m::Matrix{<:Number}, operators::SMatrix{3, 3, <:Operators, 9}, i::Int, j::Int, table::Table, k)
    m[:, :] .= zero(eltype(m))
    for op in operators[i, j]
        phase = convert(eltype(m), exp(-1im*dot(k, rcoordinate(op))))
        seq₁ = table[op[1].index']
        seq₂ = table[op[2].index]
        m[seq₁, seq₂] += op.value*phase
    end
    return m
end

function run!(sunlswt::Algorithm{<:SUNLSWT}, ins::Assignment{<:Spectra{Multipole}})
    operators = _multipoleoperators(ins.action.operators, sunlswt.frontend.hp)
    m = zeros(promote_type(valtype(sunlswt.frontend), Complex{Int}), dimension(sunlswt.frontend), dimension(sunlswt.frontend))
    data = zeros(Complex{Float64}, size(ins.data[3]))
    gauss = get(ins.action.options, :gauss, true)
    σ = gauss ? get(ins.action.options, :fwhm, 0.1)/2/√(2*log(2)) : get(ins.action.options, :fwhm, 0.1)
    for (i, q) in enumerate(ins.action.reciprocalspace)
        (eigenvalues, eigenvectors) = eigen(sunlswt; k=q, ins.action.options...)
        @timeit sunlswt.timer "spectra" (
                matrix!(m, operators, sunlswt.frontend.H.table, q);
                diag = Diagonal(eigenvectors'*m*eigenvectors);
                for (nₑ, e) in enumerate(ins.action.energies)
                    for j = (dimension(sunlswt.frontend)÷2 + 1):dimension(sunlswt.frontend)
                        temp = gauss ? 1/√(2pi)/σ*exp(-(e-eigenvalues[j])^2/2/σ^2) : σ^2/(σ^2 + (e-eigenvalues[j])^2)/pi
                        data[nₑ, i] += diag[j, j]*temp  
                    end
                end
        )
    end
    isapprox(norm(imag(data)), 0, atol=atol, rtol=rtol) || @warn "run! warning: not negligible imaginary part ($(norm(imag(data))))."
    ins.data[3][:, :] .= real(data)[:, :]
    ins.data[3][:, :] = get(ins.action.options, :scale, identity).(ins.data[3].+1)
end
function _multipoleoperators(opt::Tuple{AbstractVector{<:Operators}, AbstractVector{<:Operators}}, hp::HPTransformation{S, U}) where {S<:Operators, U<:CompositeIndex{<:Index{Int, <:FID}}}
    opt₁ = RankFilter(1).(hp.(opt[1]))
    opt₂ = RankFilter(1).(hp.(opt[2]))
    return sum(opt₁.*opt₂)
end
function matrix!(m::Matrix{<:Number}, operators::Operators, table::Table, k)
    m[:, :] .= zero(eltype(m))
    for op in operators
        phase = convert(eltype(m), exp(-1im*dot(k, rcoordinate(op))))
        seq₁ = table[op[1].index']
        seq₂ = table[op[2].index]
        m[seq₁, seq₂] += op.value*phase
    end
    return m
end
"""
    kdosOperator(u::Matrix{<:Number}, site::Int, lattice::Union{<:AbstractLattice, Int}) -> Operators

Return Operators for kDOS case. u = ⟨α|site,orbital⟩, where |site,orbital⟩ => b_{site,orbital}, |α⟩ is the target state.
"""
function kdosOperator(u::Matrix{<:Number}, site::Int, lattice::Union{<:AbstractLattice, Int})
    @assert size(u, 1) == 1 "kdosOperator error: the size of matrix is (1,n) ($(size(u)))."
    dim = isa(lattice, Int) ? lattice : dimension(lattice)
    res = []
    for (i, value) in enumerate(u)
        index₀ =  CompositeIndex(Index(site, FID{:b}(i, 0, 1)); rcoordinate=zeros(dim), icoordinate=zeros(dim))
        push!(res, Operator(value, index₀))
    end
    return Operators(res...)
end
"""
    spectraEcut(ass::Assignment{<:Spectra}, Ecut::Float64, dE::Float64) -> Tuple{Vector{Float64}, Vector{Float64}, Matrix{Float64}}

Construct the spectra with fixed energy. The energy of `abs(energy-Ecut) <= dE` is selected. `nx` and `ny` are the number of x and y segments of ReciprocalZone, respectively.
"""
function spectraEcut(ass::Assignment{<:Spectra}, Ecut::Float64, dE::Float64)
    @assert isa(ass.action.reciprocalspace, ReciprocalZone) "spectraEcut error: please input the ReciprocalZone."
    energies = ass.data[2]
    f(x) = (abs(x - Ecut) <= dE ? true : false)
    i = findall(f, energies)
    intensity = ass.data[3]
    dims = Int[]
    seg = []
    reciprocals = []
    for (i, bound) in enumerate(ass.action.reciprocalspace.bounds)
        if bound.length > 1 
            push!(dims, bound.length)
            push!(seg, bound)
            push!(reciprocals, ass.action.reciprocalspace.reciprocals[i])
        end
    end
    @assert length(dims) == 2 "spectraEcut error: the k points is not in a plane."
    y = collect(seg[2])*norm(reciprocals[2])#collect(Float64, 0:(dims[2]-1))
    x = collect(seg[1])*norm(reciprocals[1])#collect(Float64, 0:(dims[1]-1))
    z = reshape(sum(intensity[i, :], dims=1), reverse(dims)...)
    return (x, y, z)
end
"""
    multipoleOperator(j₁::Union{Int,Rational{Int}}, j₂::Union{Int,Rational{Int}}, j::Union{Int,Rational{Int}}, m::Union{Int,Rational{Int}}) -> Array{Float64}

Construct the matrix of multipole operator, i.e. ``M_{jm}^{j₁j₂} = ∑_{m₁m₂} (-1)^{j₂-m₂}C_{m₁,-m₂,m}^{j₁j₂j} b†_{j₁m₁}b_{j₂m₂} `` with ``M_{jm}^{†j₁j₂}=M_{j-m}^{j₂j₁}(-1)^{j₂-j₁+m}``. e.g. the basis of dipole matrix is [|1⟩, |0⟩, |-1⟩].
"""
function multipoleOperator(j₁::Union{Int, Rational{Int}}, j₂::Union{Int, Rational{Int}}, j::Union{Int,Rational{Int}}, m::Union{Int, Rational{Int}})
    n1 = Int(2*j₁+1)
    n2 = Int(2*j₂+1)
    m1 = collect(0:-1:-n1+1) .+ j₁
    m2 = collect(0:-1:-n2+1) .+ j₂
    res = zeros(Float64, n1, n2)
    for (i2, m₂) in enumerate(m2)
        for (i1, m₁) in enumerate(m1)
            m ≠ m₁-m₂ && continue
            res[i1, i2] = clebschgordan(Float64, j₁, m₁, j₂, -m₂, j)*(-1)^(j₂-m₂)
        end
    end
    return res
end
"""
    fluctuation(sunlswt::SUNLSWT, bz::ReciprocalZone, pid::Int, nbonson::Int; atol=1e-8) -> Float64
    fluctuation(sunlswt::SUNLSWT, bz::ReciprocalZone; atol=1e-8) -> Dict{<:AbstractPID, Float64}

To calculate ``⟨b_0^†b_0⟩`` with respect to the quantum ground state |0⟩ at zero temperature.  
"""
function fluctuation(sunlswt::SUNLSWT, bz::ReciprocalZone, pid::Int, nbonson::Int; atol=1e-8)
    table = sunlswt.H.table
    n = length(table)÷2
    indx = [Index(pid, FID{:b}(i, 0, 2)) for i in 1:nbonson]
    v = 0.0
    for q in bz
        m = matrix(sunlswt; atol=atol, k=q)
        for j in 1:nbonson
            v += sum(abs2, eigen(m).vectors[table[indx[j]], n+1:end])
        end
    end 
    res = 1 - v/length(bz)
    @assert res >= 0.0 "fluctuation error: it is negative, please increase the number of k points." 
    return res
end
function fluctuation(sunlswt::SUNLSWT, bz::ReciprocalZone; atol=1e-8)
    ms = sunlswt.hp.magneticstructure.moments
    res = Dict{keytype(ms), Float64}()
    for pid in keys(ms)
        nbonson = length(ms[pid])÷2
        v = fluctuation(sunlswt, bz, pid, nbonson; atol=atol)
        res[pid] = v
    end
    return res
end

#plot
"""
    @recipe  plot(pack::Tuple{Algorithm{<:SUNLSWT}, Assignment{<:Spectra}}, Ecut::Float64, dE::Float64=1e-3)

Define the recipe for the visualization of an assignment with Ecut and data of an algorithm.
"""
@recipe function plot(pack::Tuple{Algorithm{<:SUNLSWT}, Assignment{<:Spectra}}, Ecut::Float64, dE::Float64=1e-3)
    title --> nameof(pack[1], pack[2], Ecut)
    titlefontsize --> 10
    legend --> false
    xlabel --> "q₁"
    ylabel --> "q₂" 
    aspect_ratio := :equal
    @series begin
        seriestype := :heatmap
        data = spectraEcut(pack[2], Ecut, dE)
        data 
    end  
end
"""
    @recipe plot(pack::Tuple{Algorithm{<:SUNLSWT}, Assignment{<:Spectra}}, parameters::AbstractVector{Float64})

Define the recipe for the visualization of an assignment with parameters (title) of an algorithm.
"""
@recipe function plot(pack::Tuple{Algorithm{<:SUNLSWT}, Assignment{<:Spectra}}, parameters::AbstractVector{Float64})
    title --> nameof(pack[1], pack[2], parameters)
    titlefontsize --> 10
    legend --> false
    seriestype --> (isa(pack[2].data, Tuple{Any, Any, Any}) ? :heatmap : :path)
    pack[2].data
end
"""
    nameof(alg::Algorithm, assign::Assignment, ecut::Float64) -> String

Get the name of the combination of an algorithm, an assignment and ecut.
"""
@inline Base.nameof(alg::Algorithm, assign::Assignment{<:Spectra}, ecut::Float64) = @sprintf "%s_%s_Ecut=%s" repr(alg) assign.id ecut
function Base.nameof(alg::Algorithm, assign::Assignment{<:Spectra}, parameters::AbstractVector{Float64}) 
    result = [@sprintf "%s_%s" repr(alg) assign.id ]
    for value in parameters
        push!(result, decimaltostr(value, 10))
    end
    return join(result, "_")
end
"""
    optimorder2(sunlswt::SUNLSWT; numrand::Int = 0,rule::Function=x->x, method = LBFGS(), g_tol = 1e-12, optionskwargs... ) -> Tuple{SUNLSWT, Union{Optim.MultivariateOptimizationResults, Nothing}}

Optimize the ground state of Hamiltionian, i.e. ⟨T|H|T⟩. If the all degrees of freedom is considered, the Function `optimorder` is recommended.
"""
function optimorder2(sunlswt::SUNLSWT; numrand::Int = 0, rule::Function=x->x, method = LBFGS(), g_tol = 1e-12, optionskwargs... ) 
    ms₀ =  sunlswt.hp.magneticstructure
    x₀ = Float64[]
    ub = Float64[]
    for pid in 1:length(ms₀.cell)
        append!(x₀, ms₀.moments[pid])
        n = length(ms₀.moments[pid]) ÷ 2
        append!(ub, ones(n)*pi, ones(n)*2*pi)
    end
    return optimorder2(sunlswt, ub, x₀; numrand=numrand, rule=rule, method=method, g_tol=g_tol, optionskwargs... )
end
"""
    optimorder2(sunlswt::SUNLSWT, ub::Vector{Float64}, x₀::Vector{Float64}; numrand::Int = 0, rule::Function=x->x, method = LBFGS(), g_tol = 1e-12, optionskwargs... ) -> Tuple{SUNLSWT, Union{Optim.MultivariateOptimizationResults, Nothing}}

Optimize the ground state of Hamiltionian. If the all degrees of freedom is considered, the Function `optimorder` is recommended.
Function `rule` gives the transformation of the selected degrees of freedom to all degrees of freedom.
"""
function optimorder2(sunlswt::SUNLSWT, ub::Vector{Float64}, x₀::Vector{Float64}; numrand::Int = 0, rule::Function=x->x, method = LBFGS(), g_tol = 1e-12, optionskwargs... ) 
    numrand == 0 && return sunlswt, nothing
    hcoef = matrixcoef(sunlswt)
    ms₀ = sunlswt.hp.magneticstructure
    gen_mat = []
    ndims = Int[]
    for pid in 1:length(ms₀.cell)
        n = length(ms₀.moments[pid])÷2 + 1
        push!(gen_mat, rotation_gen(n))
        push!(ndims, n)
    end
    @assert  sum(ndims)^2 == size(hcoef, 1) "optimorder error: the dimension of matrix of exchange coefficience $(sqrt(size(hcoef,1))) is equal to the degrees of freedom $(sum(ndims)) in unit cell. "
    function fmin(x::Vector)
        n0, n1 = 0, 0
        m = sum(ndims)
        rotations = zeros(ComplexF64, m)
        xx = rule(x)
        for (i, n) in enumerate(ndims)    
            θs = xx[n0 + 1 : 2*(n - 1) + n0 ]
            u1 = rotation(θs, gen_mat[i])
            rotations[1 + n1 : n + n1] = u1[1, :] 
            n0 += 2*n - 2
            n1 += n  
        end
        u = kron(rotations, rotations)
        res = u'*hcoef*u
        return real(res)
    end
    nx₀ = length(x₀)
    lb = zeros(Float64, nx₀) .- 1e-16
    inner_optimizer = method #GradientDescent() #ConjugateGradient()
    op = optimize(fmin, lb, ub, x₀, Fminbox(inner_optimizer), Options(g_tol = g_tol, optionskwargs...))
    op₀ = op 
    for _ = 1:1:numrand-1
        x₀ = rand(nx₀) .* ub 
        op = optimize(fmin, lb, ub, x₀, Fminbox(inner_optimizer), Options(g_tol = g_tol, optionskwargs...))
        op₀ = op.minimum <= op₀.minimum ? op : op₀
    end
    x₁ = rule(op₀.minimizer)
    moments = Dict{keytype(ms₀.moments), valtype(ms₀.moments)}()
    n0 = 0
    for i in 1:length(ms₀.cell)
        n = ndims[i]
        moments[i] = x₁[n0 + 1 : 2*(n - 1) + n0] 
        n0 += 2*n - 2   
    end 
    ms₁ = MagneticStructure(ms₀.cell, moments)
    hp₁ =  HPTransformation{valtype(sunlswt.Hₛ)}(ms₁)
    newsunlswt = SUNLSWT{SUNMagnonic}(sunlswt.lattice, sunlswt.Hₛ, hp₁)
    E₀ = (newsunlswt.Ω.operators|>expand).contents[()].value
    @assert isapprox(real(E₀), op₀.minimum, atol=1e-11) "optimorder2 error: E₀($(real(E₀))) == minimum($(op₀.minimum))."
    @assert imag(E₀) < 1e-12 "optimorder2 error: the imaginary part of H₀ (classical energy $(E₀)) is larger than 1e-12. "
    return newsunlswt, op₀
end

end #module


