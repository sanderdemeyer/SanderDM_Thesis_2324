using TensorOperations
using TensorKit
using MPSKit
using MPSKitModels
using LinearAlgebra: norm

using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_groundstate_energy.jl")
include("get_thirring_hamiltonian.jl")
include("get_thirring_hamiltonian_symmetric.jl")

mass = 0
g = -2
delta_g = cos((pi-g)/2)
delta_g = 2
println(delta_g)
D_bond = 50
# (gs_energy, _) = get_groundstate_energy(mass, delta_g, 0, 0, D_bond)
println("hrere")
# println(gs_energy)


amount = 28
Delta_g_range = LinRange(-1, 6, amount)

amount = 20
mass_range = LinRange(0, 1, amount)

Energies = Array{Float64, 2}(undef, 1, amount)

for index = 1:amount
    mass = mass_range[index]
    println("mass is $mass")
    (gs_energy, _) = get_groundstate_energy(mass, 0, 0, 0, D_bond)
    println("Energy is $gs_energy")
    Energies[index] = real(gs_energy)
end

println(Energies)

@save "Check_mass_symmetric_D_30" mass_range Energies

# Energies = Array{Float64, 2}(undef, 1, D_number)
# Correlation_lengths = Array{Float64, 2}(undef, 1, D_number)

#=
for i = 1:D_number
    D = i*150
    println("D = ", D)
    (gs_energy, gs_correlation_length) = get_groundstate_energy(am_tilde_0, Delta_g, 0, D)
    #gs = get_groundstate_energy(am_tilde_0, Delta_g, D)
    Energies[i] = real(gs_energy[0] + gs_energy[1])/2
    println(Energies[i])
    #Correlation_lengths[i] = gs_correlation_length
end
=#




#=
mu_old = 0.1
magnetization_old = -0.3

mu_new = 0.01
magnetization_new = 0.3836

while abs(magnetization_new) > 0.02
    global mu_new
    global mu_old
    global magnetization_old
    global magnetization_new
    println("mu_new is $mu_new")
    println("magnetization_new is $magnetization_new")
    (gs_energy, magnetization_newest) = get_groundstate_energy(mass, delta_g, 0, mu_new, D_bond)
    mu_newest = mu_new - magnetization_new*(mu_new-mu_old)/(magnetization_new-magnetization_old)
    magnetization_old = magnetization_new
    mu_old = mu_new
    magnetization_new = magnetization_newest
    mu_new = mu_newest
end
=#

#=
D_values = [floor(Int, 25*(2^(i/2))) for i = 1:8]

D_values = [floor(Int, 12.5*(2^(i/2))) for i = 2:9]

println(D_values)

delta_gs = [0.5, 0.375, 0.25, 0.125, 0, -0.2, -0.4, -0.6, -0.8, -1]
gs = LinRange(-1.5, 1.5, 13)
println([e for e = gs])
delta_gs = cos.((pi.-gs)/2)

println(gs)
println(delta_gs)

a = zeros(5)
b = ones(5)
println(norm(b-a))

println(real(1+2im))

D = 4
j = 2
ctr = 1
state = InfiniteMPS([ℂ^2, ℂ^2],[ℂ^D, ℂ^D])
sz_mpo = TensorMap([1.0 0; 0 -1.0], ℂ^1 * ℂ^2, ℂ^2 * ℂ^1)
sz_mpo = TensorMap([1.0 0; 0 -1.0], ℂ^2, ℂ^2)

println([1 2; 3 4])

S⁺ = TensorMap([0.0 1.0; 0.0 0.0], ℂ^2, ℂ^2)
S⁻ = TensorMap([0.0 0.0; 1.0 0.0], ℂ^2, ℂ^2)

println(S⁺)
println(S⁻)

op = S_xx() + S_yy()
print(op)
print(typeof(op))


@tensor op_new[-1 -2; -3 -4] := (1/2)*(S⁺[-1; -3] * S⁻[-2; -4] + S⁺[-2; -4] * S⁻[-1; -3])
println("jkfjkdqlmfqdsjkflmjfqklsmjf")
println(op)
println(op_new)
=#

#=

Sz = S_z()
@tensor op2[-1 -2; -3 -4] := Sz[-1; -4] * Sz[-2; -3]
@tensor op2[-1 -2; -3 -4] := Sz[-1; -4] * Sz[-3; -2]
@tensor op2[-1 -3; -4 -2] := Sz[-1; -4] * Sz[-2; -3]
@tensor op2[-1 -2; -3 -4] := Sz[-1; -3] * Sz[-4; -2]
@tensor op2[-1 -2; -3 -4] := Sz[-1; -3] * Sz[-2; -4]



@tensor op3[-1 -2; -3 -4] := Sz[-2; -1] * Sz[-3; -4]

println("start comparison")
println("fjdsqklmfjksjfqklmdjkl\n")
println(op2)
println(S_zz())


=#

#=
println("here")
println(length(state))
println("fjdkqlm")
@tensor W = state.AC[1][1 2; 3] * conj(state.AC[1][1 2; 3])
println("fhe")
println(W)
print("done with W")
@tensor V[-1; -2] := sz_mpo[-1; 1] * sz_mpo[1; -2]

print(V)
print(K)


println("done");
#V = TransferMatrix(state.AR[ctr:(j - 1)], [sz_mpo sz_mpo], state.AR[ctr:(j - 1)])



println(exp(1))
println(log(exp(1)))

print(floor(Int, 3.0))

println(abs(1+1im))

println(1 + im*5)

b = zeros(ComplexF64, (1, 20))
for i = 1:5
    global b += i * ones(ComplexF64, (1, 20))
end
b ./= 5
println(b)

println(0%2)

=#

#=
# You can create a random 1 site periodic infinite mps (bond dimension 10) by calling
state = InfiniteMPS([ℂ^2],[ℂ^10]);

state.AL

#We can use a pre-defined hamiltonian from MPSKitModels
hamiltonian = transverse_field_ising();
hamiltonian = heisenberg_XXX();


#And find the groundstate
(groundstate,_) = find_groundstate(state,hamiltonian,VUMPS(maxiter = 3));

println(expectation_value(state, hamiltonian))
println(expectation_value(groundstate, hamiltonian))

spin = 3 // 2
S_x_mat, _, _ = spinmatrices(spin, ComplexF64)
pspace = ComplexSpace(size(S_x_mat, 1))
println(size(S_x_mat))
=#

#=
@mpoham begin
    sum(nearest_neighbours(lattice)) do (i, j)
        return (σˣˣ() + σʸʸ()){i, j}
        return (σˣ(){i}*σˣ(){j} + σʸ(){i}*σʸ(){j})
    end
end
{
    "name": "MethodError",
    "message": "MethodError: no method matching *(::LocalOperator{TrivialTensorMap{ComplexSpace, 2, 2, Matrix{ComplexF64}}, LatticePoint{1, FiniteChain}}, ::LocalOperator{TrivialTensorMap{ComplexSpace, 2, 2, Matrix{ComplexF64}}, LatticePoint{1, FiniteChain}})\n\nClosest candidates are:\n  *(::Any, ::Any, !Matched::Any, !Matched::Any...)\n   @ Base operators.jl:578\n  *(!Matched::Union{InitialValues.NonspecificInitialValue, InitialValues.SpecificInitialValue{typeof(*)}}, ::Any)\n   @ InitialValues ~/.julia/packages/InitialValues/OWP8V/src/InitialValues.jl:154\n  *(!Matched::Union{MPSKit.AC2_EffProj, MPSKit.AC_EffProj}, ::Any)\n   @ MPSKit ~/.julia/packages/MPSKit/fz0C5/src/algorithms/derivatives.jl:294\n  ...\n",
    "stack": "MethodError: no method matching *(::LocalOperator{TrivialTensorMap{ComplexSpace, 2, 2, Matrix{ComplexF64}}, LatticePoint{1, FiniteChain}}, ::LocalOperator{TrivialTensorMap{ComplexSpace, 2, 2, Matrix{ComplexF64}}, LatticePoint{1, FiniteChain}})\n\nClosest candidates are:\n  *(::Any, ::Any, !Matched::Any, !Matched::Any...)\n   @ Base operators.jl:578\n  *(!Matched::Union{InitialValues.NonspecificInitialValue, InitialValues.SpecificInitialValue{typeof(*)}}, ::Any)\n   @ InitialValues ~/.julia/packages/InitialValues/OWP8V/src/InitialValues.jl:154\n  *(!Matched::Union{MPSKit.AC2_EffProj, MPSKit.AC_EffProj}, ::Any)\n   @ MPSKit ~/.julia/packages/MPSKit/fz0C5/src/algorithms/derivatives.jl:294\n  ...\n\n\nStacktrace:\n  [1] (::Main.TDMPOHamiltonianTools.var\"#2#4\"{Matrix{Float64}, Int64, Int64})(::Pair{LatticePoint{1, FiniteChain}, LatticePoint{1, FiniteChain}})\n    @ Main.TDMPOHamiltonianTools ~/Documents/UGent/Ma2/Thesis-Code/Pulse Hamiltonians/TDMPOHam.jl:37\n  [2] _mapreduce(f::Main.TDMPOHamiltonianTools.var\"#2#4\"{Matrix{Float64}, Int64, Int64}, op::typeof(Base.add_sum), #unused#::IndexLinear, A::Vector{Pair{LatticePoint{1, FiniteChain}, LatticePoint{1, FiniteChain}}})\n    @ Base ./reduce.jl:435\n  [3] _mapreduce_dim(f::Function, op::Function, #unused#::Base._InitialValue, A::Vector{Pair{LatticePoint{1, FiniteChain}, LatticePoint{1, FiniteChain}}}, #unused#::Colon)\n    @ Base ./reducedim.jl:365\n  [4] #mapreduce#800\n    @ ./reducedim.jl:357 [inlined]\n  [5] mapreduce\n    @ ./reducedim.jl:357 [inlined]\n  [6] #_sum#810\n    @ ./reducedim.jl:999 [inlined]\n  [7] _sum(f::Function, a::Vector{Pair{LatticePoint{1, FiniteChain}, LatticePoint{1, FiniteChain}}}, ::Colon)\n    @ Base ./reducedim.jl:999\n  [8] #sum#808\n    @ ./reducedim.jl:995 [inlined]\n  [9] sum\n    @ ./reducedim.jl:995 [inlined]\n [10] TDMPOHam(pulse_matrix::Matrix{Float64}, update_times::Vector{Float64}, lattice::FiniteChain)\n    @ Main.TDMPOHamiltonianTools ~/Documents/UGent/Ma2/Thesis-Code/Pulse Hamiltonians/TDMPOHam.jl:31\n [11] TDMPOHam(qc::PyObject)\n    @ Main.TDMPOHamiltonianTools ~/Documents/UGent/Ma2/Thesis-Code/Pulse Hamiltonians/TDMPOHam.jl:51\n [12] top-level scope\n    @ ~/Documents/UGent/Ma2/Thesis-Code/Pulse Hamiltonians/TimeStepError.ipynb:2"
}
=#