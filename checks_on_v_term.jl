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


get_groundstate_energy(1.0, 0.5, 0, 30, true)
get_groundstate_energy(1.0, 0.5, 0, 15, false)
get_groundstate_energy(0.5, -3.5, 0, 30, true)
get_groundstate_energy(0.5, -3.5, 0, 15, false)


break;


S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
@tensor operator_threesite[-1 -2 -3; -4 -5 -6] := S⁺[-1; -4] * S_z()[-2; -5] * S⁻[-3; -6]
@tensor operator_threesite_final_asym[-1 -2 -3; -4 -5 -6] := operator_threesite[-1 -2 -3; -4 -5 -6] - operator_threesite[-3 -2 -1; -6 -5 -4]

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
Plus_space = U1Space(1 => 1)

S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace ⊗ Plus_space)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace)
S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], Plus_space ⊗ pspace, pspace ⊗ Plus_space)

@tensor operator_threesite[-1 -2 -3; -4 -5 -6] := S⁺[-1; -4 1] * S_z_symm[1 -2; -5 2] * S⁻[2 -3; -6]
@tensor operator_threesite_final_sym[-1 -2 -3; -4 -5 -6] := operator_threesite[-1 -2 -3; -4 -5 -6] - operator_threesite[-3 -2 -1; -6 -5 -4]

display(reshape(convert(Array, operator_threesite_final_asym), 8, 8))
display(reshape(convert(Array, operator_threesite_final_sym), 8, 8))

break;





println("symmetric")
hamiltonian = get_thirring_hamiltonian_symmetric(0.5, 0, 0.0)

println(typeof(hamiltonian))
println(typeof(hamiltonian[1]))

println("allee we checken het een keer")
println(collect(keys(hamiltonian[1])))
println(collect(keys(hamiltonian[2])))
display((convert(Array, hamiltonian[1][1, 1])))
display((convert(Array, hamiltonian[1][1, 2])))
display((convert(Array, hamiltonian[1][1, 3])))
display((convert(Array, hamiltonian[1][2, 3])))
display((convert(Array, hamiltonian[1][3, 3])))

display(reshape(convert(Array, hamiltonian[1][1, 1]), 2, 2))
display(reshape(convert(Array, hamiltonian[1][1, 2]), 2, 4))
display(reshape(convert(Array, hamiltonian[1][1, 3]), 2, 2))
display(reshape(convert(Array, hamiltonian[2][1, 3]), 2, 2))
display(reshape(convert(Array, hamiltonian[1][2, 3]), 2, 4))
display(reshape(convert(Array, hamiltonian[1][3, 3]), 2, 2))

println("asymmetric")
hamiltonian = get_thirring_hamiltonian(0.5, 0, 0.5)

println(collect(keys(hamiltonian[1])))
println(collect(keys(hamiltonian[2])))
display(reshape(convert(Array, hamiltonian[1][1, 1]), 2, 2))
display(reshape(convert(Array, hamiltonian[1][1, 2]), 2, 2))
display(reshape(convert(Array, hamiltonian[1][1, 3]), 2, 2))
display(reshape(convert(Array, hamiltonian[1][1, 4]), 2, 2))
display(reshape(convert(Array, hamiltonian[2][1, 4]), 2, 2))
display(reshape(convert(Array, hamiltonian[1][2, 4]), 2, 2))
display(reshape(convert(Array, hamiltonian[1][3, 4]), 2, 2))
display(reshape(convert(Array, hamiltonian[1][4, 4]), 2, 2))
