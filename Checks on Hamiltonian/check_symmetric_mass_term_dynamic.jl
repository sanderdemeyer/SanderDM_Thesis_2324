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

include("get_groundstate_energy_dynamic.jl")
include("get_thirring_hamiltonian.jl")
include("get_thirring_hamiltonian_symmetric.jl")


amount = 20
mass_range = LinRange(0.0, 1.0, amount)
Energies = Array{Float64, 2}(undef, 1, amount)
trunc = 3.0

for index = 1:amount
    mass = mass_range[index]
    println("mass is $mass")
    gs_energy = get_groundstate_energy_dynamic(mass, 0, 0, trunc, 6.0)

    println("Energy is $gs_energy")
    Energies[index] = real(gs_energy)
end

@save "Check_mass_term_symmetric_trunc_4" Energies