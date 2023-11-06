using TensorOperations
using TensorKit
using MPSKit
using MPSKitModels
using LinearAlgebra: norm

using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit

include("get_groundstate_energy_dynamic.jl")
include("get_thirring_hamiltonian_symmetric.jl")
include("phase_diagram_symmetric.jl")

v = 0.0

amount_mass = 5
amount_delta_g = 5

mass_range = LinRange(0, 0.3, amount_mass)
delta_g_range = LinRange(-1.0, 0.0, amount_delta_g)

for j = 1:amount_delta_g
    for i = 1:amount_mass
        mass = mass_range[i]
        delta_g = delta_g_range[j]
        println("started for mass = $mass and delta_g = $delta_g")
        phase_diagram_symmetric(mass, delta_g, v)
    end
end