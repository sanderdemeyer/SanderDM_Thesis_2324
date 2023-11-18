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
include("get_thirring_hamiltonian_symmetric.jl")

am_tilde_0 = 0.0
v = 0.0
trunc = 4.0
tolerance = 6.0

amount = 28
delta_gs = LinRange(-1, 6, amount)

Energies = Array{Float64, 2}(undef, 1, amount)
for j = 1:amount
    Delta_g = delta_gs[j]
    (gs_energy, _, _, _) = get_groundstate_energy_dynamic(am_tilde_0, Delta_g, v, trunc, tolerance; number_of_loops = 4, final_iterations = 30, D_start = 3, mps_start = 0)
    Energies[j] = gs_energy
end

@save "Comparison_XXZ_dynamic_trunc_4" Energies delta_gs

