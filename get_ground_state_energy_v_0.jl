using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_groundstate_energy.jl")
include("get_thirring_hamiltonian.jl")

D = 50

D_values = [floor(Int, 12.5*(2^(i/2))) for i = 2:9]

masses = [0, 0.241, 0.405, 0.5]
gs = LinRange(-1.5, 1.5, 13)
delta_gs = cos.((pi.-gs)/2)

for d = 1:length(D_values)
    energies = zeros(ComplexF64, length(masses), length(delta_gs), 2)
    D_bond = D_values[d]
    for i = 1:length(masses)
        for j = 1:length(delta_gs)
            mass = masses[i]
            delta_g = delta_gs[j]
            println("started for D = $D_bond, mass = $i, and delta(g) = $j")
            (gs_energy, _) = get_groundstate_energy(mass, delta_g, 0, D_bond)
            energies[i,j,:] = gs_energy
        end
    end
    @save "Thirring_E_gs_v_0_D_$D_bond" energies
end