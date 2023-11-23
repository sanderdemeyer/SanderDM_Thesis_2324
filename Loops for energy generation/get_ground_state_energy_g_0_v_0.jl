using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_groundstate_energy.jl")
include("get_thirring_hamiltonian.jl")

D = 50
m_range = 20
masses = LinRange(0,1,m_range)
println(masses)
energies = zeros(ComplexF64, m_range, 2)
masses_array = zeros(ComplexF64, m_range, 1)

for index = 1:m_range
    am_tilde_0 = masses[index]
    (energy, _) = get_groundstate_energy(am_tilde_0, 0, 0, D)
    energies[index,:] = energy
    masses_array[index,:] = am_tilde_0
end

@save "Thirring_groundstate_energy_g_0_v_0" energies masses

