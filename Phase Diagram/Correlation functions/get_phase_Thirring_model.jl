using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_groundstate_energy.jl")
include("get_thirring_hamiltonian.jl")

# Gapped
am_tilde_0 = 0.4
Delta_g = -0.2
v = 0.0

# Gapless
am_tilde_0 = 0.2
Delta_g = -0.8


# Test for gs gs_energy
am_tilde_0 = 0
Delta_g = -0.7

am_tilde_0 = 0.5
Delta_g = 0.0

D_number = 1

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

D = 50

masses = [0, 0.1, 0.241, 0.405, 0.5]
delta_gs = [0.5, 0.375, 0.25, 0.125, 0, -0.2, -0.4, -0.6, -0.8, -1]

corr_functions = zeros(ComplexF64, length(masses), length(delta_gs), 200)
energies = zeros(ComplexF64, length(masses), length(delta_gs), 2)

for i = 1:length(masses)
    for j = 1:length(delta_gs)
        mass = masses[i]
        delta_g = delta_gs[j]
        println("started for mass = $i and delta(g) = $j")
        (gs_energy, corr_function) = get_groundstate_energy(mass, delta_g, 0, D)
        corr_functions[i,j,:] = corr_function
        energies[i,j,:] = gs_energy
    end
end

@save "Thirring_2023_10_18_E_and_corr" corr_functions energies

#=
Mass_number = 20
D = 50
for i = 1:Mass_number
    am_tilde_0 = 0.02*i
    println("Mass = ", am_tilde_0)
    (groundstate_energy, gs_correlation_length) = get_groundstate_energy(am_tilde_0, Delta_g, v, D)
    #gs = get_groundstate_energy(am_tilde_0, Delta_g, D)
    println(groundstate_energy)
    #Energies[i] = real(Statistics.mean(groundstate_energy))
    #Correlation_lengths[i] = gs_correlation_length
end

#spectrum = transfer_spectrum(gs)

println(Energies)
println(Correlation_lengths)
@save "Thirring_first_try_0p3_m0p2" Energies Correlation_lengths
=#

