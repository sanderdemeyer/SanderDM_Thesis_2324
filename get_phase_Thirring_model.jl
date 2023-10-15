using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit

include("get_groundstate_energy.jl")
include("get_thirring_hamiltonian.jl")

# Gapped
am_tilde_0 = 0.4
Delta_g = -0.2
v = 0.0

# Gapless
am_tilde_0 = 0
Delta_g = -0.8


D_number = 1

Energies = Array{Float64, 2}(undef, 1, D_number)
Correlation_lengths = Array{Float64, 2}(undef, 1, D_number)
for i = 1:D_number
    D = i*10
    println("D = ", D)
    (gs_energy, gs_correlation_length) = get_groundstate_energy(am_tilde_0, Delta_g, v, D)
    #gs = get_groundstate_energy(am_tilde_0, Delta_g, D)
    Energies[i] = real(gs_energy[0] + gs_energy[1])/2
    Correlation_lengths[i] = gs_correlation_length
end

#spectrum = transfer_spectrum(gs)

println(Energies)
println(Correlation_lengths)
@save "Thirring_first_try_0p3_m0p2" Energies Correlation_lengths