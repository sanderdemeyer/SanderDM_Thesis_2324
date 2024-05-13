using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots

am_tilde_0 = 0.3
Delta_g = -0.45
v = 0.0
trunc = 2.5

@load "SanderDM_Thesis_2324/Dispersion_asymm_Delta_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)" gs_energy bounds energies Bs
energies_asymm = energies

trunc  = 3.0
@load "SanderDM_Thesis_2324/Dispersion_Delta_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)_trunc_$(trunc)_all_sectors" gs_energy energies Bs anti_energies anti_Bs zero_energies zero_Bs bounds_k


k0 = 18
gap = (energies[k0]+anti_energies[k0])/2
Δ = (energies[k0]-anti_energies[k0])/2
gap_asymm = energies_asymm[k0]

k_values = LinRange(-bounds_k, bounds_k, 35)

plt = plot(k_values, real.(energies_asymm))
plot!(k_values, real.(energies .- Δ))
plot!(k_values, real.(anti_energies .+ Δ))
display(plt)