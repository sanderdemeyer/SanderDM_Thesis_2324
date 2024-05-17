using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using DelimitedFiles

mass = 0.06
delta_g = -0.15
v = 0.0
trunc = 2.5

@load "Code/SanderDM_Thesis_2324/Dispersion_Data/Full range/Dispersion_Delta_m_$(mass)_delta_g_$(delta_g)_v_$(v)_trunc_$(trunc)_all_sectors_newv" energies Bs anti_energies anti_Bs

energies = real(energies) .+ delta_g
anti_energies = real(anti_energies) .- delta_g

plt = plot(1:69, energies)
display(plt)

name_new = "Code/SanderDM_Thesis_2324/Dispersion_Data/Full range/Energies_Dispersion_Delta_m_$(mass)_delta_g_$(delta_g)_v_$(v)_trunc_$(trunc).csv"
writedlm(name_new, energies)
