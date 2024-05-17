using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots

include("get_groundstate.jl")

am_tilde_0 = 0.06
Delta_g = -0.15
v = 0.0
bounds_k = pi/2
trunc = 2.5

L = 180
nodp = div(L,2)-1

include_2 = false

isnew = true

(mps,envs) = get_groundstate(am_tilde_0, Delta_g, v, [50 100], trunc, trunc+3.0; new = isnew)

# @load "gs_mps_trunc_$(trunc)_mass_$(am_tilde_0)_v_$(v)_Delta_g_$(Delta_g)" mps envs

hamiltonian = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v; new = isnew)
gs_energy = expectation_value(mps, hamiltonian);

k_values = [(pi)/nodp*i - (pi/2) for i = 0:nodp-1]

# N = 69
# k_values = [(2*pi)/N*i - pi for i = 0:N-1]./2

(energies,Bs) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,mps,envs, sector = Irrep[U₁](1));
# (anti_energies,anti_Bs) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,mps,envs, sector = Irrep[U₁](-1));
# (zero_energies,zero_Bs) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,mps,envs, sector = Irrep[U₁](0));
if include_2
    (energies_2,_) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,mps,envs, sector = Irrep[U₁](2));
    (energies_m2,_) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,mps,envs, sector = Irrep[U₁](-2));
end

print("Done with excitations")

energies_real = [real(e) for e in energies]
# anti_energies_real = [real(e) for e in anti_energies]
# zero_energies_real = [real(e) for e in zero_energies]
if include_2
    energies_2_real = [real(e) for e in energies_2]
    energies_m2_real = [real(e) for e in energies_m2]
end    

name = "SanderDM_Thesis_2324/Dispersion_Data/Full range/Dispersion_Delta_N_$(L)_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)_trunc_$(trunc)_all_sectors"
if isnew
    name = name*"_newv"
end

plt = plot(k_values, energies_real, label="particles", xlabel="k", ylabel="energies", linewidth=2)
plot!(k_values, anti_energies_real, label="anti-particles", linewidth=2)
plot!(k_values, zero_energies_real, label="zero-particles", linewidth=2)
display(plt)
savefig(plt, name*".png")

if include_2
    @save name*"_and_more" gs_energy energies Bs anti_energies anti_Bs zero_energies zero_Bs bounds_k energies_2 energies_m2
else
    @save name gs_energy energies Bs anti_energies anti_Bs zero_energies zero_Bs bounds_k
    @save name gs_energy energies Bs anti_energies anti_Bs bounds_k
    @save name gs_energy energies Bs 
    using DelimitedFiles
    name_energy_save = "SanderDM_Thesis_2324/Dispersion_Data/Full range/Energies_Dispersion_Delta_L_$(L)_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)_trunc_$(trunc).csv"

    writedlm(name_energy_save, real.(energies))
end

Δ = (minimum(energies_real)-minimum(anti_energies_real))/2
# Δ2 = (minimum(energies_2_real)-minimum(energies_m2_real))/2

E_theory = [-v*sin(2*k)/2 + sqrt(am_tilde_0^2+sin(k)^2) for k in k_values]

plt = plot(k_values, energies_real .- Δ, label="particles", xlabel="k", ylabel="energies", linewidth=2)
plot!(k_values, anti_energies_real .+ Δ, label="anti-particles", linewidth=2)
plot!(k_values, E_theory, label="theory", linewidth=2)
# plot!(k_values, energies_2_real .- 2*Δ, label="+2", xlabel="k", ylabel="energies", linewidth=2)
# plot!(k_values, energies_m2_real .+ 2*Δ, label="-2", linewidth=2)
plot!(k_values, zero_energies_real, label="zero-particles", linewidth=2)
display(plt)
