using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots

include("get_thirring_hamiltonian.jl")

am_tilde_0 = 0.3
Delta_g = -0.45
v = 0.0
trunc = 2.5
bounds = pi/2

@load "SanderDM_Thesis_2324/gs_mps_wo_symmetries_trunc_$(trunc)_mass_$(am_tilde_0)_v_$(v)_Delta_g_$(Delta_g)" mps envs

hamiltonian = get_thirring_hamiltonian(am_tilde_0, Delta_g, v)
gs_energy = expectation_value(mps, hamiltonian);

println("Energy is $gs_energy")

k_values = LinRange(-bounds, bounds,35)
(energies,Bs) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,mps,envs);

print("Done with excitations")

energies_real = [real(e) for e in energies]
plt = plot(k_values, energies_real, label="particles", xlabel="k", ylabel="energies", linewidth=2)
display(plt)

@save "SanderDM_Thesis_2324/Dispersion_asymm_Delta_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)" gs_energy bounds energies Bs

