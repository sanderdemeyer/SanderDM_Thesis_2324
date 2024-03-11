using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_groundstate.jl")

am_tilde_0 = 1.0
Delta_g = -0.5
v = 0.0
bounds = pi/2

# (mps,envs) = get_groundstate(am_tilde_0, Delta_g, v, [15 50], 4.0, 7.0)

@load "SanderDM_Thesis_2324/gs_mps_trunc_$(trunc)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)" mps envs

hamiltonian = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v)
gs_energy = expectation_value(mps, hamiltonian);

k_values = LinRange(-bounds, bounds,35)
(energies,Bs) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,mps,envs, sector = Irrep[U₁](1));
(anti_energies,anti_Bs) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,mps,envs, sector = Irrep[U₁](-1));

print("Done with excitations")

energies_real = [real(e) for e in energies]
anti_energies_real = [real(e) for e in anti_energies]

plt = plot(k_values, energies_real, label="particles", xlabel="k", ylabel="energies", linewidth=2)
plot!(k_values, anti_energies_real, label="anti-particles", xlabel="k", ylabel="anti-energies", linewidth=2)
display(plt)

# @save "Dispersion_Delta_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)_irrep_1" gs_energy bounds energies Bs anti_energies anti_Bs

