using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_groundstate.jl")

am_tilde_0 = 0.7
Delta_g = -0.1
v = 0.01
bounds = pi/16

(mps,envs) = get_groundstate(am_tilde_0, Delta_g, v, [15 50], 4.0, 7.0)
hamiltonian = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v)
gs_energy = expectation_value(mps, hamiltonian);


k_values = LinRange(-bounds, bounds,35)
(energies,Bs) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,mps,envs, sector = Irrep[U₁](0));

print("Done with excitations")

@save "Dispersion_Delta_m_0.7_delta_g_-0.1_v_0.01" gs_energy bounds energies Bs

