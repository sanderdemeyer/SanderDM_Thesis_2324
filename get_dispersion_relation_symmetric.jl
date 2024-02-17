using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_groundstate.jl")

am_tilde_0 = 1.0
Delta_g = -0.5
v = 0.9
bounds = pi/2

(mps,envs) = get_groundstate(am_tilde_0, Delta_g, v, [15 50], 4.0, 7.0)
hamiltonian = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v)
gs_energy = expectation_value(mps, hamiltonian);


k_values = LinRange(-bounds, bounds,35)
(energies,Bs) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,mps,envs, sector = Irrep[U₁](1));

print("Done with excitations")

@save "Dispersion_Delta_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)_irrep_1" gs_energy bounds energies Bs

