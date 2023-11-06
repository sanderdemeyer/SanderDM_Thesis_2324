using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_groundstate_energy.jl")
include("get_thirring_hamiltonian_symmetric.jl")
include("get_thirring_hamiltonian.jl")

am_tilde_0 = 0.0
Delta_g = 0.0
v = 0.9
D = 12

#

println("Really started")

state = InfiniteMPS([ℂ^2, ℂ^2],[ℂ^D, ℂ^D])
hamiltonian = get_thirring_hamiltonian(am_tilde_0, Delta_g, v)

(groundstate,envs,_) = find_groundstate(state,hamiltonian,VUMPS(maxiter = 25));
gs_energy = expectation_value(groundstate, hamiltonian);

k_values = LinRange(-pi/6,pi/6,11)

(energies,Bs) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,groundstate,envs);


@save "asymmetric_2023_10_24_dispersion_relation_small_k_values_m_0p5_v_0p9" gs_energy k_values energies Bs