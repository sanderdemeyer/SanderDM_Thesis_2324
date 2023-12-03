using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_groundstate.jl")

bounds = pi/12
bound = fill(bounds, 1, 1)

for v_term = 0:1
    for delta_g_index = 1:5
        for mass_index = 1:6
            v = 0.15 + 0.6*v_term
            Delta_g = -0.15*delta_g_index
            am_tilde_0 = 0.1*mass_index

            (mps,envs) = get_groundstate(am_tilde_0, Delta_g, v, [15 50], 4.5, 7.0)
            hamiltonian = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v)
            gs_energy = expectation_value(mps, hamiltonian);

            k_values = LinRange(-bounds, bounds,51)
            (energies,Bs) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,mps,envs, sector = Irrep[U₁](1));

            print("Done with excitations")

            @save "Dispersion_m_$am_tilde_0 _delta_g_$Delta_g _v_$v" gs_energy bounds energies Bs bound
        end
    end
end
