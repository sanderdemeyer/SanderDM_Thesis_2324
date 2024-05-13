using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_groundstate.jl")

bounds = pi/72
bound = fill(bounds, 1, 1)


factor_max = 48
k_min = bounds/(factor_max*4)

k_values = [0.0]

for factor = 1:factor_max
    for j = 1:4
        if !(j*factor*k_min in k_values)
            push!(k_values, j*factor*k_min)
            push!(k_values, -j*factor*k_min)
        end
    end
end

println(length(k_values))
println(k_values)
# NEW: for v = 0
v = 0.0
trunc = 2.5

for delta_g_index = [2 4]
    for mass_index = 2:6
        Delta_g = -0.15*delta_g_index
        am_tilde_0 = 0.1*mass_index

        (mps,envs) = get_groundstate(am_tilde_0, Delta_g, v, [15 50], trunc, 7.0)
        hamiltonian = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v)
        gs_energy = expectation_value(mps, hamiltonian);

        (energies,Bs) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,mps,envs, sector = Irrep[U₁](-1));

        print("Done with excitations")

        @save "Dispersion_pi_over_72_v_0_U1_-1_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)_trunc_$(trunc)" gs_energy bounds energies Bs bound
    end
end

# OLD: for v != 0
# for v_term = 0:1
#     for delta_g_index = 4:5
#         for mass_index = 1:6
#             v = 0.15 + 0.6*v_term
#             Delta_g = -0.15*delta_g_index
#             am_tilde_0 = 0.1*mass_index

#             (mps,envs) = get_groundstate(am_tilde_0, Delta_g, v, [15 50], 3.5, 7.0)
#             hamiltonian = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v)
#             gs_energy = expectation_value(mps, hamiltonian);

#             (energies,Bs) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,mps,envs, sector = Irrep[U₁](1));

#             print("Done with excitations")

#             @save "Dispersion_pi_over_72_m_$am_tilde_0 _delta_g_$Delta_g _v_$v" gs_energy bounds energies Bs bound
#         end
#     end
# end
