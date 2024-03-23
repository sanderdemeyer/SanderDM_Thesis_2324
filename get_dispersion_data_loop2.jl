using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots

include("get_groundstate.jl")

function get_dispersion_data(am_tilde_0, Delta_g; plotting = false)
    # am_tilde_0 = 0.3
    # Delta_g = -0.45
    v = 0.0
    bounds_k = pi/2
    trunc = 3.0

    (mps,envs) = get_groundstate(am_tilde_0, Delta_g, v, [20 50], 4.0, 7.0)

    # @load "SanderDM_Thesis_2324/gs_mps_trunc_$(trunc)_mass_$(am_tilde_0)_v_$(v)_Delta_g_$(Delta_g)" mps envs

    hamiltonian = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v)
    gs_energy = expectation_value(mps, hamiltonian);

    k_values = LinRange(-bounds_k, bounds_k,35)
    (energies,Bs) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,mps,envs, sector = Irrep[U₁](1));
    (anti_energies,anti_Bs) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,mps,envs, sector = Irrep[U₁](-1));
    (zero_energies,zero_Bs) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,mps,envs, sector = Irrep[U₁](0));

    print("Done with excitations")

    energies_real = [real(e) for e in energies]
    anti_energies_real = [real(e) for e in anti_energies]
    zero_energies_real = [real(e) for e in zero_energies]

    if plotting
        plt = plot(k_values, energies_real, label="particles", xlabel="k", ylabel="energies", linewidth=2)
        plot!(k_values, anti_energies_real, label="anti-particles", linewidth=2)
        plot!(k_values, zero_energies_real, label="zero-particles", linewidth=2)
        display(plt)
        savefig(plt, "SanderDM_Thesis_2324/Dispersion_Delta_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)_all_sectors.png")
    end

    @save "SanderDM_Thesis_2324/Dispersion_Delta_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)_trunc_$(trunc)_all_sectors" gs_energy energies Bs anti_energies anti_Bs zero_energies zero_Bs bounds_k
end

for m = [0.01 0.05 0.1 0.2]
    for deltag = [0.0 0.1 0.2 0.3]
        get_dispersion_data(m, deltag)
    end
end
