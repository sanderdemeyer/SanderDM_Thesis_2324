using LinearAlgebra
using Base
using JLD2
using MPSKit
using MPSKitModels
using TensorKit
using Statistics

include("get_thirring_hamiltonian_symmetric_separate.jl")
include("get_groundstate.jl")


function uniform_time_evolution(dt)
    Delta_g = 0.0
    v = 0.0
    m0 = 0.1
    m_end = 0.4
    D = 50

    RAMPING_TIME = 5
    # dt = 0.1
    t_end = (m_end-m0)*RAMPING_TIME*1.5

    timesteps = div(t_end, dt)
    println("Performing $(timesteps) timesteps")

    function f_v(t)
        #return min(0.4, 0.2+t/10) # Fast
        return min(v_max, t/RAMPING_TIME)
    end

    function f(t)
        return min(m_end, m0+t/RAMPING_TIME)
    end

    saving_time = 1.0

    function my_finalize(t, Ψ, H, envs, Es)
        if ((t) % (frequency_of_saving*dt) == 0.0)
            push!(Es, expectation_value(Ψ, H(t),envs))
        end
        return (Ψ, envs)
    end

    (Hopping_term, Mass_term, Interaction_term, Interaction_v_term) = get_thirring_hamiltonian_symmetric_separate(1.0, Delta_g, v) # For v sweep
    H_without_mass = Hopping_term + Interaction_term + Interaction_v_term

    truncation = 2.5

    println("started")
    (mps, envs) = get_groundstate(m0, Delta_g, v, [20 50], truncation, truncation+3.0)

    tot_bonddim = 0
    for i = 1:2
        # global tot_bonddim
        tot_bonddim += dims((mps.AL[i]).codom)[1] + dims((mps.AL[i]).dom)[1]
    end
    println("Tot bonddim is $tot_bonddim")

    Es = []
    alg = TDVP2(; trscheme = truncdim(D))#; finalize = (t, Ψ, H, envs) -> my_finalize(t, Ψ, H, envs, Es)) #, finalize=(t, Ψ, H, envs) -> my_finalize(t, Ψ, H, envs, Es))
    t_span = 0:dt:t_end

    Es_time_evolve = []
    Es_local = []
    fidelities = []

    # (mps, envs) = time_evolve!(mps, H, t_span, alg, envs; verbose=true);
    for j = 1:timesteps
        t = j*dt
        # global mps
        # global envs

        println("Timestep j = $(j), t = $(t)")

        (mps, envs) = timestep(mps, H_without_mass + Mass_term*f(t), dt, TDVP()) #geen envs meegeven V SWEEP
        energy = expectation_value(mps, H_without_mass + Mass_term*f(t))
        push!(Es, energy)

        if j % 3 == 0
            (groundstate_mps_local, groundstate_envs_local) = find_groundstate(mps,H_without_mass + Mass_term*f(t),VUMPS(maxiter = 50, tol_galerkin = 1e-12)) # For v sweep
            true_gs_energy = expectation_value(groundstate_mps_local, H_without_mass + Mass_term*f(t))

            fidelity = @tensor groundstate_mps_local.AC[1][1 2; 3] * conj(mps.AC[1][1 2; 3])

            push!(Es_time_evolve, energy)
            push!(Es_local, true_gs_energy)
            push!(fidelities, fidelity)
        end
    end
    @save "mass_time_evolution_dt_$(dt)_D_$(D)_trunc_$(truncation)_mass_$(m0)_to_$(m_end)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_tend_$(t_end)_v_$(v)" Es Es_time_evolve Es_local fidelities
end

for dt = [0.5 1.0]
    println("Started for dt = $(dt)")
    uniform_time_evolution(dt)
end