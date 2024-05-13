using LinearAlgebra
using KrylovKit
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_thirring_hamiltonian_symmetric_separate.jl")
include("get_thirring_hamiltonian_symmetric.jl")
include("get_thirring_hamiltonian_window.jl")
include("get_groundstate.jl")
include("get_occupation_number_matrices.jl")


function spatial_ramping_S(i, ib, κ)
    value = 1.0 - 1/(1+exp(2*κ*(i-ib)))
    if value < 1e-4
        return 0.0
    elseif value > 1 - 1e-4
        return 1.0
    end
    return value
end

function my_finalize(t, Ψ, H, envs, name)
    check = (t) % (frequency_of_saving*dt)
    if (((check < 1e-5) || (frequency_of_saving*dt - check < 1e-5)) && (t != 0.0))
        println("Currently saving for t = $(t)")
        file = jldopen(name*"/$(t).jld2", "w")
        file["MPSs"] = copy(Ψ)
        file["Es"] = expectation_value(Ψ, H(t), envs)
        file["sigmaz"] = expectation_value(Ψ.middle, Sz)
        close(file)
    end
    return (Ψ, envs)
end

function time_evolution_bhole()
    N = parse(Int, ARGS[1])
    κ = parse(Float64, ARGS[2])
    dt = parse(Float64, ARGS[3])
    number_of_timesteps = parse(Int, ARGS[4])
    am_tilde_0 = parse(Float64, ARGS[5])
    Delta_g = parse(Float64, ARGS[6])
    v_max = parse(Float64, ARGS[7])

    @assert N % 2 == 0
    frequency_of_saving = 3
    RAMPING_TIME = 5
    v = 0.0
    D = 4
    truncation = 1.5
    ib = div(2*N,3)

    lijst_ramping = [spatial_ramping_S(i, ib, κ) for i = 1:N]
    f(t) = sign(v_max)*min(abs(v_max), t/RAMPING_TIME)
    f0(t) = 0.0


    (Hopping_term, Mass_term, Interaction_term, Interaction_v_term, Interaction_v_term_window, Mass_term_window) = get_thirring_hamiltonian_window(1.0, 1.0, 1.0, N, lijst_ramping)


    spin = 1//2
    pspace = U1Space(i => 1 for i in (-spin):spin)
    Plus_space = U1Space(1 => 1)
    Min_space = U1Space(-1 => 1)
    trivspace = U1Space(0 => 1)
    S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], trivspace ⊗ pspace, pspace ⊗ Plus_space)
    S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace ⊗ trivspace)
    S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], Plus_space ⊗ pspace, pspace ⊗ Plus_space)
    S⁺2 = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Min_space ⊗ pspace, pspace ⊗ trivspace)
    S⁻2 = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], trivspace ⊗ pspace, pspace ⊗ Min_space)
    S_z_symm2 = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], Min_space ⊗ pspace, pspace ⊗ Min_space)
    MPO1 = (im*0.5) * (repeat(MPOHamiltonian([S⁺, S_z_symm, S⁻]),N));
    MPO2 = -(im*0.5) * (repeat(MPOHamiltonian([S⁻2, S_z_symm2, S⁺2]),N));
    Interaction_v_term = (im*0.5) * (repeat(MPOHamiltonian([S⁺, S_z_symm, S⁻]),2)-repeat(MPOHamiltonian([S⁻2, S_z_symm2, S⁺2]),2));
    Interaction_v_term_window = MPO1+MPO2

    for i = 1:N
        for k = 1:Interaction_v_term_window.odim-1
            Interaction_v_term_window[i][k,Interaction_v_term_window.odim] *= lijst_ramping[i]
        end
    end

    H_without_v = Hopping_term + am_tilde_0*Mass_term + Delta_g*Interaction_term

    H1 = Window(H_without_v, repeat(H_without_v, div(N,2)), H_without_v)
    H2 = Window(0*Interaction_v_term, Interaction_v_term_window, Interaction_v_term)
    WindowH = LazySum([H1, MultipliedOperator(H2, f)])


    # Stuff for files and directories
    # name = "bhole_time_evolution_variables_N_$(N)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_D_$(D)_savefrequency_$(frequency_of_saving)"
    name = "bhole_test_time_evolution_variables_N_$(N)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(0)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_D_$(D)_savefrequency_$(frequency_of_saving).jld2"

    if !isdir(name)
        mkdir(name)
        t_prev = 0.0
        (gs_mps, _) = get_groundstate(am_tilde_0, Delta_g, v, [50 100], truncation, truncation+3.0; number_of_loops=3)
        Ψ = WindowMPS(gs_mps,N; fixleft=true, fixright=false); # state is a windowMPS
    else
        files = sort(readdir(name))
        t_prev = parse(Float64, files[end][1:end-5])
        file = jldopen(name*"/$(t_prev).jld2", "r")
        Ψ = file["MPSs"]
        close(file)    
    end

    t_end = t_prev + dt*number_of_timesteps
    envs = environments(Ψ, WindowH);

    # Algorithms for time evolution
    left_alg = right_alg = TDVP()
    middle_alg =  TDVP2(; trscheme=truncdim(D));
    alg = WindowTDVP(;left=left_alg,middle=middle_alg,right=right_alg,
                finalize=(t, Ψ, H, envs) -> my_finalize(t, Ψ, H, envs, name*"_new"));
    t_span = t_prev:dt:t_end
    t = t_prev

    # Perform time evolution
    (Ψ, envs) = time_evolve!(Ψ, WindowH, t_span, alg, envs; verbose=true);

    println("done")
end