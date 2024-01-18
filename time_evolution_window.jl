using LinearAlgebra
# using Base
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics

# include(raw"C:\Users\Sande\Documents\School\0600 - tweesis\Code\MPSKit.jl\src\MPSKit.jl")
# include(raw"C:\Users\Sande\Documents\School\0600 - tweesis\Code\MPSKitModels.jl\src\MPSKitModels.jl")

include("get_thirring_hamiltonian_symmetric_separate.jl")
include("get_thirring_hamiltonian_window.jl")
include("get_groundstate.jl")


function spatial_ramping_lin(i, i_start, i_end)
    if i < i_start
        return 0
    elseif i < i_end
        return (i-i_start)/(i_end-i_start)
    end
    return 1.0
end

function spatial_ramping_tanh(i, i_middle, κ)
    return tanh((i-i_middle)*κ)
end

function spatial_ramping_S(i, i_middle, κ)
    return 1 - 1/(1+exp(2*κ*(i-i_middle)))
end


# function myfinalize(t,Ψ,H,env,tobesaved,O,gl,gs,gr,timefun)
# 	zs = map(i->expectation_value(Ψ,O,i),0:length(Ψ)+1)
# 	hs = [gl*timefun(t),(timefun(t).*gs)...,gr*timefun(t)]
# 	push!(tobesaved,[t,expectation_value(Ψ,H(t),env),zs,hs])
# 	return Ψ,env
# end

# function save_energy(t,Ψ,H,E,energies)
#     println("energy is $(E)")
#     println("energy2 is $(expectation_value(Ψ.window, H.middle(t)))")
#     push!(energies, expectation_value(Ψ.window, H.middle(t)))
# end

N = 20 # Number of sites
i_b = div(N,2)
κ = 0.5
lijst_ramping = [spatial_ramping_S(i, i_b, κ) for i in 1:N]

# spatial_sweep = i_end-i_start

dt = 0.01
max_time_steps = 100 #3000 #7000
t_end = dt*max_time_steps

am_tilde_0 = 0.0
Delta_g = 0.0 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie
v = 0.0
v_max = 1.5


RAMPING_TIME = 10
f(t) = min(v_max, t/RAMPING_TIME)
f0(t) = 0.0

truncation = 1.5

(Hopping_term, Mass_term, Interaction_term, Interaction_v_term, Interaction_v_term_window, Mass_term_window) = get_thirring_hamiltonian_window(1.0, 1.0, 1.0, N, lijst_ramping)
H_without_v = Hopping_term + am_tilde_0*Mass_term + Delta_g*Interaction_term

(gs_mps, gs_envs) = get_groundstate(am_tilde_0, Delta_g, v, [5 10], truncation, 8.0; number_of_loops = 2)

Ψ = WindowMPS(gs_mps,N); # state is a windowMPS

#=
# left Hamiltonian

Ht_left_td = 0*Interaction_v_term
Ht_left = SumOfOperators([H_without_v, Ht_left_td])
H_left = SumOfOperators([H_without_v, Ht_left_td])
# Ht_left = SumOfOperators([H_without_v, TimedOperator(Interaction_v_term,f0)])

# middle Hamiltonian
# H_mid_base = repeat(H_without_v,N)
H_mid_base = H_without_v

J = [1.0 -1.0]

#H_mid_v = @mpoham sum(spatial_ramping(linearize_index(i),i_start,i_end)*Interaction_v_term for i in vertices(FiniteChain(N)))
# H_mid_v = @mpoham sum(J[i]*Interaction_v_term for i in vertices(InfiniteChain(2)))

H_mid_v = Interaction_v_term
# H_mid_v = @mpoham sum(spatial_ramping(i,i_start,i_end)*Interaction_v_term for i in vertices(InfiniteChain(N)))
Ht_mid = SumOfOperators([H_mid_base, TimedOperator(H_mid_v,f)])
H_mid = SumOfOperators([H_mid_base, H_mid_v])

# right Hamiltonian
Ht_right = SumOfOperators([H_without_v, TimedOperator(Interaction_v_term,f)])
H_right = SumOfOperators([H_without_v, Interaction_v_term])


Ht_right = SumOfOperators([H_without_v, 0*Interaction_v_term])
Ht_mid = SumOfOperators([repeat(H_without_v,N), Interaction_v_term_window])
Ht_left = SumOfOperators([H_without_v, Interaction_v_term])

WindowH = Window(Ht_left,Ht_mid,Ht_right);
# WindowH = Window(H_left,H_mid,H_left);
WindowE = environments(Ψ,WindowH);
=#
t = 0.0
tobesaved = []
energies = []
t_span    = 0:dt:t_end
number_of_timesteps = max_time_steps
# alg       = TDVP(finalize=(t,Ψ,H,E)->myfinalize(t,Ψ,H,E,tobesaved,xWindow,g,gs,0.0))
# alg       = TDVP(expalg=Lanczos())
alg       = TDVP()
Ψt = copy(Ψ)

# energies = zeros(ComplexF64, (N, number_of_timesteps))
# gs_energies = zeros(ComplexF64, (N, div(number_of_timesteps,frequency_of_VUMPS)+1))

frequency_of_saving = 10

Ht_right = H_without_v + TimedOperator(Interaction_v_term, f0)
Ht_mid = repeat(H_without_v,N) + TimedOperator(Interaction_v_term_window,f)
Ht_left = H_without_v + TimedOperator(Interaction_v_term,f)

WindowH = Window(Ht_left,Ht_mid,Ht_right);
WindowE = environments(Ψ,WindowH);

MPSs = Vector{WindowMPS}(undef,div(number_of_timesteps,frequency_of_saving)+1)

for n = 1:number_of_timesteps
    global t
    global WindowE
    global Ψt
    t += dt
    println("Timestep n = $(n), t = $(t)")

    Ψt,WindowE = timestep!(Ψt,WindowH,t,dt,alg,WindowE;leftevolve=true,rightevolve=false)

    if (n % frequency_of_saving == 0)
        MPSs[div(n,frequency_of_saving)] = Ψt
    end

    # E_values = expectation_value(Ψt.window, WindowH.middle(t))
    # energies[:,n] = E_values



    # Ht_right = SumOfOperators([H_without_v, 0*Interaction_v_term])
    # Ht_mid = SumOfOperators([repeat(H_without_v,N), f(t)*Interaction_v_term_window])
    # Ht_left = SumOfOperators([H_without_v, f(t)*Interaction_v_term])

    # mass sweep
    # Ht_left = SumOfOperators([H_without_mass, f(t)*Mass_term])
    # Ht_mid = SumOfOperators([repeat(H_without_mass,N), f(t)*Mass_term_window])
    # Ht_right = SumOfOperators([H_without_mass, 0*Mass_term])


    # Ht_right = SumOfOperators([H_without_v, TimedOperator(Interaction_v_term, f0)])
    # Ht_mid = SumOfOperators([repeat(H_without_v,N), TimedOperator(Interaction_v_term_window,f)])
    # Ht_left = SumOfOperators([H_without_v, TimedOperator(Interaction_v_term,f)])


    # window_dt,WindowE = time_evolve!(Ψt,WindowH,t_span,alg,WindowE;verbose=true,rightevolve=true)

    # if (n % frequency_of_VUMPS == 0)
    #     (groundstate_mps_local, groundstate_envs_local) = find_groundstate(Ψt.window,Ht_mid,VUMPS(maxiter = 50, tol_galerkin = 1e-12)) # For v sweep
    #     true_gs_energy = expectation_value(groundstate_mps_local.window, Ht_mid)
    #     gs_energies[div(n,frequency_of_VUMPS)] = true_gs_energy
    # end

end

@save "window_time_evolution_v_sweep_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_savefrequency_$(frequency_of_saving)" MPSs
