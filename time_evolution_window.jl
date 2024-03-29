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
include("get_occupation_number_matrices.jl")


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
    value = 1 - 1/(1+exp(2*κ*(i-i_middle)))
    if value < 1e-4
        return 0.0
    elseif value > 1 - 1e-4
        return 1.0
    end
    return value
end

function my_finalize(t, Ψ, H, envs, name)
    if ((t) % (frequency_of_saving*dt) == 0.0)
        println("Currently saving for t = $(t)")
        if !isfile(name)
            println("File not found")
        end
        @assert (t != 0.0)
        file = jldopen(name*"$(t).jld2", "w")
        file["MPSs"] = copy(Ψ)
        file["Es"] = expectation_value(Ψ, H(t), envs)
        close(file)
    end
    return (Ψ, envs)
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

N = 70 # Number of sites
D = 50

@assert N % 2 == 0

i_b = div(2*N,3)
κ = 0.5
lijst_ramping = [spatial_ramping_S(i, i_b, κ) for i in 1:N]

# spatial_sweep = i_end-i_start

dt = 0.1
max_time_steps = 1500 #3000 #7000
t_end = dt*max_time_steps

am_tilde_0 = 0.03
Delta_g = 0.0 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie
v = 0.0
v_max = 1.5


RAMPING_TIME = 5
f(t) = min(v_max, t/RAMPING_TIME)
f0(t) = 0.0

truncation = 2.0

(Hopping_term, Mass_term, Interaction_term, Interaction_v_term, Interaction_v_term_window, Mass_term_window) = get_thirring_hamiltonian_window(1.0, 1.0, 1.0, N, lijst_ramping)
H_without_v = Hopping_term + am_tilde_0*Mass_term + Delta_g*Interaction_term

(gs_mps, gs_envs) = get_groundstate(am_tilde_0, Delta_g, v, [50 100], truncation, truncation+3.0; number_of_loops=7)

Ψ = WindowMPS(gs_mps,N); # state is a windowMPS
t = 0.0
tobesaved = []
energies = []
t_span    = 0:dt:t_end
number_of_timesteps = max_time_steps
# alg       = TDVP(finalize=(t,Ψ,H,E)->myfinalize(t,Ψ,H,E,tobesaved,xWindow,g,gs,0.0))
# alg       = TDVP(expalg=Lanczos())
alg       = TDVP()
Ψt = copy(Ψ)

σ = 2*(2*pi/N)
x₀ = 40

# energies = zeros(ComplexF64, (N, number_of_timesteps))
# gs_energies = zeros(ComplexF64, (N, div(number_of_timesteps,frequency_of_VUMPS)+1))

frequency_of_saving = 50

# Ht_right = LazySum([H_without_v, MultipliedOperator(Interaction_v_term, f)])
# Ht_mid = LazySum([repeat(H_without_v,div(N,2)), MultipliedOperator(Interaction_v_term_window,f)])
# Ht_left = LazySum([H_without_v, MultipliedOperator(Interaction_v_term,f0)])
# WindowH = Window(Ht_left,Ht_mid,Ht_right);
#MPO ham voor finite deel, altijd infinite lattice. 
#window van multipliedoperator
#LazySum([Window(, )])
H1 = Window(H_without_v, repeat(H_without_v, div(N,2)), H_without_v)
H2 = Window(Interaction_v_term, Interaction_v_term_window, 0*Interaction_v_term)
WindowH = LazySum([H1, MultipliedOperator(H2, f)])


envs = environments(Ψ,WindowH);

MPSs = Vector{WindowMPS}(undef,div(number_of_timesteps,frequency_of_saving))
WindowMPSs = Vector{FiniteMPS}(undef,div(number_of_timesteps,frequency_of_saving))
# Es = Vector{Vector{ComplexF64}}(undef,div(number_of_timesteps,frequency_of_saving))
# occ_numbers = Vector{Vector{Float64}}(undef,div(number_of_timesteps,frequency_of_saving))
# Es = zeros(ComplexF64, div(number_of_timesteps,frequency_of_saving), N)
Es = []
occ_numbers = zeros(Float64,div(number_of_timesteps,frequency_of_saving), div(N,2)-1)

testt = [0.0]

name = "window_time_evolution_variables_v_sweep_N_$(N)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_savefrequency_$(frequency_of_saving)"

if isfile(name*".jld2")
    println("Warning - file already exists -- appending with new")
    count_files = 1
    while isfile(name*"_new$(count_files).jld2")
        global count_files
        count_files += 1
    end
    name = name * "_new$(count_files)"
end
name = name * "/"
# name = name * ".jld2"


left_alg = right_alg = TDVP()
# middle_alg =  TDVP2(; trscheme=truncdim(D));
middle_alg =  TDVP2(; trscheme=truncbelow(10^(-truncation)));
alg = WindowTDVP(;left=left_alg,middle=middle_alg,right=right_alg,
            finalize=(t, Ψ, H, envs) -> my_finalize(t, Ψ, H, envs, name));
t_span = 0:dt:t_end

if !isdir(name)
    mkdir(name)
end

file = jldopen(name*"0.0.jld2", "w")
file["MPSs"] = copy(Ψ)
file["Es"] = expectation_value(Ψ, WindowH(t), envs)
close(file)


Ψ, envs = time_evolve!(Ψ, WindowH, t_span, alg, envs; verbose=true);

# @save "test_window_time_evolution_test_v_sweep_N_$(N)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_savefrequency_$(frequency_of_saving)"
