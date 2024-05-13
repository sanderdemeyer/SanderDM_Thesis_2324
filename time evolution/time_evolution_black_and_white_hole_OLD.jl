using LinearAlgebra
# using Base
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_thirring_hamiltonian_symmetric_separate.jl")
include("get_thirring_hamiltonian_symmetric.jl")
include("get_thirring_hamiltonian_window.jl")
include("get_groundstate.jl")
include("get_occupation_number_matrices.jl")


saving_time = 1.0

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

function spatial_ramping_S(i, ib, iw, κ)
    value = 0.5 - (1/(1+exp(2*κ*(i-ib)))+1/(1+exp(2*κ*(iw-i))))/2
    if value < 1e-4
        return 0.0
    elseif value > 1 - 1e-4
        return 1.0
    end
    return value
end

function my_finalize(t, Ψ, H, envs, name)
    if (((t) % (frequency_of_saving*dt) == 0.0) && (t != 0.0))
        println("Currently saving for t = $(t)")
        file = jldopen(name*"/$(t).jld2", "w")
        file["MPSs"] = copy(Ψ)
        file["Es"] = expectation_value(Ψ, H, envs)
        # file["Es"] = expectation_value(Ψ, H(t), envs)
        file["sigmaz"] = expectation_value(Ψ.middle, Sz)
        close(file)
    end
    return (Ψ, envs)
end

N = 120 # Number of sites
D = 20

@assert N % 2 == 0

κ = 0.5
N = 60
ib = 12
iw = 20

lijst_ramping = [spatial_ramping_S(i, ib, iw, κ) for i = 1:N]

dt = 1.0
max_time_steps = 30 #3000 #7000
t_end = dt*max_time_steps

am_tilde_0 = 0.03
Delta_g = 0.0 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie
v = 0.0
v_max = 1.5


RAMPING_TIME = 5
f(t) = min(v_max, t/RAMPING_TIME)
f0(t) = 0.0

truncation = 1.5

# (Hopping_term, Mass_term, Interaction_term, Interaction_v_term, Interaction_v_term_window, Mass_term_window) = get_thirring_hamiltonian_window(1.0, 1.0, 0.0, N, lijst_ramping);

lijst = [i for i = 1:2]
my_next_nearest_neighbours(chain::InfiniteChain) = map(v -> (v, v + 1, v + 2), vertices(chain))
# Interaction_v_term_window = @mpoham sum(lijst[i]*σᶻ(){i}*σᶻ(){j} for (i,j) in nearest_neighbours(InfiniteChain(N)))

# Interaction_v_term_window = MPOHamiltonian([σᶻ() σᶻ()])

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
trivspace = U1Space(0 => 1)
Z = TensorMap([1.0+0.0im 0.0+0.0im;0.0+0.0im -1.0+0.0im;], trivspace ⊗ pspace, pspace ⊗ trivspace)
Z2 = TensorMap([1.0+0.0im 0.0+0.0im;0.0+0.0im -1.0+0.0im;], pspace, pspace)

Plus_space = U1Space(1 => 1)
Min_space = U1Space(-1 => 1)
S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], trivspace ⊗ pspace, pspace ⊗ Plus_space)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace ⊗ trivspace)
S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], Plus_space ⊗ pspace, pspace ⊗ Plus_space)
S⁺2 = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Min_space ⊗ pspace, pspace ⊗ trivspace)
S⁻2 = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], trivspace ⊗ pspace, pspace ⊗ Min_space)
S_z_symm2 = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], Min_space ⊗ pspace, pspace ⊗ Min_space)
 

@tensor operator_threesite[-1 -2 -3; -4 -5 -6 -7 -8] := S⁺[-7 -1; -4 1] * S_z_symm[1 -2; -5 2] * S⁻[2 -3; -6 -8]

MPO1 = (im*0.5) * (repeat(MPOHamiltonian([S⁺, S_z_symm, S⁻]),N));
MPO2 = -(im*0.5) * (repeat(MPOHamiltonian([S⁻2, S_z_symm2, S⁺2]),N));
MPO_inf = (im*0.5) * (repeat(MPOHamiltonian([S⁺, S_z_symm, S⁻]),2)-repeat(MPOHamiltonian([S⁻2, S_z_symm2, S⁺2]),2));
MPO = MPO1+MPO2

for i = 1:N
    for k = 1:MPO.odim-1
        MPO[i][k,MPO.odim] *= lijst_ramping[i]
    end
end

H_without_v = Hopping_term + am_tilde_0*Mass_term + Delta_g*Interaction_term;


# Interaction_v_term = @mpoham (im*v*0.5) * sum( for i in vertices(InfiniteChain(2)))

# MPO = repeat(MPOHamiltonian([Z, Z, Z]),N);

# MPO[2][1,2] = 2*MPO[2][1,2]
# MPO[3][1,2] = 3*MPO[3][1,2]
lijst = 1:3
MPOmacro = @macroexpand @mpoham sum(lijst[i]*Z2{i}*Z2{j}*Z2{k} for (i,j,k) in my_next_nearest_neighbours(InfiniteChain(3)));

# H_without_v = Hopping_term + am_tilde_0*Mass_term + Delta_g*Interaction_term + 0*Interaction_v_term;

H0 = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v)
(gs_mps, gs_envs) = get_groundstate(am_tilde_0, Delta_g, v, [50 50], truncation, truncation+3.0; number_of_loops=3)

gal = MPSKit.calc_galerkin(gs_mps, gs_envs)
var = variance(gs_mps, H0)

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
Sz = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)
HSz = @mpoham sum(Sz{i} for i in vertices(InfiniteChain(2)))

Ψ = WindowMPS(gs_mps,N; fixleft=true, fixright=true); # state is a windowMPS
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

frequency_of_saving = 1

# Ht_right = LazySum([H_without_v, MultipliedOperator(Interaction_v_term, f)])
# Ht_mid = LazySum([repeat(H_without_v,div(N,2)), MultipliedOperator(Interaction_v_term_window,f)])
# Ht_left = LazySum([H_without_v, MultipliedOperator(Interaction_v_term,f0)])
# WindowH = Window(Ht_left,Ht_mid,Ht_right);
#MPO ham voor finite deel, altijd infinite lattice. 
#window van multipliedoperator
#LazySum([Window(, )])

H1 = Window(H_without_v, repeat(H_without_v, div(N,2)), H_without_v);
# H2 = Window(0*Interaction_v_term, Interaction_v_term_window, 0*Interaction_v_term);
H2 = Window(0*MPO_inf, MPO, 0*MPO_inf);
WindowH = H1 + H2;



# HW = Window(H0, repeat(H0, div(N,2)), H0)
# HW_mass = Window(Mass_term, repeat(Mass_term, div(N,2)), Mass_term)
# WindowH = LazySum([HW, MultipliedOperator(HW_mass,f)])

# envs = environments(Ψ,WindowH);

MPSs = Vector{WindowMPS}(undef,div(number_of_timesteps,frequency_of_saving))
WindowMPSs = Vector{FiniteMPS}(undef,div(number_of_timesteps,frequency_of_saving))
# Es = Vector{Vector{ComplexF64}}(undef,div(number_of_timesteps,frequency_of_saving))
# occ_numbers = Vector{Vector{Float64}}(undef,div(number_of_timesteps,frequency_of_saving))
# Es = zeros(ComplexF64, div(number_of_timesteps,frequency_of_saving), N)
Es = []
Exp_Zs = []

occ_numbers = zeros(Float64,div(number_of_timesteps,frequency_of_saving), div(N,2)-1)

testt = [0.0]

name = "bw_hole_trivial_time_evolution_variables_N_$(N)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_savefrequency_$(frequency_of_saving)"

if isfile(name*".jld2")
    println("Warning - file already exists -- appending with new")
    name = name * "_new"
end
name = name * ".jld2"
if (isfile(name))
    println("new also exists already, aborting!")
    @assert 0 == 1
end

Es = []
left_alg = right_alg = TDVP()
middle_alg =  TDVP();
alg = WindowTDVP(;left=left_alg,middle=middle_alg,right=right_alg,
            finalize=(t, Ψ, H, envs) -> my_finalize(t, Ψ, H, envs, name));
t_span = 0:dt:t_end

if !isdir(name)
    mkdir(name)
end

# file = jldopen(name*"/0.0.jld2", "w")
# file["MPSs"] = copy(Ψ)
# file["Es"] = expectation_value(Ψ, WindowH(t), envs)
# close(file)

# (Ψ, envs) = time_evolve!(Ψ, WindowH, t_span, alg, envs; verbose=true);

Ψ = WindowMPS(gs_mps,N; fixleft=true, fixright=true); # state is a windowMPS
# env = environments(Ψ, WindowH, nothing, environments(gs_mps, H0), environments(gs_mps, H0));
env = environments(Ψ, WindowH);

(Ψ, envs) = time_evolve!(Ψ, WindowH, 0:1.0:2.0, alg, env; verbose=true);

Exp_Sz = expectation_value(Ψ, Sz)

println("done")

# @save name Es
# @save "bw_hole_time_evolution_variables_N_$(N)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_savefrequency_$(frequency_of_saving)" Es Exp_Zs

# plt = plot(1:28, real.(exp_Sz))
# display(plt)