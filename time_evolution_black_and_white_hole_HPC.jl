include("dep_helper.jl")

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


function spatial_ramping_S(i, ib, iw, κ)
    value = 1.0 - (1/(1+exp(2*κ*(i-ib)))+1/(1+exp(2*κ*(iw-i))))
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
        file["Es"] = expectation_value(Ψ, H(t), envs)
        file["sigmaz"] = expectation_value(Ψ.middle, Sz)
        close(file)
    end
    return (Ψ, envs)
end

N = 160 # Number of sites
D = 25

@assert N % 2 == 0

κ = 0.5
ib = 40
iw = 90


dt = 0.1
number_of_timesteps = 5000 #3000 #7000
t_end = dt*number_of_timesteps
frequency_of_saving = 5
RAMPING_TIME = 5

am_tilde_0 = 0.03
Delta_g = 0.0 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie
v = 0.0
v_max = 1.5

truncation = 2.0

lijst_ramping = [spatial_ramping_S(i, ib, iw, κ) for i = 1:N]
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

# Get the groundstate MPS, together with some checks on convergence
H0 = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v)
(gs_mps, gs_envs) = get_groundstate(am_tilde_0, Delta_g, v, [50 100], truncation, truncation+3.0; number_of_loops=3)

# gal = MPSKit.calc_galerkin(gs_mps, gs_envs)
# var = variance(gs_mps, H0)

# Some stuff for postprocessing
spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
Sz = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)

# Construct initial MPS and hamiltonian
Ψ = WindowMPS(gs_mps,N; fixleft=true, fixright=true); # state is a windowMPS

H1 = Window(H_without_v, repeat(H_without_v, div(N,2)), H_without_v)
H2 = Window(0*Interaction_v_term, Interaction_v_term_window, 0*Interaction_v_term)
WindowH = LazySum([H1, MultipliedOperator(H2, f)])

envs = environments(Ψ, WindowH);

# Stuff for files and directories
name = "bw_hole_trivial_time_evolution_variables_N_$(N)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_D_$(D)_savefrequency_$(frequency_of_saving)"

if isfile(name*".jld2")
    println("Warning - file already exists -- appending with new")
    name = name * "_new"
end
name = name * ".jld2"
if (isfile(name))
    println("new also exists already, aborting!")
    @assert 0 == 1
end

if !isdir(name)
    mkdir(name)
end

# Algorithms for time evolution
left_alg = right_alg = TDVP()
middle_alg =  TDVP2(; trscheme=truncdim(D));
alg = WindowTDVP(;left=left_alg,middle=middle_alg,right=right_alg,
            finalize=(t, Ψ, H, envs) -> my_finalize(t, Ψ, H, envs, name));
t_span = 0:dt:t_end
t = 0.0

# Save for t = 0.0
file = jldopen(name*"/0.0.jld2", "w")
file["MPSs"] = copy(Ψ)
file["Es"] = expectation_value(Ψ, WindowH(t), envs)
close(file)

# Perform time evolution
(Ψ, envs) = time_evolve!(Ψ, WindowH, t_span, alg, envs; verbose=true);

println("done")


# Ψ = WindowMPS(gs_mps,N; fixleft=true, fixright=true); # state is a windowMPS
# env = environments(Ψ.middle, WindowH.middle);
# (Ψ, envs) = time_evolve!(Ψ.middle, WindowH.middle, 0:1.0:2.0, TDVP(), env; verbose=true);