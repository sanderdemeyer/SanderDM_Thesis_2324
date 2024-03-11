using LinearAlgebra
# using Base
# using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_thirring_hamiltonian_symmetric_separate.jl")
include("get_thirring_hamiltonian_window.jl")
include("get_groundstate.jl")
include("get_occupation_number_matrices.jl")

m = 0.3
Delta_g = 0.0
RAMPING_TIME = 5
dt = 0.01
v_max = 1.5
spatial_sweep = 10
truncation = 3.0
nr_steps = 1500
kappa = 0.6
frequency_of_saving = 5

symmetric = false

if symmetric
    @load "SanderDM_Thesis_2324/gs_mps_trunc_$(truncation)_mass_$(m)_v_0.0_Delta_g_$(Delta_g)" mps
else
    @load "SanderDM_Thesis_2324/gs_mps_wo_symmetries_trunc_$(truncation)_mass_$(m)_v_0.0_Delta_g_$(Delta_g)" mps
end

saving_time = 3.0

function my_finalize(t, Ψ, H, envs, MPSs, Es)
    push!(MPSs, Ψ)
    E = zeros(ComplexF64, N+4)
    for op in H(t).ops
        E += expectation_value(Ψ, op)
    end
    push!(Es, E[3:end-2])
    if (t % saving_time == 0.0)
        println("Saving")
        @save "test_window_time_evolution_test_v_sweep_N_$(N)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_savefrequency_$(frequency_of_saving)" MPSs Es
    end
    return (Ψ, envs)
end

N = 10 # Number of sites
D = 3

dt = 1.0
max_time_steps = 12 #3000 #7000
t_end = dt*max_time_steps

am_tilde_0 = 1.0
Delta_g = 0.0 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie
v = 0.0
v_max = 2.0


RAMPING_TIME = 5
f(t) = min(v_max, t/RAMPING_TIME)
f0(t) = 0.0

# (Hopping_term, Mass_term, Interaction_term, Interaction_v_term, Interaction_v_term_window, Mass_term_window) = get_thirring_hamiltonian_window(1.0, 1.0, 1.0, N, lijst_ramping)
# H_without_v = Hopping_term + am_tilde_0*Mass_term + Delta_g*Interaction_term

# (gs_mps, gs_envs) = get_groundstate(am_tilde_0, Delta_g, v, [5 10], truncation, 8.0; number_of_loops=2)

k = 0.5
X = [(2*pi)/N*i - pi for i = 0:N-1]
σ = 5
x₀ = div(N,2)

(V₊,V₋) = V_matrix(X, m)
gaussian = gaussian_array(X, k, σ, x₀)

array = adjoint(gaussian)*V₊

my_list = [i <= 10 ? 1.0 : (i == 11 ? 0.5 : 0.0) for i in 1:21]

S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^1⊗ℂ^2, ℂ^2⊗ℂ^1)
Sz = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], ℂ^2, ℂ^2)

Hs = []
for i = 1:N
    lijst_S⁺ = [j < i ? 1.0 : 0.0 for j = 1:N]
    lijst_Sz = [j == 1 ? 1.0 : 0.0 for j = 1:N]
    H = @mpoham sum((lijst_S⁺[j]*S⁺){j} for j in vertices(InfiniteChain(N)))
    push!(Hs, H)
end
Interaction_v_term = @mpoham (im*v*0.5) * sum(operator_threesite_final{i, i + 1, i + 2} for i in vertices(InfiniteChain(2)))


break

Ψ = gs_mps; # state is a windowMPS
t = 0.0
tobesaved = []
energies = []
t_span    = 0:dt:t_end
number_of_timesteps = max_time_steps
# alg       = TDVP(finalize=(t,Ψ,H,E)->myfinalize(t,Ψ,H,E,tobesaved,xWindow,g,gs,0.0))
# alg       = TDVP(expalg=Lanczos())
alg       = TDVP()
Ψt = copy(Ψ)



frequency_of_saving = 1

H1 = Window(H_without_v, repeat(H_without_v, div(N,2)), H_without_v)
H2 = Window(Interaction_v_term, Interaction_v_term_window, 0*Interaction_v_term)
WindowH = LazySum([H1, MultipliedOperator(H2, f)])


envs = environments(Ψ,WindowH);

MPSs = Vector{WindowMPS}(undef,div(number_of_timesteps,frequency_of_saving))
WindowMPSs = Vector{FiniteMPS}(undef,div(number_of_timesteps,frequency_of_saving))
Es = []
occ_numbers = zeros(Float64,div(number_of_timesteps,frequency_of_saving), div(N,2)-1)

testt = [0.0]

left_alg = right_alg = TDVP()
middle_alg =  TDVP2(; trscheme=truncdim(D));
alg = WindowTDVP(;left=left_alg,middle=middle_alg,right=right_alg,
            finalize=(t, Ψ, H, envs) -> my_finalize(t, Ψ, H, envs, MPSs, Es));
t_span = 0:dt:t_end

Ψ, envs = time_evolve!(Ψ, WindowH, t_span, alg, envs; verbose=true);

@save "test_window_time_evolution_test_v_sweep_N_$(N)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_savefrequency_$(frequency_of_saving)" MPSs Es