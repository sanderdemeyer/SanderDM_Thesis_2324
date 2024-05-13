# include("dep_helper.jl")

using LinearAlgebra
# using Base
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_occupation_number_matrices.jl")
include("get_thirring_hamiltonian.jl")
include("get_groundstate_wo_symmetries.jl")

function avged(Es)
    return real.([(Es[2*i+1]+Es[2*i+2])/2 for i = 0:div(length(Es),2)-1])
end

function my_finalize(t, Ψ, H, envs, Es, name)
    Eafter = expectation_value(Ψ,H)
    push!(Es, Eafter)

    @save name Es Ψ
    return (Ψ, envs)

end

truncation = 0.5
mass = 0.3
Delta_g = -0.15
v = 0.0
bogoliubov = true

(mps, gs_envs) = get_groundstate_wo_symmetries(mass, Delta_g, v, [50 100], truncation, truncation+3.0; number_of_loops=2)

tot_bonddim = 0
for i = 1:2
    global tot_bonddim
    tot_bonddim += dims((mps.AL[i]).codom)[1] + dims((mps.AL[i]).dom)[1]
end
println("tot bonddim = $(tot_bonddim)")

S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
Sz = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], ℂ^2, ℂ^2)
middle = (-2*im)*Sz

hamiltonian = get_thirring_hamiltonian(mass, Delta_g, v)


N = 2

k = 1.0
X = [(2*pi)/N*i - pi for i = 0:N-1]
σ = 0.7
σ = 2/(sqrt(40*pi))
x₀ = 23 # div(N,2)

if bogoliubov
    (V₊,V₋) = V_matrix_bogoliubov(mps, N, mass; symmetric = false)
else
    (V₊,V₋) = V_matrix(X, mass)
end
gaussian = gaussian_array(X, k, σ, x₀)

wi = gaussian*adjoint(V₊)

println("making mps's")
mps_tensors = []
for site = 2:2*N-1 #assuming wi[0] and wi[end] are zero
    tensors = Array{TrivialTensorMap{ComplexSpace, 2, 1, Matrix{ComplexF64}}, 1}(undef,0)
    # tensors = Vector{TrivialTensorMap{ComplexSpace, 2, 1, Matrix{ComplexF64}}}
    for i = 1:site-1
        @tensor tensor_new[-1 -2; -3] := mps.AL[i][-1 2; -3] * middle[-2; 2]
        push!(tensors, tensor_new)
    end
    @tensor tensor_new[-1 -2; -3] := mps.AC[site][-1 2; -3] * S⁺[-2; 2]
    push!(tensors, tensor_new)
    for i = site+1:2*N
        push!(tensors, mps.AR[i])
    end
    mps_tensor = wi[site]*FiniteMPS(tensors)
    push!(mps_tensors, mps_tensor)
end

println("summing them")
wpstate = sum(mps_tensors)

Ψ = copy(wpstate)

println("making H and envs")
wps_envs = environments(wpstate, hamiltonian)
envs = wps_envs

E = expectation_value(wpstate,hamiltonian)
Es = []
push!(Es, E)

dt = 0.8
t_end = 0.8

name = "SanderDM_Thesis_2324/test_wavepacket_right_moving_gs_mps_wo_symmetries_trunc_$(truncation)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)_N_$(N)_k_$(k)_sigma_$(round(σ,digits=3))_dt_$(dt)_tend_$(t_end)"

alg = TDVP(; finalize=(t, Ψ, H, envs) -> my_finalize(t, Ψ, H, envs, Es, name))
t_span = 0:dt:t_end


println("time_evolve")


for i = 1:16
    global Ψ
    global envs
    (Ψ, envs) = time_evolve!(Ψ, hamiltonian, t_span, alg, envs; verbose=true);

    # plt = plot(1:N, avged(Eafter), label = "i = $(i)")
    # display(plt)

end

# @save "SanderDM_Thesis_2324/test_wavepacket_right_moving_gs_mps_wo_symmetries_trunc_$(truncation)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)_N_$(N)_k_$(k)_sigma_$(round(σ,digits=3))_dt_$(dt)_tend_$(t_end)" Es
