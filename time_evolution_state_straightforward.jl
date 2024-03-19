using LinearAlgebra
# using Base
# using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots

include("get_occupation_number_matrices.jl")
include("get_thirring_hamiltonian.jl")

function avged(Es)
    return real.([(Es[2*i+1]+Es[2*i+2])/2 for i = 0:div(length(Es),2)-1])
end

truncation = 1.5
mass = 0.3
Delta_g = 0.0
v = 0.0

@load "SanderDM_Thesis_2324/gs_mps_wo_symmetries_trunc_$(truncation)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)" mps

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


N = 50

k = 0.4
X = [(2*pi)/N*i - pi for i = 0:N-1]
σ = 0.7
σ = 8/(sqrt(N*pi))
x₀ = div(N,2)

(V₊,V₋) = V_matrix(X, mass)
gaussian = gaussian_array(X, k, σ, x₀)

wi = gaussian*adjoint(V₊)

# gaussian = adjoint([(i==div(N,2)) for i = 1:N])
# wi = gaussian*V₊

plt = plot(1:N, real.(adjoint(gaussian)), label = "gaussian")
display(plt)

plt = plot(1:2*N, real.(adjoint(wi)), label = "wi")
display(plt)

break


E = expectation_value(mps,hamiltonian)
println(E)
# plt = plot(1:N, avged(E), label = "before")
# display(plt)

# break

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

Ψ0 = copy(wpstate)

println("making H and envs")
wps_envs = environments(wpstate, hamiltonian)

dt = 0.1
t_end = 0.5
alg = TDVP()
t_span = 0:dt:t_end

println("plotting")
E = expectation_value(wpstate,hamiltonian)
plt = plot(1:N, avged(E), label = "before")
display(plt)



println("time_evolve")
(Ψ, envs) = time_evolve!(wpstate, hamiltonian, t_span, alg, wps_envs; verbose=true);

Eafter = expectation_value(Ψ,hamiltonian)
plt = plot(1:N, avged(Eafter), label = "after")
display(plt)

for i = 1:10
    global Ψ
    global envs
    (Ψ, envs) = time_evolve!(Ψ, hamiltonian, t_span, alg, envs; verbose=true);

    Eafter = expectation_value(Ψ,hamiltonian)
    plt = plot(1:N, avged(Eafter), label = "i = $(i)")
    display(plt)
end