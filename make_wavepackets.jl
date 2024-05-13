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

truncation = 1.5
mass = 0.3
Delta_g = 0.0
v = 0.0


(mps, gs_envs) = get_groundstate_wo_symmetries(mass, Delta_g, v, [50 100], truncation, truncation+3.0; number_of_loops=7)

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


N = 16

k = -1.5
X = [(2*pi)/N*i - pi for i = 0:N-1]
σ = 0.7
σ = 2/(sqrt(N*pi))
x₀ = div(N,2)

Energy_theoretical = sqrt(mass^2 + sin(k)^2)

(V₊,V₋) = V_matrix(X, mass)
gaussian = gaussian_array(X, k, σ, x₀)

wi = gaussian*adjoint(V₊)

# plt = plot(1:N, real.(adjoint(gaussian)), label = "gaussian")
# display(plt)

# plt = plot(1:2*N, real.(adjoint(wi)), label = "wi")
# display(plt)


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

dt = 0.8
t_end = 4.0
alg = TDVP()
t_span = 0:dt:t_end

# println("plotting")
E = expectation_value(wpstate,hamiltonian)
# plt = plot(1:N, avged(E), label = "before")
# display(plt)