using LinearAlgebra
using KrylovKit
using TensorKit
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots

include("get_occupation_number_matrices.jl")

N = 100
X = [(2*pi)/N*i - pi for i = 0:N-1]
σ = 0.5
x₀ = 1

occs = []

for (k_index,k₀) in enumerate(X)
    G = gaussian_array(X, k₀, σ, x₀)
    occ = sum([(abs(G[i])^2)*(k < 0.0) for (i,k) in enumerate(X)]) 
    println([(abs(G[i])^2)*(k < 0.0) for (i,k) in enumerate(X)])
    push!(occs, occ)
end

mass = 0.3
v = 0.0
Delta_g = 0.0
trunc = 5.0

@load "SanderDM_Thesis_2324/gs_mps_trunc_$(trunc)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)" mps
(X_finer, occ) = get_occupation_number_matrices_right_moving(mps, N, mass, σ, x₀)
    

plt = scatter(X, occs, label = "theoretical")
scatter!(X, occ, label = "mps calculation")
display(plt)
