using LinearAlgebra
using KrylovKit
using TensorKit
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using QuadGK
using LaTeXStrings
using DelimitedFiles

include("get_occupation_number_matrices.jl")

truncation = 3.0
mass = 0.3
v = 0.0

N = 100

x₀ = 1
σ = 0.05

delta_gs = [0.0 -0.6]
N_values = [20]

X_finers = []
occs_bog = []
occs_nobog = []

for Delta_g in delta_gs
    println("started for $(Delta_g)")
    for N in N_values
        println("started for N = $(N)")

        @load "SanderDM_Thesis_2324/gs_mps/gs_mps_trunc_$(truncation)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)" mps

        (X_finer, occ_bog) = get_occupation_number_bogoliubov_right_moving(mps, N, mass, σ, x₀; bogoliubov = true)
        (X_finer, occ_nobog) = get_occupation_number_bogoliubov_right_moving(mps, N, mass, σ, x₀; bogoliubov = false)
        push!(X_finers,X_finer)
        push!(occs_bog,occ_bog)
        push!(occs_nobog,occ_nobog)
    end
end

plt = scatter(X_finers[1], occs_bog[1], label = "with Bogoliubov - free")
scatter!(X_finers[1], occs_nobog[1], label = "without Bogoliubov - free")
scatter!(X_finers[2], occs_bog[2], label = "with Bogoliubov")
scatter!(X_finers[2], occs_nobog[2], label = "without Bogoliubov")
display(plt)

writedlm("SanderDM_Thesis_2324/Bogoliubov/error measures correct/bog_nobog_g_0_-0.6.csv", vcat(occs_bog, occs_nobog))