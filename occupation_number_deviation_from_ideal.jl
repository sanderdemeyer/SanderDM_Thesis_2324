using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using LaTeXStrings
using Random

include("get_occupation_number_matrices.jl")

L = 200
m = 0.2
Delta_g = -0.45
RAMPING_TIME = 5
dt = 0.01
v_max = 1.5
spatial_sweep = 10
truncation = 2.0
nr_steps = 1500
kappa = 0.6
frequency_of_saving = 5


@load "SanderDM_Thesis_2324/gs_mps_trunc_$(truncation)_mass_$(m)_v_0.0_Delta_g_$(Delta_g)" mps

# X = [2*((2*pi)/N*i - pi) for i = 0:N-1]

σ = 2*(2*pi/L)
σ = pi/L/4
x₀ = div(L,2)

m_eff = m*0.825

mass_renorms = LinRange(0.40, 0.65, 50)
masses = mass_renorms*m

points_below_zero = 250
nodp = 2*points_below_zero+1
ideal = [i <= points_below_zero ? 1.0 : (i == points_below_zero+1 ? 0.5 : 0.0) for i in 1:nodp]

distance_from_ideal = []
occ_numbers = []
ideal_mass = 0.0
ideal_index = -1
current_distance = 1e9
for (iₘ, mass) in enumerate(masses)
    global current_distance
    global ideal_mass
    global ideal_index
    (X_finer, occ_number_data) = get_occupation_number_matrices(mps, L, mass, σ, x₀; datapoints = nodp)
    distance = norm(occ_number_data-ideal)
    if distance < current_distance
        ideal_mass = mass
        ideal_index = iₘ
        current_distance = distance
    end
    println("for im = $(iₘ), thus mass = $(mass), distance is $(distance)")
    push!(distance_from_ideal, norm(occ_number_data-ideal))
    push!(occ_numbers, occ_number_data)
end

@save "SanderDM_Thesis_2324/ideal_renormaliztaion_mass_$(m)_Delta_g_$(Delta_g)_trunc_$(truncation)" mass_renorms distance_from_ideal occ_numbers ideal_mass ideal_index


break

# occ_number_data_eff = get_occupation_number_matrices(mps, L, m_eff, σ, x₀)


plt = plot(X_finer, occ_numbers[4][2:end], label="occupation number for m", xlabel="k", ylabel="occupation", linewidth=2, title="m = $(m), m_eff = $(round(m_eff,digits=4))")
display(plt)
plt = plot(X_finer, occ_numbers[30][2:end], label="occupation number for m", xlabel="k", ylabel="occupation", linewidth=2, title="m = $(m), m_eff = $(round(m_eff,digits=4))")
display(plt)
plt = plot(X_finer, occ_numbers[end][2:end], label="occupation number for m", xlabel="k", ylabel="occupation", linewidth=2, title="m = $(m), m_eff = $(round(m_eff,digits=4))")
display(plt)
# savefig("occupation_number_m_$(m).png")

# plt = plot(X, Es)
# display(plt)