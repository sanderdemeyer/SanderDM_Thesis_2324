using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using LaTeXStrings
using Random

include("SanderDM_Thesis_2324/get_occupation_number_matrices.jl")

function spatial_ramping_S(i, i_middle, κ)
    value = 1 - 1/(1+exp(2*κ*(i-i_middle)))
    if value < 1e-4
        return 0.0
    elseif value > 1 - 1e-4
        return 1.0
    end
    return value
end

function fermi_dirac(ω, κ)
    return 1/(1+exp(ω/κ))
end

function energy(k, m)
    if (k < 0.0)
        return -sqrt(m^2+sin(k/2)^2)
    else
        return sqrt(m^2+sin(k/2)^2)
    end
end

L = 200
m = 0.3
Delta_g = 0.0
RAMPING_TIME = 5
dt = 0.01
v_max = 1.5
spatial_sweep = 10
truncation = 2.5
nr_steps = 1500
kappa = 0.6
frequency_of_saving = 5


@load "SanderDM_Thesis_2324/gs_mps_trunc_$(truncation)_mass_$(m)_v_0.0_Delta_g_$(Delta_g)" mps

N = 10
X = [(2*pi)/N*i - pi for i = 0:N-1]
X_new = [(2*pi)/N*i for i = 0:N-1]
Es = [energy(k,m) for k in X]
# X = [2*((2*pi)/N*i - pi) for i = 0:N-1]

σ = pi/L/4
x₀ = div(L,2)

m_eff = m*0.825

(X_finer_L, occ_L) = get_occupation_number_matrices_left_moving(mps, N, m, σ, x₀)
(X_finer_R, occ_R) = get_occupation_number_matrices_right_moving(mps, N, m, σ, x₀)
# occ_number_data_eff = get_occupation_number_matrices(mps, L, m_eff, σ, x₀; V = "old")

# for i = div(N,2)-5:div(N,2)+5
#     println("k = $(X[i])")
#     println("$(occ_number_data[i]) for m")
#     println("$(occ_number_data_eff[i]) for m_eff")
# end


κ = (kappa/2)*v_max
Es_fd = [fermi_dirac(ω, κ) for ω in Es]

plt = scatter(X_finer_L, occ_L, label="occupation number for m", xlabel="k", ylabel="occupation", linewidth=2, title="m = $(m), m_eff = $(round(m_eff,digits=4))")
display(plt)
plt = scatter(X_finer_R, occ_R, label="occupation number for m", xlabel="k", ylabel="occupation", linewidth=2, title="m = $(m), m_eff = $(round(m_eff,digits=4))")
display(plt)

break

# plt = plot(X, occ_number_data, label="occupation number for m", xlabel="k", ylabel="occupation", linewidth=2, title="m = $(m), m_eff = $(round(m_eff,digits=4))")
plt = plot(X_finer, occ, label="occupation number for m", xlabel="k", ylabel="occupation", linewidth=2, title="m = $(m), m_eff = $(round(m_eff,digits=4))")
# plot!(X, occ_number_data_eff, label="occupation number for m_eff", xlabel="k", ylabel="occupation", linewidth=2, title="m = $(m), m_eff = $(round(m_eff,digits=4))")
# plot!(Es, Es_fd, label="fermi-dirac", xlabel="E", ylabel="occupation", linestyle=:dash, linewidth=2)

display(plt)
# savefig("occupation_number_m_$(m).png")

# plt = plot(X, Es)
# display(plt)