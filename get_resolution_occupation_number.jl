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

L = 70
m = 0.0
Delta_g = 0.0
RAMPING_TIME = 5
dt = 0.01
v_max = 1.5
spatial_sweep = 10
truncation = 1.5
nr_steps = 1500
kappa = 0.6
frequency_of_saving = 5


@load "SanderDM_Thesis_2324/gs_mps_trunc_2.5_mass_$(m)_v_0.0_Delta_g_0.0" mps

N = div(L,2)-1
X = [(2*pi)/N*i - pi for i = 0:N-1]
Es = [energy(k,m) for k in X]
# X = [2*((2*pi)/N*i - pi) for i = 0:N-1]

σ = 2*(2*pi/L)
x₀ = div(L,2)

occ_number_data = get_occupation_number_matrices(mps, L, m, σ, x₀)

κ = (kappa/2)*v_max
Es_fd = [fermi_dirac(ω, κ) for ω in Es]


plt = plot(Es, occ_number_data, label="occupation number", xlabel="E", ylabel="occupation", linewidth=2)
plot!(Es, Es_fd, label="fermi-dirac", xlabel="E", ylabel="occupation", linestyle=:dash, linewidth=2)
display(plt)


# plt = plot(X, Es)
# display(plt)