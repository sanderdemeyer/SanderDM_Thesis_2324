using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using LaTeXStrings

include("get_occupation_number.jl")

Nₗ = 100
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

# @load "SanderDM_Thesis_2324/window_time_evolution_v_sweep_N_$(Nₗ)_mass_$(m)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(nr_steps)_vmax_$(v_max)_kappa_$(kappa)_trunc_$(truncation)_savefrequency_$(frequency_of_saving)" MPSs Es

@load "SanderDM_Thesis_2324/gs_mps_trunc_2.5_mass_$(m)_v_0.0_Delta_g_0.0" mps

dilution = 30
occupation_numbers = Vector{Vector{ComplexF64}}(undef,div(nr_steps,frequency_of_saving*dilution))

N = div(Nₗ,2)-1
X = [(2*pi)/N*i - pi for i = 0:N-1]

occ_gs = get_occupation_number(mps, Nₗ, m, 0.0; σ = 1e-3, x₀ = 15)
plt = plot(X, occ_gs, xlabel = "k")
title!("Occupation number for N = $(N)")
display(plt)
break


occ_first = get_occupation_number(MPSs[1], Nₗ, m, 0.0; σ = 0.1, x₀ = 23)
occ_last = get_occupation_number(MPSs[300], Nₗ, m, 0.0; σ = 0.1, x₀ = 23)

break

for n_t = 1:1#div(nr_steps,frequency_of_saving*dilution)
    println("Started for n_t = $(n_t)")
    # occ = get_occupation_number(MPSs[n_t*dilution], Nₗ, m, 0.0; σ = 1e-5, x₀ = 15)
    occ = get_occupation_number(mps, Nₗ, m, 0.0, x₀ = 15)
    occupation_numbers[n_t] = occ
end

@save "SanderDM_Thesis_2324/window_time_evolution_postprocessed_v_sweep_N_$(Nₗ)_mass_$(m)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(nr_steps)_vmax_$(v_max)_kappa_$(kappa)_trunc_$(truncation)_savefrequency_$(frequency_of_saving)" MPSs Es X occupation_numbers

occ_plus = [occ[i] + 0.5*sin(X[i]/2) for i = 1:N]

for n_t = 1:div(nr_steps,frequency_of_saving*dilution)
    plt = plot(X, [real(occupation_numbers[n_t][i]) for i = 1:N], xlabel = "k", ylabel = L"$\left<\hat{N}\right>$")
    title!("Occupation number for N = $(N) and t = $(dt*frequency_of_saving*dilution*n_t)")
    display(plt)
end

plt = plot(X, [real(occupation_numbers[1][i]) for i = 1:N], xlabel = "k")
title!("Occupation number for N = $(N)")
display(plt)


# plt = plot(X, occ_plus, xlabel = "k", ylabel = L"$\left<\hat{N}\right>$")
# title!("Occupation number for N = $(N)")
# display(plt)
