using LinearAlgebra
# using Base
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots


N = 70
am_tilde_0 = 0.3
Delta_g = 0.0
RAMPING_TIME = 5
dt = 0.01
number_of_timesteps = 1500
v_max = 1.5
κ = 0.5
truncation = 2.0
frequency_of_saving = 1
t = 0.0

L = 36

name = "SanderDM_Thesis_2324/window_time_evolution_variables_v_sweep_N_$(N)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_savefrequency_$(frequency_of_saving)_new1/$(t).jld2"

Es = 0
file = jldopen(name, "r")
t = 15.0
# println(file["MPSs"])
Es = (file["Es"])
# mps = file["MPSs"]
close(file)
    # println(file["MPSs/$(t)"])


plt = plot(0:19, real.(Exp_Zs[1]))
display(plt)
L = 10


Es_avg = [(Exp_Zs[150][2*i+1]+Exp_Zs[150][2*i+2])/2 for i = 0:L-1]
plt = plot(0:L-1, real.(Es_avg), label="occupation number for t = $(0.0)")
display(plt)

println(length(Es))
Es_avg = [(Es[2*i+1]+Es[2*i+2])/2 for i = 0:L]

plt = plot(0:L, real.(Es_avg), label="occupation number for t = $(0.0)")

for t = [0.01 0.02 0.04 0.08]
    global plt
    name = "SanderDM_Thesis_2324/window_time_evolution_variables_v_sweep_N_$(N)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_savefrequency_$(frequency_of_saving)_new1/$(t).jld2"
    file = jldopen(name, "r")
    Es = file["Es"]
    Es_avg = [(Es[2*i+1]+Es[2*i+2])/2 for i = 0:L]
    # mps = file["MPSs"]
    println(length(Es))
    # println(file["MPSs/$(t)"])
    plot!(0:L, real.(Es_avg), label="occupation number for $(t)")
    close(file)
end

display(plt)
    