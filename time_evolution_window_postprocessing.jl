using LinearAlgebra
# using Base
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots

# include(raw"C:\Users\Sande\Documents\School\0600 - tweesis\Code\MPSKit.jl\src\MPSKit.jl")
# include(raw"C:\Users\Sande\Documents\School\0600 - tweesis\Code\MPSKitModels.jl\src\MPSKitModels.jl")

include("get_thirring_hamiltonian_symmetric_separate.jl")
include("get_thirring_hamiltonian_window.jl")
include("get_groundstate.jl")

N = 70
mass = 0.3
delta_g = 0.0
ramping = 5
dt = 0.01
nrsteps = 1500
vmax = 1.5
kappa = 0.5
trunc = 7.0
savefrequency = 50

@load "SanderDM_Thesis_2324/window_time_evolution_v_sweep_N_$(N)_mass_$(mass)_delta_g_$(delta_g)_ramping_$(ramping)_dt_$(dt)_nrsteps_$(nrsteps)_vmax_$(vmax)_kappa_$(kappa)_trunc_$(trunc)_savefrequency_$(savefrequency)" Es
# @load "SanderDM_Thesis_2324/window_time_evolution_v_sweep_N_$(N)_mass_$(mass)_delta_g_$(delta_g)_ramping_$(ramping)_dt_$(dt)_nrsteps_$(nrsteps)_vmax_$(vmax)_kappa_$(kappa)_trunc_$(trunc)_savefrequency_$(savefrequency)" MPSs



Es = real.(Es)

Es_av = [(Es[1][2*j+1]+Es[1][2*j+2])/2 for j = 0:29]
plt = plot(1:30, Es_av, label = "n_t = $(1)")
for i = 2:length(Es)
    Es_av = [(Es[i][2*j+1]+Es[i][2*j+2])/2 for j = 0:29]
    plot!(1:30, Es_av, label = "n_t = $(i)")
end
display(plt)
savefig("boundary-effects.png")

break
tot_bonddim = 0
for i = 1:N
    global tot_bonddim
    tot_bonddim += dims((MPSs[1].window.AL[i]).codom)[1] + dims((MPSs[1].window.AL[i]).dom)[1]
end


println("Tot bonddim is $tot_bonddim")
println("bonddim per site is $(tot_bonddim/N)")

println(typeof(MPSs[1]))

N = 2
tot_bonddim = 0
for i = 1:N
    global tot_bonddim
    tot_bonddim += dims((gs_mps.AL[i]).codom)[1] + dims((gs_mps.AL[i]).dom)[1]
end
