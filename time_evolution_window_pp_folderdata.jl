# import Pkg
# Pkg.activate(".")
# # Pkg.rm("JLD2")
# Pkg.add("JLD2")
using LinearAlgebra
# using Base
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
# using NamedTupleTools
# using UnPack

include("get_occupation_number_matrices.jl")
# include("get_groundstate.jl")

L = 70
κ = 0.5
truncation = 1.5
# spatial_sweep = i_end-i_start

dt = 0.01
number_of_timesteps = 4500 #3000 #7000
am_tilde_0 = 0.03
Delta_g = 0.0 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie

v_max = 1.5

RAMPING_TIME = 5
frequency_of_saving = 50
name = "SanderDM_Thesis_2324/window_time_evolution_variables_v_sweep_N_$(L)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_savefrequency_$(frequency_of_saving)/"

# @unpack Es, MPSs = name*"0.0.jld2" # equivalent to x = file["x"]; y = file["y"]
# @load "SanderDM_Thesis_2324/window_time_evolution_variables_v_sweep_N_$(L)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_savefrequency_$(frequency_of_saving)/0.0.jld2" Es MPSs
# println(typeof(MPSs))
# JLD2.wconvert(::Type{ASerialization}, a::A) = ASerialization([a.x])
# JLD2.rconvert(::Type{A}, a::ASerialization) = A(only(a.x))
# JLD2.rconvert(::Type{NamedTuple}, x::WindowMPS) = B(x)
# JLD2.rconvert(::Type{WindowMPS}, nt::NamedTuple) = UpdatedStruct(Float64(nt.x), nt.y, nt.x*nt.y)
# break


Es_full = []
occ_numbers = []
X_finers = []

N = div(L,2)

σ = pi/L/4
x₀ = div(L,2)
points_below_zero = 250
nodp = 2*points_below_zero+1

# (gs_mps, gs_envs) = get_groundstate(am_tilde_0, Delta_g, 0.0, [2 4], truncation, truncation+3.0; number_of_loops=2)

number_of_savepoints = length(readdir(name))-1
times = [dt*frequency_of_saving*n_t for n_t in 0:number_of_savepoints]


for n_t = [0 3 6 9 12 15 18 21 24 27 30]
    println("started for $(n_t)")
    t = dt*frequency_of_saving*n_t
    file = jldopen(name*"$(t).jld2", "r")
    Es = file["Es"]
    Es_new = [real(Es[2*j+1]+Es[2*j+2])/2 for j = 0:N-1]
    push!(Es_full, Es_new)
    # mps = WindowMPS(gs_mps, L)
    # println(keys(file))
    # println(haskey(file,"MPSs"))
    # println(typeof(file["MPSs"]))
    mps = file["MPSs"]
    mps = getfield(mps, 1)[1]

    tot_bonddim = 0
    for i = 1:L
        tot_bonddim += dims((mps.middle.AL[i]).codom)[1] + dims((mps.middle.AL[i]).dom)[1]
    end
    println("length is $(length(mps.middle))")
    println("bonddim per site is $(tot_bonddim/N)")

    # println(typeof(mps))
    # (X_finer, occ_number_data) = get_occupation_number_matrices_left_moving(mps.middle, L, am_tilde_0, σ, x₀; datapoints = nodp)
    # push!(occ_numbers, occ_number_data)
    # push!(X_finers, X_finer)
    # close(file)
end

break

plt = plot(X_finers[1], occ_numbers[1])
for j = 2:length(X_finers)
    plot!(X_finers[j], occ_numbers[j])
end
display(plt)

k0 = 300
occs = [occ_numbers[i][k0] for i = 1:11]
plt = plot(1:11, occs)
display(plt)

break

z1 = 1
z2 = N


plt = plot(z1:z2, Es_full[1][z1:z2])
for i = 2:length(Es_full)
    plot!(z1:z2, Es_full[i][z1:z2])
end
display(plt)


z1 = 20
z2 = 25

plt = plot(times, [Es_full[j][z1] for j = 1:number_of_savepoints+1], label  = "i=$(z1)")
for z0 = z1+1:z2
    println(z0)
    plot!(times, [Es_full[j][z0] for j = 1:number_of_savepoints+1], label = "i=$z0")
end
display(plt)

i = 23

wave1 = [Es_full[j][i] for j = 1:number_of_savepoints+1]
wave2 = [Es_full[j][i+1] for j = 1:number_of_savepoints+1]
time1 = times[findfirst(x->x==maximum(wave1), wave1)]
time2 = times[findfirst(x->x==maximum(wave2), wave2)]

speed = 1/(time2-time1)
println("speed is $(speed)")