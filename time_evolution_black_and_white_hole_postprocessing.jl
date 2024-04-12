using LinearAlgebra
using KrylovKit
using TensorKit
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots

include("get_occupation_number_matrices.jl")
# include("get_groundstate.jl")

function avged(lijst)
    println(lijst)
    return [real(lijst[2*j+1]+lijst[2*j+2])/2 for j = 0:N-1]
end

L = 160
κ = 0.5
truncation = 2.0
D = 25
# spatial_sweep = i_end-i_start

frequency_of_saving = 5
RAMPING_TIME = 5

dt = 0.1
number_of_timesteps = 5000 #3000 #7000
am_tilde_0 = 0.03
Delta_g = 0.0 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie

v_max = 1.5


test = true
if test
    # name = "bw_hole_trivial_time_evolution_variables_N_$(L)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_savefrequency_$(frequency_of_saving).jld2"
    name = "bw_hole_trivial_time_evolution_variables_N_$(L)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_D_$(D)_savefrequency_$(frequency_of_saving).jld2"
else
    # name = "bw_hole_time_evolution_variables_N_$(L)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_savefrequency_$(frequency_of_saving).jld2"
    name = "bw_hole_time_evolution_variables_N_$(L)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_D_$(D)_savefrequency_$(frequency_of_saving).jld2"
end

Es_full = []
occ_numbers = []
X_finers = []
Z_values = []

N = div(L,2)-1

σ = pi/L/4
σ = 0.1784124116152771
x₀ = 125 # div(L,2)
points_below_zero = 250
nodp = N

# (gs_mps, gs_envs) = get_groundstate(am_tilde_0, Delta_g, 0.0, [2 4], truncation, truncation+3.0; number_of_loops=2)

number_of_savepoints = length(readdir(name))-1
times = [dt*frequency_of_saving*n_t for n_t in 0:number_of_savepoints-1]

check_occupation_numbers = true

skip_frequency = 5
for n_t = [skip_frequency*i for i = 0:div(number_of_savepoints,skip_frequency)] #1:number_of_savepoints
    println("started for $(n_t)")
    t = dt*frequency_of_saving*n_t
    file = jldopen(name*"/$(t).jld2", "r")
    Es = file["Es"]
    Es_new = [real(Es[2*j+1]+Es[2*j+2])/2 for j = 0:N-1]
    push!(Es_full, Es_new)

    # push!(Z_values, file["sigmaz"])

    mps = file["MPSs"]
    mps = getfield(mps, 1)[1]

    tot_bonddim = 0
    for i = 1:L
        tot_bonddim += dims((mps.middle.AL[i]).codom)[1] + dims((mps.middle.AL[i]).dom)[1]
    end
    println("length is $(length(mps.middle))")
    println("bonddim per site is $(tot_bonddim/N)")

    # println(typeof(mps))
    if check_occupation_numbers
        (X_finer, occ_number_data) = get_occupation_number_matrices_left_moving(mps.middle, N, am_tilde_0, σ, x₀; datapoints = nodp)
        push!(occ_numbers, occ_number_data)
        push!(X_finers, X_finer)
    end
    close(file)
end

break

# Plot of the energies

iL = 1
iR = N
plt = plot(iL:iR, Es_full[1][iL:iR], legend = true)#, label = "t = $(0.0)")
for i = [i for i = 1:number_of_savepoints]
    plot!(iL:iR, Es_full[i][iL:iR])#, label = "t = $(0.5*i)")
end
xlabel!("Position")
ylabel!("Energy")
display(plt)

break

# Animation of the energies

# Create a function to plot each frame
function plot_frame(data, i)
    plot!(iL:iR, data[i][iL:iR])#, label = "t = $(0.5*i)")
    # heatmap(data, c=:blues, clim=(0, 1), title="Frame $frame")
end

# Create frames and save them
frames = []
for i in 1:size(Es_full, 1)
    push!(frames, plot_frame(Es_full, i))
end

# Create GIF
anim = @animate for frame in frames
    frame
end

gif(anim, "animation.gif", fps = 2)



break

iL = 1
iR = N
plt = plot(iL:iR, avged(Z_values[1]), legend = true)#, label = "t = $(0.0)")
for i = [i for i = 1:number_of_savepoints]
    plot!(iL:iR, avged(Z_values[i]))#, label = "t = $(0.5*i)")
end
xlabel!("Position")
ylabel!("< sigma_z >")
display(plt)


times = [dt*frequency_of_saving*skip_frequency*i for i = 0:div(number_of_savepoints,skip_frequency)]
z1 = 50
z2 = 63
plt = plot(times, [Es_full[j][z1] for j = 1:div(number_of_savepoints,skip_frequency)+1], label  = "i=$(z1)")
for z0 = z1+1:z2
    println(z0)
    plot!(times, [Es_full[j][z0] for j = 1:div(number_of_savepoints,skip_frequency)+1], label = "i=$z0")
end
xlabel!("Time")
ylabel!("Energy")
display(plt)

maximal_times = []
for z0 = z1:z2
    current_list = [Es_full[j][z0] for j = 1:number_of_savepoints]
    max_time = findfirst(x -> x == maximum(current_list), current_list)
    push!(maximal_times, times[max_time])
end

plt = scatter(maximal_times, z1:z2)
display(plt)

velocity = (z2-z1)/(maximal_times[end]-maximal_times[1])

break

# Plot the occupation number in function of Energy and compare with fermi_dirac

kappa = κ*v_max/2
E_finers = [sqrt(am_tilde_0^2+sin(k/2)^2)*(1-2*(k<0.0)) for k = X_finers[end]]
fd = [1/(1+exp(-2*pi*e/(kappa))) for e in E_finers]

# plt = scatter(E_finers, occ_numbers[1]./maximum(occ_numbers[1]), label = "t = 0")
# scatter!(E_finers, occ_numbers[end]./maximum(occ_numbers[end]), label = "t = 75")
plt = scatter(E_finers, occ_numbers[1], label = "t = 0")
plot_frequency = 5
for i = [skip_frequency*frequency_of_saving*i for i = 2:div(number_of_savepoints,skip_frequency*skip_frequency)+1] #1:number_of_savepoints
    scatter!(E_finers, occ_numbers[i], label = "t = $(dt*plot_frequency)")
end
plot!(E_finers, fd, label = "Expected fermi-dirac")
plot!([am_tilde_0], seriestype="vline", line=(:black,1.0,:dash), label = "E = m")
plot!([-am_tilde_0], seriestype="vline", line=(:black,1.0,:dash), label = "E = -m")
xlabel!("E")
ylabel!("Occupation number")
# vline(-am_tilde_0)
display(plt)


break

plt = scatter(X_finers[1], occ_numbers[1])
for j = 2:length(X_finers) #[5 10 15 20 25]
    scatter!(X_finers[j], occ_numbers[j])
end
display(plt)

occ_t = []
for i = 1:length(occ_numbers)
    push!(occ_t, occ_numbers[i][div(N,2)-5])
end
plt = scatter((1:length(occ_numbers))*5, occ_t)
xlabel!("Time")
ylabel!("Occupation number for k = $(-round(X_finers[1][div(N,2)-5],digits=3))")
display(plt)

E_finers = zeros(Float64, 34)
for (i,k) in enumerate(X_finers[1])
    lambda = sqrt(am_tilde_0^2 + sin(k/2)^2)
    if k < 0.0
        E_finers[i] = lambda
    else
        E_finers[i] = -lambda
    end
end

plt = scatter(X_finers, E_finers)
display(plt)


plt = scatter(E_finers, occ_numbers[end])
display(plt)

last_occ_numbers = occ_numbers[end]
last_E_finer = E_finers[end]

@save "SanderDM_Thesis_2324/test_fermi_dirac" E_finers last_occ_numbers

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


z1 = 21
z2 = 25

plt = plot(times, [Es_full[j][z1] for j = 1:number_of_savepoints], label  = "i=$(z1)")
for z0 = z1+1:z2
    println(z0)
    plot!(times, [Es_full[j][z0] for j = 1:number_of_savepoints], label = "i=$z0")
end
display(plt)

first_wave_arrival = []
for z = 21:25
    E = [Es_full[j][z] for j = 1:60]
    index_max = findfirst(x->x==maximum(E), E)
    println(index_max)
    println(times[index_max])
    push!(first_wave_arrival, times[index_max])
end

plt = scatter(21:25, first_wave_arrival)
sites = LinRange(21, 25, 1000)
plot!(sites, [(26.4-i)*4 for i = sites])
display(plt)

i = 23

wave1 = [Es_full[j][i] for j = 1:number_of_savepoints+1]
wave2 = [Es_full[j][i+1] for j = 1:number_of_savepoints+1]
time1 = times[findfirst(x->x==maximum(wave1), wave1)]
time2 = times[findfirst(x->x==maximum(wave2), wave2)]

speed = 1/(time2-time1)
println("speed is $(speed)")


@save "SanderDM_Thesis_2324/window_time_evolution_variables_v_sweep_N_$(L)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_savefrequency_$(frequency_of_saving)/postprocessing" Es_full occ_numbers X_finers
