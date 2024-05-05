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

L = 140
κ = 1.0
truncation = 1.5
D = 18
# spatial_sweep = i_end-i_start

frequency_of_saving = 5
RAMPING_TIME = 5

dt = 0.2
number_of_timesteps = 6000 #3000 #7000
am_tilde_0 = 0.03
Delta_g = 0.0 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie

v_max = -2.0


test = false
if test
    name = "bhole_trivial_time_evolution_variables_N_$(L)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_D_$(D)_savefrequency_$(frequency_of_saving).jld2"
else
    name = "bhole_time_evolution_variables_N_$(L)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_D_$(D)_savefrequency_$(frequency_of_saving).jld2"
end

Es_full = []
occ_matrices_R = []
occ_numbeces_L = []
X_finers = []

N = div(L,2)-1

σ = pi/L/4
σ = 0.1784124116152771
x₀ = 75 # div(L,2)
nodp = N

# (gs_mps, gs_envs) = get_groundstate(am_tilde_0, Delta_g, 0.0, [2 4], truncation, truncation+3.0; number_of_loops=2)

number_of_savepoints = length(readdir(name))-1
times = [dt*frequency_of_saving*n_t for n_t in 0:number_of_savepoints-1]

check_occupation_numbers = true

skip_frequency = 2
for n_t = [skip_frequency*i for i = 0:div(number_of_savepoints,skip_frequency)] #1:number_of_savepoints
    println("started for $(n_t)")
    t = dt*frequency_of_saving*n_t
    file = jldopen(name*"/$(t).jld2", "r")
    Es = file["Es"]
    Es_new = [real(Es[2*j+1]+Es[2*j+2])/2 for j = 0:N-1]
    push!(Es_full, Es_new)

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
        (occ_matrix₊, occ_matrix₋, X, X_finer) = get_occupation_matrix(mps.middle, N, am_tilde_0; datapoints = nodp)
        push!(occ_matrices_R, occ_matrix₊)
        push!(occ_matrices_L, occ_matrix₋)
        push!(X_finers, X_finer)
    end
    close(file)
end

break
# Plot of the energies (all energies or between two sites)

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
# Plot the energies between two sites in function of time, and calculate (with a plot) the speed of the waves

times = [dt*frequency_of_saving*skip_frequency*i for i = 0:div(number_of_savepoints,skip_frequency)]
z1 = 35
z2 = 45
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
# calculate the occupation numbers for different values of σ and x₀

σ = 0.1784124116152771
x₀ = div(2*L,3) - 5*σ

right = true

occ_numbers = []
for n_t = 0:div(number_of_savepoints,skip_frequency)
    occ = zeros(Float64, datapoints)
    for (i,k₀) in enumerate(X_finers[n_t])
        array = gaussian_array(X_finers[n_t], k₀, σ, x₀)
        if right
            occupation_number = (array)*occ_matrices_R[n_t]*adjoint(array)
        else
            occupation_number = (array)*occ_matrices_L[n_t]*adjoint(array)
        end
        if (abs(imag(occupation_number)) > 1e-2)
            println("Warning, complex number for occupation number: $(occupation_number)")
        end
        occ[i] = real(occupation_number)
    end
    push!(occ_numbers, occ)
end

break
# Plot the occupation number in function of Energy and compare with fermi_dirac

kappa = κ*v_max/2
E_finers = [sqrt(am_tilde_0^2+sin(k/2)^2)*(1-2*(k<0.0)) for k = X_finers[end]]
fd = [1/(1+exp(-2*pi*e/(kappa))) for e in E_finers]

# plt = scatter(E_finers, occ_numbers[1]./maximum(occ_numbers[1]), label = "t = 0")
# scatter!(E_finers, occ_numbers[end]./maximum(occ_numbers[end]), label = "t = 75")
plt = scatter(E_finers, occ_numbers[1], label = "t = 0")
plot_frequency = 1
# for i = 2:div(number_of_savepoints,skip_frequency*plot_frequency)+1 #1:number_of_savepoints
#     scatter!(E_finers, occ_numbers[i], label = "t = $(round(i*dt*plot_frequency*skip_frequency*frequency_of_saving,digits=3))")
# end
scatter!(E_finers, occ_numbers[end]./maximum(occ_numbers[end]), label = "t = end")
plot!(E_finers, fd, label = "Expected fermi-dirac")
plot!([am_tilde_0], seriestype="vline", line=(:black,1.0,:dash), label = "E = m")
plot!([-am_tilde_0], seriestype="vline", line=(:black,1.0,:dash), label = "E = -m")
xlabel!("E")
ylabel!("Occupation number")
# vline(-am_tilde_0)
display(plt)


break
# Plot the occupation numbers for all times in function of k

plt = scatter(X_finers[1], occ_numbers[1])
for j = 2:length(X_finers) #[5 10 15 20 25]
    scatter!(X_finers[j], occ_numbers[j])
end
display(plt)


break
# Plot the occupation number of a certain momentum in function of time

occ_t = []
for i = 1:length(occ_numbers)
    push!(occ_t, occ_numbers[i][div(N,2)-5]/maximum(occ_numbers[i]))
end
plt = scatter((1:length(occ_numbers))*5, occ_t)
xlabel!("Time")
ylabel!("Occupation number for k = $(-round(X_finers[1][div(N,2)-5],digits=3))")
display(plt)


break
# Calculate the first wave arrival, also useful for calculating the speed

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