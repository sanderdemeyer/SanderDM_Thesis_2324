using LinearAlgebra
using KrylovKit
using TensorKit
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using QuadGK
# using LsqFit
using Polynomials

include("get_occupation_number_matrices.jl")
include("get_entropy.jl")
# include("get_groundstate.jl")

function convert_to_occ(occ_matrices, X, σ, x₀)
    occ_numbers = []
    for n_t = 1:length(occ_matrices)
        occ = zeros(Float64, N)
        for (i,k₀) in enumerate(X)
            array = gaussian_array(X, k₀, σ, x₀)
            occupation_number = (array)*occ_matrices[n_t]*adjoint(array)
            if (abs(imag(occupation_number)) > 1e-2)
                println("Warning, complex number for occupation number: $(occupation_number)")
            end
            occ[i] = real(occupation_number)
        end
        push!(occ_numbers, occ)
    end
    return occ_numbers
end

function convert_to_occ_both(occ_matrices_L, occ_matrices_R, X, σ, x₀)
    occs_L = convert_to_occ(occ_matrices_L, X, σ, x₀)
    occs_R = convert_to_occ(occ_matrices_R, X, σ, x₀)
    occs = []
    for n_t = 1:length(occ_matrices_L)
        occ_L = occs_L[n_t]#/maximum(occs_L[n_t])
        occ_R = occs_R[n_t]#/maximum(occs_R[n_t])
        occ = zeros(Float64, length(X))
        for (i,k) in enumerate(X)
            if k < 0.0
                occ[i] = 1-occ_R[i]
            else
                occ[i] = occ_L[i]
            end
        end
        push!(occs, occ)
    end
    return occs
end

function avged(lijst)
    println(lijst)
    return [real(lijst[2*j+1]+lijst[2*j+2])/2 for j = 0:N-1]
end

# To investigate
# L = 200, m = 0.05, dt = 0.2, vmax = -2, kappa = 1
# L = 160, m = 0.03, dt = 0.1, vmax = 1.5, kappa = 0.5
# L = 180, m = 0.03, dt = 0.1, vmax = 1.5, kappa = 0.5
# L = 140, m = 0.03, dt = 0.1, vmax = -2.0, kappa = 1.0 # nogal moeilijk te interpreteren


L = 180
κ = 0.8
truncation = 2.5
D = 14
# spatial_sweep = i_end-i_start

frequency_of_saving = 3
RAMPING_TIME = 5

dt = 0.05
number_of_timesteps = 2500 #3000 #7000
am_tilde_0 = 0.06
Delta_g = -0.15 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie

v_max = 2.0

# σ = 0.17
# x₀ = 1.0*(1.0*L-70)


test = false
if test
    name = "bh_time_evolution/bhole_test_time_evolution_variables_N_$(L)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_D_$(D)_savefrequency_$(frequency_of_saving).jld2"
else
    name = "bh_time_evolution/bhole_time_evolution_variables_N_$(L)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_D_$(D)_savefrequency_$(frequency_of_saving).jld2"
end

Es_full = []
occ_matrices_R = []
occ_matrices_L = []
X_finers = []
norms = []
norms_AC = []
entropies = []
bipartite_entropies = []
old_occupations = []

ib = div(2*L,3)
N = div(L,2)-1

# σ = pi/L/4
# σ = 0.1784124116152771
# x₀ = 75 # div(L,2)
nodp = N

# (gs_mps, gs_envs) = get_groundstate(am_tilde_0, Delta_g, 0.0, [2 4], truncation, truncation+3.0; number_of_loops=2)

number_of_savepoints = length(readdir(name))-1
times = [dt*frequency_of_saving*n_t for n_t in 0:number_of_savepoints-1]
check_occupation_numbers = true
check_entropy = false
size_entropy_region = 10

skip_frequency = 5
reduced_times = [dt*frequency_of_saving*skip_frequency*i for i = 0:div(number_of_savepoints,skip_frequency)]
for n_t = [skip_frequency*i for i = 0:div(number_of_savepoints,skip_frequency)] #1:number_of_savepoints
    println("started for $(n_t)")
    t = dt*frequency_of_saving*n_t
    file = jldopen(name*"/$(round(t,digits=2)).jld2", "r")
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
    println("true bonddim per site is $(tot_bonddim/L)")
    push!(norms, norm(mps.middle))
    push!(norms_AC, [norm(mps.middle.AC[1]),norm(mps.middle.AC[1])])
    normalize!(mps.middle)
    normalize!(mps.left)
    normalize!(mps.right)
    if check_occupation_numbers
        (occ_matrix₊, occ_matrix₋, X, X_finer) = get_occupation_matrix_bogoliubov(mps.middle, N, am_tilde_0; datapoints = nodp, bogoliubov=true)
        # (occ_matrix₊, occ_matrix₋, X, X_finer) = get_occupation_matrix(mps.middle, N, am_tilde_0)
        # (X_olds, occs_old) = get_occupation_number_matrices_right_moving(mps.middle, N, am_tilde_0, σ, x₀; datapoints = nodp)
        # push!(old_occupations, occs_old)
        push!(occ_matrices_R, occ_matrix₊)
        push!(occ_matrices_L, occ_matrix₋)
        push!(X_finers, X_finer)
    end
    if check_entropy
        bipartite_entropy_mps = [real(entropy(mps.middle, i)) for i = 1:length(mps.middle)]
        push!(bipartite_entropies, bipartite_entropy_mps)
        push!(entropies, get_entropy(mps, ib-size_entropy_region:ib))
    end
    close(file)
end

X = [(2*pi)/N*i - pi for i = 0:N-1]
X_finer = [(2*pi)/nodp*i - pi for i = 0:nodp-1]

break
# Plot of the energies (all energies or between two sites)

# @load "everythingapril17" Es_full occ_matrices_L occ_matrices_R

iL = 1
iR = N
plt = plot(iL:iR, Es_full[1][iL:iR], legend = true)#, label = "t = $(0.0)")
for i = [i for i = 1:div(number_of_savepoints,skip_frequency)+1]
    plot!(iL:iR, Es_full[i][iL:iR])#, label = "t = $(0.5*i)")
end
xlabel!("Position")
ylabel!("Energy")
display(plt)

break
# Plot the energies between two sites in function of time, and calculate (with a plot) the speed of the waves

using LsqFit

times = [dt*frequency_of_saving*skip_frequency*i for i = 0:div(number_of_savepoints,skip_frequency)]
z1 = 44
z2 = 61
plt = plot(times, [Es_full[j][z1] for j = 1:div(number_of_savepoints,skip_frequency)+1], label  = "i=$(z1)")
for z0 = z1+1:z2
    println(z0)
    plot!(times, [Es_full[j][z0] for j = 1:div(number_of_savepoints,skip_frequency)+1], label = "i=$z0")
end
xlabel!("Time")
ylabel!("Energy")
display(plt)

max_time_considered = div(number_of_savepoints,skip_frequency)
max_time_considered = 90

first_maximum = []
for z0 = z1:z2
    for j = 2:max_time_considered
        if ((Es_full[j][z0] > Es_full[j-1][z0]) && (Es_full[j][z0] > Es_full[j+1][z0])) 
            push!(first_maximum, times[j])
            break
        end
    end
end

plt = scatter(z1:z2, first_maximum)
display(plt)

maximal_times = []
for z0 = z1:z2
    current_list = [Es_full[j][z0] for j = 1:max_time_considered]
    max_time = findfirst(x -> x == maximum(current_list), current_list)
    println(max_time)
    push!(maximal_times, times[max_time])
end

plt = scatter(maximal_times, z1:z2)
display(plt)

velocity = (z2-z1)/(maximal_times[end]-maximal_times[1])

model(x, p) = p[1] * x .+ p[2]  # Assuming a linear model here, adjust according to your actual model

# Initial guess for the parameters
p0 = [0.0, 0.0]  # Adjust according to the number of parameters in your model

# Convert maximal_times into a column vector
maximal_times_col = vec(Float64.(first_maximum))

# Fit the model to the data
fit_result = curve_fit(model, maximal_times_col, Float64.([z for z in z1:z2]), p0)

fitted_params = fit_result.param
std_errors = stderror(fit_result)

# Access the standard deviation of each parameter
std_dev_param1 = std_errors[1]
std_dev_param2 = std_errors[2]

break
# calculate the occupation numbers for different values of σ and x₀

X = [(2*pi)/N*i - pi for i = 0:N-1]

σ = 0.1784124116152771
x₀ = div(2*L,3) - 5*σ

#Good
σ = 0.18
x₀ = 50

σ = 0.2
x₀ = 0.5*(40-L)
# k₀ = 1.5

right = false


occ_numbers = convert_to_occ(occ_matrices_R, X, σ, x₀)

break
# stuff 19/04

X = [(2*pi)/N*i - pi for i = 0:N-1]

index = 30
x₀ = 40
nodp = 50
σs = LinRange(1e-3, 0.5, nodp)
k₀ = X[index]

occ_ifo_s = []
for σ = σs
    occ_numbers = convert_to_occ(occ_matrices_L, X, σ, x₀)
    occ_here = [occ_numbers[n_t][index] for n_t = 1:25]
    push!(occ_ifo_s, occ_here)
end

plt = plot(1:25, occ_ifo_s[1], label = "sigma = $(round(σs[1],digits=3))")
for i = 2:5:50
    plot!(1:25, occ_ifo_s[i], label = "sigma = $(round(σs[i],digits=3))")
end
display(plt)


break
# Plot the occupation number in function of Energy and compare with fermi_dirac


switch = true
both = true

σ = 0.178
x₀ = 31 # if L = 180
# x₀ = 24 # if L = 140


if both
    occ_numbers = convert_to_occ_both(occ_matrices_L, occ_matrices_R, X_finers[end], σ, x₀)
else
    if switch
        occ_numbers = convert_to_occ(occ_matrices_R, X, σ, x₀)
        #occ_numbers = convert_to_occ_both(occ_matrices_L, occ_matrices_R, X, σ, x₀)
    else
        occ_numbers = convert_to_occ(occ_matrices_L, X, σ, x₀)
        #occ_numbers = convert_to_occ_both(occ_matrices_L, occ_matrices_R, X, σ, x₀)
    end
end

# calculate the minimal occupation number in function of momentum
occ_numbers_minima = []
for i = 1:length(occ_numbers[1])
    push!(occ_numbers_minima, minimum([occ_numbers[j][i] for j = 1:length(occ_numbers)]))
end


kappa = abs(κ)
E_finers = [-sqrt(am_tilde_0^2+sin(k/2)^2)*(1-2*(k<0.0)) for k = X]
# E_finers = [-E_finers[index]*(1-2*(k<0.0)) for (index,k) in enumerate(X)]
fd = [1/(1+exp(-2*pi*e/(kappa))) for e in E_finers]
fd_vmax = [1/(1+exp(-2*pi*e/(kappa*abs(v_max)))) for e in E_finers]
fd08 = [1/(1+exp(-2*pi*e/(kappa*0.9))) for e in E_finers]
fd06 = [1/(1+exp(-2*pi*e/(kappa*0.8))) for e in E_finers]
fd04 = [1/(1+exp(-2*pi*e/(kappa*0.7))) for e in E_finers]

function gauss(k, k₀, σ)
    return (1/σ*sqrt(2*pi)) * exp(-(k-k₀)^2/(2*σ^2)) / (2*pi)
end
function fermi_dirac(k, mass, κ)
    E = sqrt(mass^2 + sin(k/2)^2)*(1-2*(k<0.0))
    return 1/(1+exp(-2*pi*E/κ))
end

function fermi_dirac_wo_2pi(k, mass, κ)
    E = sqrt(mass^2 + sin(k/2)^2)*(1-2*(k<0.0))
    return 1/(1+exp(-E*κ))
end

energies_not_convoluted = [fermi_dirac(k, am_tilde_0, kappa) for k in X]
energies_convoluted = [quadgk(x -> fermi_dirac(x, am_tilde_0, kappa)*gauss(k, x, σ), -pi, k, pi; atol = 1e-10)[1] for k in X]
energies_convoluted_smallest = [quadgk(x -> fermi_dirac(x, am_tilde_0, kappa/(2*pi))*gauss(k, x, σ), -pi, k, pi; atol = 1e-10)[1] for k in X]
energies_convoluted_factor = [quadgk(x -> fermi_dirac(x, am_tilde_0, kappa*(v_max/(2*pi)))*gauss(k, x, σ), -pi, k, pi; atol = 1e-10)[1] for k in X]
# energies_convoluted_expected = [quadgk(x -> fermi_dirac_wo_2pi(x, am_tilde_0, kappa*abs(v_max/2))*gauss(k, x, σ), -pi, k, pi; atol = 1e-10)[1] for k in X]
# energies_convoluted_biggest = [quadgk(x -> fermi_dirac(x, am_tilde_0, kappa*abs(v_max))*gauss(k, x, σ), -pi, k, pi; atol = 1e-10)[1] for k in X]
# energies_convoluted_08 = [quadgk(x -> fermi_dirac(x, am_tilde_0, kappa*0.8)*gauss(k, x, σ), -pi, k, pi; atol = 1e-10)[1] for k in X]
# energies_convoluted_06 = [quadgk(x -> fermi_dirac(x, am_tilde_0, kappa*0.6)*gauss(k, x, σ), -pi, k, pi; atol = 1e-10)[1] for k in X]
# energies_convoluted_04 = [quadgk(x -> fermi_dirac(x, am_tilde_0, kappa*0.4)*gauss(k, x, σ), -pi, k, pi; atol = 1e-10)[1] for k in X]


# plt = scatter(E_finers, occ_numbers[1]./maximum(occ_numbers[1]), label = "t = 0")
# scatter!(E_finers, occ_numbers[end]./maximum(occ_numbers[end]), label = "t = 75")
plt = scatter(-(1-2*switch)*E_finers, occ_numbers[1], label = "t = 0")
plt = scatter!((-1+2*switch)*E_finers, old_occupations[end], label = "old occupations - end")
plt = scatter!((-1+2*switch)*E_finers, old_occupations[15], label = "old occupations - t = $(reduced_times[15])")
plt = scatter!((-1+2*switch)*E_finers, old_occupations[div(2*length(old_occupations),3)], label = "old occupations - 2/3")
plot_frequency = 1
# for i = 2:div(number_of_savepoints,skip_frequency*plot_frequency)+1 #1:number_of_savepoints
#     scatter!(E_finers, occ_numbers[i], label = "t = $(round(i*dt*plot_frequency*skip_frequency*frequency_of_saving,digits=3))")
# end
# scatter!(E_finers, [minimum(occ_numbers[:][i]) for i = 1:length(occ_numbers[1])], label = "t = max")

# scatter!(E_finers, occ_numbers[74], label = "t = end")
# scatter!(E_finers, occ_numbers[62]./maximum(occ_numbers[62]), label = "t = 62")
# for i in [5*j for j = 2:div(length(occ_numbers),5)]
scatter!(-(1-2*switch)*E_finers, occ_numbers[end], label = "t = $(reduced_times[end])")
scatter!(-(1-2*switch)*E_finers, occ_numbers[24], label = "t = $(reduced_times[24])")
scatter!(-(1-2*switch)*E_finers, occ_numbers[98], label = "t = $(reduced_times[98])")
# scatter!((1-2*switch)*E_finers, occ_numbers[15], label = "t = $(reduced_times[15])")
scatter!(-(1-2*switch)*E_finers, occ_numbers_minima, label = "minimal values")
# scatter!((1-2*switch)*E_finers, occ_numbers[128], label = "t = $(102)")
# for i in [5*j for j = 2:div(length(occ_numbers),5)]
#     scatter!(E_finers, occ_numbers[i], label = "t = $(1.5*i)")
# end
    # scatter!(E_finers, occ_numbers[13]./maximum(occ_numbers[13]), label = "t = middle")
# plot!(E_finers, fd, label = "Expected fermi-dirac")
# plot!(E_finers, fd2, label = "Expected fermi-dirac * 1.25")
plot!(E_finers, energies_not_convoluted, label = "unconvoluted")
plot!(E_finers, energies_convoluted, label = "with convolution")
# plot!(E_finers, energies_convoluted_smallest, label = "kappa / (2pi)")
# plot!(E_finers, energies_convoluted_factor, label = "kappa v_max / (2pi)")
# plot!(E_finers, energies_convoluted_expected, label = "kappa v_max / 2")
# plot!(E_finers, energies_convoluted_biggest, label = "kappa v_max - real")


# plot!(E_finers, energies_convoluted_08, label = "Renormalized kappa 0.8")
# plot!(E_finers, energies_convoluted_06, label = "Renormalized kappa 0.6")
# plot!(E_finers, energies_convoluted_04, label = "Renormalized kappa 0.4")
plot!([am_tilde_0], seriestype="vline", line=(:black,1.0,:dash), label = "E = m")
plot!([-am_tilde_0], seriestype="vline", line=(:black,1.0,:dash), label = "E = -m")
xlabel!("E")
ylabel!("Occupation number")
# vline(-am_tilde_0)
display(plt)

break
# plot on linear graph

plt = scatter(E_finers, log.(1 ./occ_numbers[1] .- 1), label = "t = 0")
# scatter!(E_finers, log.(1 ./occ_numbers[44] .- 1), label = "t = $(reduced_times[44])")
scatter!(E_finers, log.(1 ./occ_numbers_minima .- 1), label = "minimal")
# plot!(E_finers, [(2*pi/(κ*v_max))*e for e in E_finers], label = "expected: kappa * vmax")
plot!(E_finers, [(2*pi/(κ*v_max/2))*e for e in E_finers], label = "expected: kappa * vmax / 2")
plot!(E_finers, [(2*pi/(κ))*e for e in E_finers], label = "expected: kappa")
plot!(E_finers, [(2*pi/(0.8))*e for e in E_finers], label = "expected: kappa = 0.8")
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

momentum = 10
occ_t = []
for i = 1:length(occ_numbers)
    push!(occ_t, occ_numbers[i][div(N,2)+momentum])#/maximum(occ_numbers[i]))
end
plt = scatter([times[i]*skip_frequency for i = 1:div(length(times),skip_frequency)], occ_t)
xlabel!("Time")
ylabel!("Occupation number for k = $(-round(X_finers[1][div(N,2)+momentum],digits=3))")
display(plt)

break
# fitting to FD

k1 = div(N,2)-11
k2 = div(N,2)
x_data = E_finers[k1:k2]
y_data = occ_numbers[70][k1:k2]

m(e,p) = 1 ./(exp.(2*pi.*e./(p[1])) .+ 1)

p0 = [1.0]
fit = curve_fit(m, x_data, y_data, p0)

plt = scatter(E_finers, occ_numbers[80])
scatter!(E_finers, [m(e,fit.param) for e in E_finers])
display(plt)

break
# Show on linear graph

start = 38
ending = 44

linears = log.(1 ./(occ_numbers[102]) .- 1)
plt = scatter(E_finers, linears)
display(plt)

display(scatter(E_finers[start:ending], linears[start:ending]))

inv_rico = (linears[start]-linears[ending])/(E_finers[start]-E_finers[ending])

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

break
# Plot the norm in function of time

plt = plot(times, norms[2:end])
xlabel!("t")
ylabel!("norm")
display(plt)

break
#Plot entropies

iL = 1
iR = 2*N
plt = plot(iL:iR, entropies[1][iL:iR], legend = true, label = "t = $(0.0)")
plot_skipping = 1
# for i = [plot_skipping*i for i = 1:div(number_of_savepoints,skip_frequency*plot_skipping)+1]
for i = [i for i = 1:16]
    plot!(iL:iR, entropies[i][iL:iR], label = "t = $(i*0.75)")
end
xlabel!("Position")
ylabel!("Energy")
display(plt)

break
using DelimitedFiles
using CSV

plt = scatter(reduced_times[1:end], entropies[1:end])
xlabel!("t")
ylabel!("entropy S[ib-$(size_entropy_region),ib]")
display(plt)


entropy_for_file = hcat(reduced_times, entropies)

writedlm(name*"_postprocessing_entropy_fucking_wrong", hcat(E_finers, occ_numbers_minima, energies_convoluted))

break
name_save = name*"_postprocessing"
@save name_save Es_full entropies occ_matrices_L occ_matrices_R number_of_savepoints skip_frequency X N κ v_max times

break

name_save2 = 
name_save2 = "bhole_time_evolution_variables_N_180_mass_0.06_delta_g_0.0_ramping_5_dt_0.05_nrsteps_2500_vmax_2.0_kappa_0.8_trunc_2.5_D_14_savefrequency_3.jld2_postprocessing_figures_E_finers_occ_numbers_minima_energies_convoluted_kvmaxover2.csv"
writedlm(name*"_pp_E_min_kvo2", hcat(E_finers, occ_numbers_minima, energies_convoluted))