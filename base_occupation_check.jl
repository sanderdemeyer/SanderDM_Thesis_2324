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

function avged(lijst)
    println(lijst)
    return [real(lijst[2*j+1]+lijst[2*j+2])/2 for j = 0:N-1]
end

L = 140
κ = 1.5
truncation = 2.5
D = 14
# spatial_sweep = i_end-i_start

frequency_of_saving = 3
RAMPING_TIME = 5

dt = 0.2
number_of_timesteps = 750 #3000 #7000
am_tilde_0 = 0.06
Delta_g = 0.0 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie

v_max = 5.0


σ = 0.25
x₀ = -1.0*(0.0*L-80)

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

nodp = N

number_of_savepoints = length(readdir(name))-1
times = [dt*frequency_of_saving*n_t for n_t in 0:number_of_savepoints-1]
check_occupation_numbers = true
check_entropy = true
size_entropy_region = 10

skip_frequency = 20
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
    push!(norms, norm(mps.middle))
    push!(norms_AC, [norm(mps.middle.AC[1]),norm(mps.middle.AC[1])])
    normalize!(mps.middle)
    normalize!(mps.left)
    normalize!(mps.right)
    if check_occupation_numbers
        # (occ_matrix₊, occ_matrix₋, X, X_finer) = get_occupation_matrix_bogoliubov(mps.middle, N, am_tilde_0; datapoints = nodp, bogoliubov=true)
        (occ_matrix₊, occ_matrix₋, X, X_finer) = get_occupation_matrix(mps.middle, N, am_tilde_0)
        (X_olds, occs_old) = get_occupation_number_matrices_right_moving(mps.middle, N, am_tilde_0, σ, x₀)
        push!(old_occupations, occs_old)
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

# calculate the occupation numbers for different values of σ and x₀

X = [(2*pi)/N*i - pi for i = 0:N-1]



occ_numbers = convert_to_occ(occ_matrices_R, X, σ, x₀)