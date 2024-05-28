using LinearAlgebra
using KrylovKit
using TensorKit
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using QuadGK
using LaTeXStrings
using DelimitedFiles

include("get_occupation_number_matrices.jl")

function bogoliubov_check(mps, N, m, delta_g; datapoints = N, plot_k = false)
    @load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
    corr = zeros(ComplexF64, 2*N, 2*N)
    for i = 2:2*N+1
        corr_bigger = correlator(mps, S⁺, S⁻, S_z_symm, i, 2*N+2)
        corr[i-1,:] = corr_bigger[2:2*N+1]
    end
    
    X = [(2*pi)/N*i - pi for i = 0:N-1]
    X_finer = [(2*pi)/datapoints*i - pi for i = 0:datapoints-1]

    N = length(X)

    (F, PN) = V_matrix_unpermuted(X, m)
    V = F * PN
    diag_free = adjoint(V) * corr * V

    # (eigenvalues, eigenvectors) = eigen(diag_free)
    trace_total = tr(corr * (I-corr))/(N/2)
    trace_total_gamma = tr(diag_free * (I-diag_free))/(N/2)

    thetas = []
    phis = []

    trace_sum = 0.0
    traces_k = []
    for i = 1:N
        matr_k = diag_free[2*i-1:2*i,2*i-1:2*i]
        (eigval, A) = eigen(matr_k)
        a = A[1,1]
        b = A[2,1]
        push!(thetas, atan(abs(b/a)))
        push!(phis, mod(angle(b/a) + 2*pi, 2*pi))
        local_trace = real(tr(matr_k * (I - matr_k)))
        trace_sum += local_trace
        push!(traces_k, local_trace)
    end
    trace_sum /= (N/2)

    plt = scatter(X, phis)
    title!("phis")
    display(plt)

    plt = scatter(X, thetas)
    display(plt)

    println(phis)
    println(thetas)
    println(hcat(phis,thetas))
    println(vcat(phis,thetas))
    # writedlm("SanderDM_Thesis_2324/Bogoliubov/Bogliuobov_parameters_m_$(m)_N_$(N)_Delta_g_$(delta_g)", hcat(phis,thetas))


    if plot_k
        plt = scatter(X, traces_k)
        xlabel!("Momentum")
        ylabel!("Error measure")
        display(plt)
    end

    return (trace_total, trace_total_gamma, trace_sum, maximum(traces_k), thetas, phis)

    measure = sum([x*(1-x) for x in real.(eigenvalues)])/(2*N)
    # Bog_matrix = zeros(ComplexF64, 2*N, 2*N)
    # for i = 1:N
    #     Diag = get_2D_matrix(i, diag_free)
    #     (eigval, A) = eigen(Diag)
    #     Bog_matrix[2*i-1:2*i,2*i-1:2*i] = A
    # end
    norms = []
    for i = 1:N
        matr_k = diag_free[2*i-1:2*i,2*i-1:2*i]
        push!(norms, norm(matr_k))
    end
    return (eigenvalues, measure, mean(norms), diag_free)
end

truncation = 3.0
mass = 0.3
v = 0.0

N = 20

delta_gs = [0.0 -0.15 -0.3 -0.45 -0.6]
# nodp_N = 7
N_values = [25 50 100 200 400]

norms = []
measures = []
traces_tot = []
traces_tot_gamma = []
traces_sum = []
traces_max = []

for Delta_g in delta_gs
    println("started for $(Delta_g)")
    traces_deltag_tot = []
    traces_deltag_tot_gamma = []
    traces_deltag_sum = []
    traces_deltag_max = []
    for N in N_values
        println("started for N = $(N)")

        @load "SanderDM_Thesis_2324/gs_mps/gs_mps_trunc_$(truncation)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)" mps

        (trace_tot, trace_total_gamma, trace_sum, trace_max) = bogoliubov_check(mps, N, mass, Delta_g, plot_k = false)
        push!(traces_deltag_tot, real(trace_tot))
        push!(traces_deltag_tot_gamma, real(trace_total_gamma))
        push!(traces_deltag_sum, real(trace_sum))
        push!(traces_deltag_max, real(trace_max))
    end
    push!(traces_tot, traces_deltag_tot)
    push!(traces_tot_gamma, traces_deltag_tot_gamma)
    push!(traces_sum, traces_deltag_sum)
    push!(traces_max, traces_deltag_max)
end

writedlm("measure_of_eigenvalues_max.csv", traces_max)
writedlm("measure_of_eigenvalues_tot.csv", traces_tot)
writedlm("measure_of_eigenvalues_sum.csv", traces_sum)
writedlm("measure_of_eigenvalues_tot_gamma.csv", traces_tot_gamma)

print(a)

plt = scatter((N_values), (traces_tot[1]), xaxis=:log, yaxis=:log, label="\$\\Delta(g) = 0.0\$")
for (ind, g) in enumerate(delta_gs[2:end])
    scatter!((N_values), (traces_tot[ind+1]), xaxis=:log, yaxis=:log, label="\$\\Delta(g) = $(g)\$")
end
xlabel!("System size \$ N \$")
ylabel!(L"\frac{2}{N} {Tr} (\Gamma (1-\Gamma))")
display(plt)

# plt = scatter(([10*i for i = 1:nodp_N]), (traces_tot_gamma[1]), xaxis=:log, yaxis=:log, label="\$\\Delta(g) = 0.0\$")
# for (ind, g) in enumerate(delta_gs[2:end])
#     scatter!(([10*i for i = 1:nodp_N]), (traces_tot_gamma[ind+1]), xaxis=:log, yaxis=:log, label="\$\\Delta(g) = $(g)\$")
# end
# xlabel!("System size \$ N \$")
# ylabel!(L"\frac{2}{N} {Tr} (\Gamma (1-\Gamma))")
# display(plt)

plt = scatter((N_values), (traces_sum[1]), xaxis=:log, yaxis=:log, label="\$\\Delta(g) = 0.0\$")
for (ind, g) in enumerate(delta_gs[2:end])
    scatter!((N_values), (traces_sum[ind+1]), xaxis=:log, yaxis=:log, label="\$\\Delta(g) = $(g)\$")
end
xlabel!("System size \$ N \$")
ylabel!(L"\frac{2}{N} \sum_i {Tr} (\Gamma_i (1-\Gamma_i))")
display(plt)

plt = scatter((N_values), (traces_max[1]), xaxis=:log, yaxis=:log, label="\$\\Delta(g) = 0.0\$")
for (ind, g) in enumerate(delta_gs[2:end])
    scatter!((N_values), (traces_max[ind+1]), xaxis=:log, yaxis=:log, label="\$\\Delta(g) = $(g)\$")
end
xlabel!("System size \$ N \$")
ylabel!(L"2 max \left( {Tr} (\Gamma_i (1-\Gamma_i)) \right)")
display(plt)



break

using DelimitedFiles
writedlm("measure_of_eigenvalues_tot.csv", traces_tot, ',')
writedlm("measure_of_eigenvalues_tot_gamma.csv", traces_tot_gamma, ',')
writedlm("measure_of_eigenvalues_sum.csv", traces_sum, ',')

traces0 = traces[1]
traces1 = traces[2]
traces2 = traces[3]
traces3 = traces[4]
traces4 = traces[5]



@save "SanderDM_Thesis_2324//measure_of_eigenvalues.jld2" traces0 traces1 traces2 traces3 traces4

@save "deviations_from_mean_field" measures norms

break


using KrylovKit
using TensorKit
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using QuadGK
using LaTeXStrings

@load "SanderDM_Thesis_2324//measure_of_eigenvalues.jld2" traces0 traces1 traces2 traces3 traces4
