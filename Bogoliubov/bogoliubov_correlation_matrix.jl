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

function V_matrix_unpermuted(X, m)
    N = length(X)

    F = zeros(ComplexF64, 2*N, 2*N)
    for i = 0:N-1
        k = X[i+1]
        for n = 0:N-1
            F[2*n+1, 2*i+1] = 1/sqrt(N) * exp(-im*k*(n+1))
            F[2*n+2, 2*i+2] = 1/sqrt(N) * exp(-im*k*(n+1))
        end
    end

    PN = zeros(ComplexF64, 2*N, 2*N)
    for i = 0:N-1
        k = X[i+1]
        A = [m -(im/2)*(1-exp(im*k)); (im/2)*(1-exp(-im*k)) -m]
        eigen_result = eigen(A)
        eigenvectors_matrix = eigen_result.vectors
        PN[2*i+1:2*i+2,2*i+1:2*i+2] = eigenvectors_matrix
    end
    return (F, PN)
end

function get_2D_matrix(index, matrix)
    return matrix[2*index-1:2*index,2*index-1:2*index] 
end

function get_bogoliubov(mps, mass, N)
    @load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
    corr = zeros(ComplexF64, 2*N, 2*N)
    for i = 2:2*N+1
        corr_bigger = correlator(mps, S⁺, S⁻, S_z_symm, i, 2*N+2)
        corr[i-1,:] = corr_bigger[2:2*N+1]
    end

    X = [(2*pi)/N*i - pi for i = 0:N-1]

    (F, PN) = V_matrix_unpermuted(X, mass)
    V = F * PN
    diag_free = adjoint(V) * corr * V

    thetas = zeros(N)
    phis = zeros(N)
    for i = 1:N
        Diag = get_2D_matrix(i, diag_free)
        (eigval, A) = eigen(Diag)
        println(A)
        (d1₊, d2₊, d1₋, d2₋) = (A[1,1], A[2,1], A[1,2], A[2,2])
        should_be_zero = d1₊ * conj(d1₋) + d2₊ * conj(d2₋)
        println("this should be 0: $(should_be_zero)")
        phase = exp(im*angle(d1₊))
        theta = acos(abs(d1₊))
        println("theta = $(theta)")
        should_be_zero2 = abs(d2₊/d1₊) - tan(theta)
        println("this scould be 0: $(should_be_zero2)")
        phi = angle(d2₊/phase)
        thetas[i] = theta
        phis[i] = phi
    end
    return (phis, thetas)
end

L = 102
m = 0.3
Delta_g = 0.0 # -0.45
v = 0.0
RAMPING_TIME = 5
dt = 0.01
v_max = 1.5
spatial_sweep = 10
truncation = 5.0
nr_steps = 1500
kappa = 0.6
frequency_of_saving = 5



@load "SanderDM_Thesis_2324/gs_mps_trunc_$(truncation)_mass_$(m)_v_0.0_Delta_g_$(Delta_g)" mps

N = div(L,2)-1


N = 200
X = [(2*pi)/N*i - pi for i = 0:N-1]
X_new = [(2*pi)/N*i for i = 0:N-1]
Es = [energy(k,m) for k in X]
# X = [2*((2*pi)/N*i - pi) for i = 0:N-1]

m_eff = m*0.825

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
unit_tensor = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], pspace, pspace)
H_unit = @mpoham sum((unit_tensor){i} for i in vertices(InfiniteChain(2)))
H = get_thirring_hamiltonian_symmetric(m, Delta_g, v)

# (X_finer_R, occ_R) = get_occupation_number_matrices_right_moving(mps, N, m, σ, x₀)


datapoints = N
X = [(2*pi)/N*i - pi for i = 0:N-1]
X_finer = [(2*pi)/datapoints*i - pi for i = 0:datapoints-1]

# (F, PN) = V_matrix_unpermuted(X, m)
# (V₊,V₋) = V_matrix(X, m)
# occ_matrix = adjoint(V₊)*corr*(V₊)

# na_F = adjoint(F) * corr * F
# na_PN = adjoint(PN) * na_F * PN

# for ind = 1:N
#     A = get_2D_matrix(ind, na_PN)
#     (eigval, eigvec) = eigen(A)
#     println("For index = $(ind), we get matrix = \n $(eigvec)")
# end


# index = 1
# k = X[index]
# lambda = sqrt(m^2 + sin(k/2)^2)

# d1₊ = exp(im*k/2)
# d2₊ = - sin(k/2)/(m-lambda)
# d1₋ = exp(im*k/2)
# d2₋ = - sin(k/2)/(m+lambda)


# (phis, thetas) = get_bogoliubov(mps, m, N)
# (phis_int, thetas_int) = get_bogoliubov(mps_int, m, N)

# plt = plot(X, thetas, label = L"\Delta(g) = 0")
# plot!(X, thetas_int, label = L"\Delta(g) = -0.45")
# xlabel!(L"momentum $k$")
# ylabel!(L"Bogoliubov angle $\theta$")
# display(plt)

Delta_gs = [0.0 -0.15 -0.3 -0.45 -0.6]

occs_200 = []
occ_betters_200 = []

σ = 0.1
x₀ = div(N,2)

for Delta_g = Delta_gs
    println("Delta_g = $(Delta_g)")
    truncation = 5.0
    m = 0.3
    σ = 0.1
    x₀ = div(N,2)
    @load "SanderDM_Thesis_2324/gs_mps_trunc_$(truncation)_mass_$(m)_v_0.0_Delta_g_$(Delta_g)" mps
    (X_finer, occ) = get_occupation_number_bogoliubov_right_moving(mps, N, m, σ, x₀; bogoliubov = false)
    (X_finer, occ_better) = get_occupation_number_bogoliubov_right_moving(mps, N, m, σ, x₀; bogoliubov = true)
    push!(occs_200, occ)
    push!(occ_betters_200, occ_better)
end

X_finer = [(2*pi)/100*i - pi for i = 0:100-1]
X_finer_200 = [(2*pi)/200*i - pi for i = 0:200-1]


plt = plot(X_finer, occs[1], label = L"$\Delta(g) = 0$")
for i = 2:length(occs)
    plt = plot!(X_finer, occs[i], label = "Delta(g) = $(Delta_gs[i])")
end
title!("N = 100, no bogoliubov")
display(plt)

plt = plot(X_finer_200, occs_200[1], label = L"$\Delta(g) = 0$")
for i = 2:length(occs)
    plt = plot!(X_finer_200, occs_200[i], label = "Delta(g) = $(Delta_gs[i])")
end
title!("N = 200, no bogoliubov")
display(plt)

plt = plot(X_finer, occ_betters[1], label = L"$\Delta(g) = 0$")
for i = 2:length(occs)
    plt = plot!(X_finer, occ_betters[i], label = "Delta(g) = $(Delta_gs[i])")
end
title!("N = 100, bogoliubov")
display(plt)

plt = plot(X_finer_200, occ_betters_200[1], label = L"$\Delta(g) = 0$")
for i = 2:length(occs)
    plt = plot!(X_finer_200, occ_betters_200[i], label = "Delta(g) = $(Delta_gs[i])")
end
title!("N = 200, bogoliubov")
display(plt)

@save "Bogoliubov_2024_04_20_N_100_and_200_different_deltags" X_finer X_finer_200 occs occs_200 occ_betters occ_betters_200

break

(X_finer, occ) = get_occupation_number_bogoliubov_right_moving(mps, N, m, σ, x₀; bogoliubov = false)
(X_finer, occ_better) = get_occupation_number_bogoliubov_right_moving(mps, N, m, σ, x₀; bogoliubov = true)
(X_finer, occ_int) = get_occupation_number_bogoliubov_right_moving(mps_int, N, m, σ, x₀; bogoliubov = false)
(X_finer, occ_int_better) = get_occupation_number_bogoliubov_right_moving(mps_int, N, m, σ, x₀; bogoliubov = true)

plt = scatter(X_finer, occ, label = "$\Delta(g) = 0$, no bogoliubov", markersize = 2)
scatter!(X_finer, occ_int, label = L"$\Delta(g) = -0.45$, no bogoliubov", markersize = 2)
scatter!(X_finer, occ_int_better, label = L"$\Delta(g) = -0.45$, bogoliubov", markersize = 2)
xlabel!(L"momentum $k$")
ylabel!(L"Occupation number $\hat{N}$")
display(plt)
