using LinearAlgebra
using KrylovKit
using TensorKit
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using QuadGK
using LaTeXStrings

include("get_occupation_number_matrices.jl")

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

    (eigval, diag_1) = eigen(diag_free)

    thetas = zeros(N)
    phis = zeros(N)
    diag_2 = zeros(2*N,2*N)
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
        diag_2[2*i-1:2*i,2*i-1:2*i] .= real.(A)
    end

    return (phis, thetas, diag_1, diag_2, diag_free)

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

N = 150

(phis, thetas, diag_1, diag_2, diag_free) = get_bogoliubov(mps, m, N)

weights = []
for i_index = 1:N
    i = 2*i_index
    sorted = sort([abs(diag_1[j,i])^2 for j = 1:2*N], rev = true)
    weight = sorted[1]+sorted[2]
    push!(weights,weight)
end


maximum([abs(diag_1[2,i])^2 for i = 1:2*N])