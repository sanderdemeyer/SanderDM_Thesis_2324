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

N = 100
X = [(2*pi)/N*i - pi for i = 0:N-1]
σ = 0.1
x₀ = div(N,2)

mass = 30.0

(V₊,V₋) = V_matrix(X, mass) #

occ_matrix = zeros(ComplexF64, 2*N, 2*N)
for m = 1:2*N
    for n = 1:2*N
        occ_matrix[m,n] = sum([V₊[m,l_index] * adjoint(V₊)[l_index,n]*(l < 0.0) for (l_index, l) in enumerate(X)])
    end
end

lambdas₊ = [sqrt(mass^2+sin(k/2)^2) for k in X]
summation = sum(lambdas₊)

matrix_energy_base = zeros(ComplexF64, N, N)
for k = 1:N
    if X[k] < 0.0
        matrix_energy_base[k,k] = -summation - lambdas₊[k] # - 3*lambdas₊[k]
    else
        matrix_energy_base[k,k] = -summation + lambdas₊[k]
    end
end
occ_matrix_energy = zeros(ComplexF64, 2*N, 2*N)
for m = 1:2*N
    for n = 1:2*N
        occ_matrix_energy[m,n] = sum([adjoint(V₊)[k,m] * matrix_energy_base[k,k] * V₊[n,k] for k in 1:N])
    end
end

E0 = -summation/(2*N)
E0_extensive = -summation

occ = zeros(Float64, N)
occ_energy = zeros(Float64, N)
for (i,k₀) in enumerate(X)
    array = gaussian_array(X, k₀, σ, x₀)
    occupation_number = (array) * adjoint(V₊) * occ_matrix * V₊ * adjoint(array)
    if (abs(imag(occupation_number)) > 1e-2)
        println("Warning, complex number for occupation number: $(occupation_number)")
    end
    occ[i] = real(occupation_number)
    occupation_number_energy = (array) * adjoint(V₊) * transpose(occ_matrix_energy-E0_extensive*I) * V₊ * adjoint(array)# / (1-occupation_number)
    occ_energy[i] = real(occupation_number_energy)
end

expected = [(-1)^(k < 0.0)*sqrt(mass^2+sin(k/2)^2) for k in X]
plt = scatter(X, occ_energy)
scatter!(X, expected)
display(plt)

@save "SanderDM_Thesis_2324/correct_occupation_matrices_N_$(N)_mass_$(mass)" occ_matrix occ_matrix_energy