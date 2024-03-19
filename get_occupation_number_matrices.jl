using LinearAlgebra
# using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit

include("get_thirring_hamiltonian_symmetric.jl")


function V_matrix(X, m)
    N = length(X)

    F = zeros(ComplexF64, 2*N, 2*N)
    for i = 0:N-1
        k = X[i+1]
        for n = 0:N-1
            F[2*n+1, 2*i+1] = 1/sqrt(N) * exp(-im*k*n)
            F[2*n+2, 2*i+2] = 1/sqrt(N) * exp(-im*k*n)
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
    
    V = F * PN

    Vp_indices = []
    Vm_indices = []
    for iₖ = 0:N-1
        k = X[iₖ+1]
        if (k < 0.0)
            push!(Vp_indices, 2*iₖ+1)
            push!(Vm_indices, 2*iₖ+2)
        else
            push!(Vp_indices, 2*iₖ+2)
            push!(Vm_indices, 2*iₖ+1)
        end
    end
    permutation = vcat(Vp_indices, Vm_indices)

    Vpermuted = zeros(ComplexF64, 2*N, 2*N)
    for i = 1:2*N
        for j = 1:2*N
            Vpermuted[i,j] = V[i,permutation[j]]
            # Vpermuted[i,permutation[j]] = V[i,j]
        end
    end
    
    V₋ = Vpermuted[:,1:N]
    V₊ = Vpermuted[:,N+1:2*N]

    return (V₊,V₋)
end

function get_occupation_number_matrices_dirac_delta(mps, L, m)
    N = div(L,2)-1
    @load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
    corr = zeros(ComplexF64, 2*N, 2*N)
    for i = 2:2*N+1
        corr_bigger = correlator(mps, S⁺, S⁻, S_z_symm, i, 2*N+2)
        corr[i-1,:] = corr_bigger[2:2*N+1]
    end
    
    X = [(2*pi)/N*i - pi for i = 0:N-1]
    N̂ = zeros(Float64, N)
    
    (V₊,V₋) = V_matrix(X, m)

    occ_matrix = V₊*corr*adjoint(V₊)

    occ = zeros(Float64, N)
    for i = 1:N
        occ[i] = real(occ_matrix[i,i])
    end
    
    return occ
end

function gaussian_array(X, k₀, σ, x₀)
    N = length(X)
    array = zeros(ComplexF64, N)
    normalization = 0.0
    for (i,k) in enumerate(X)
        # k = mod(k + pi, 2*pi) - pi
        array[i] = exp(-im*k*x₀) * exp(-((k-k₀)/(2*σ))^2)
        normalization += (exp(-((k-k₀)/(2*σ))^2))^2
    end
    return adjoint(array) / sqrt(normalization)
end

function dirac_delta_array(X, k₀, σ, x₀)
    N = length(X)
    array = zeros(ComplexF64, N)
    for (i,k) in enumerate(X)
        if (k == k₀)
            array[i] = 1.0
        end
    end
    return array
end

function get_occupation_number_matrices_base(mps, N, m, σ, x₀)
    @load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
    corr = zeros(ComplexF64, 2*N, 2*N)
    for i = 2:2*N+1
        corr_bigger = correlator(mps, S⁺, S⁻, S_z_symm, i, 2*N+2)
        corr[i-1,:] = corr_bigger[2:2*N+1]
    end
    
    X = [(2*pi)/N*i - pi for i = 0:N-1]
    N̂ = zeros(Float64, N)
    
    (V₊,V₋) = V_matrix(X, m)

    occ_matrix = V₊*corr*adjoint(V₊)

    occ = zeros(Float64, N)
    for (i,k₀) in enumerate(X)
        array = gaussian_array(X, k₀, σ, x₀)
        occupation_number = (array)*occ_matrix*adjoint(array)
        if (abs(imag(occupation_number)) > 1e-3)
            println("Warning, complex number for occupation number: $(occupation_number)")
        end
        occ[i] = real(occupation_number)
    end

    return occ
end

function get_occupation_number_matrices_right_moving(mps, N, m, σ, x₀; datapoints = N)
    @load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
    corr = zeros(ComplexF64, 2*N, 2*N)
    for i = 2:2*N+1
        corr_bigger = correlator(mps, S⁺, S⁻, S_z_symm, i, 2*N+2)
        corr[i-1,:] = corr_bigger[2:2*N+1]
    end
    
    X = [(2*pi)/N*i - pi for i = 0:N-1]
    X_finer = [(2*pi)/datapoints*i - pi for i = 0:datapoints-1]

    (V₊,V₋) = V_matrix(X, m)
    occ_matrix = adjoint(V₊)*corr*(V₊)

    occ = zeros(Float64, datapoints)
    for (i,k₀) in enumerate(X)
        array = gaussian_array(X, k₀, σ, x₀)
        occupation_number = (array)*occ_matrix*adjoint(array)
        if (abs(imag(occupation_number)) > 1e-2)
            println("Warning, complex number for occupation number: $(occupation_number)")
        end
        occ[i] = real(occupation_number)
    end

    return (X_finer, occ)
end

function get_occupation_number_matrices_left_moving(mps, N, m, σ, x₀; datapoints = N)
    @load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
    corr = zeros(ComplexF64, 2*N, 2*N)
    for i = 2:2*N+1
        corr_bigger = correlator(mps, S⁺, S⁻, S_z_symm, i, 2*N+2)
        corr[i-1,:] = corr_bigger[2:2*N+1]
    end

    X = [(2*pi)/N*i - pi for i = 0:N-1]
    X_finer = [(2*pi)/datapoints*i - pi for i = 0:datapoints-1]
    (V₊,V₋) = V_matrix(X, m)

    occ_matrix = adjoint(V₋)*corr*(V₋)

    occ = zeros(Float64, datapoints)
    for (i,k₀) in enumerate(X_finer)
        array = gaussian_array(X, k₀, σ, x₀)
        occupation_number = (array)*occ_matrix*adjoint(array)
        if (abs(imag(occupation_number)) > 1e-3)
            println("Warning, complex number for occupation number: $(occupation_number)")
        end
        occ[i] = real(occupation_number)
    end

    return (X_finer, occ)
end