using LinearAlgebra
# using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit

include("get_thirring_hamiltonian_symmetric.jl")
include("correlator_new.jl")

function V_matrix(X, m)
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
        PN[2*i+1:2*i+2,2*i+1:2*i+2] = eigenvectors_matrix # left = negative energy. Right = positive energy
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

function V_matrix_pos_neg_energy(X, m)
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
        if k == 0.0
            PN[2*i+1:2*i+2,2*i+1:2*i+2] = [1.0 0.0; 0.0 1.0]
        else
            A = [m -(im/2)*(1-exp(im*k)); (im/2)*(1-exp(-im*k)) -m]
            eigen_result = eigen(A)
            eigenvectors_matrix = eigen_result.vectors
            PN[2*i+1:2*i+2,2*i+1:2*i+2] = eigenvectors_matrix # left = negative energy. Right = positive energy
        end
    end
    
    V = F * PN

    Vp_indices = []
    Vm_indices = []
    for iₖ = 0:N-1
        k = X[iₖ+1]
        push!(Vp_indices, 2*iₖ+2)
        push!(Vm_indices, 2*iₖ+1)
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

function check_Vs(V₊,V₋)
    id₊ = adjoint(V₊) * V₊
    id₋ = adjoint(V₋) * V₋

    @assert norm(id₊ - I) < 1e-10
    @assert norm(id₋ - I) < 1e-10

    check2 = V₊ * adjoint(V₊) + V₋ * adjoint(V₋)
    @assert norm(check2 - I) < 1e-10
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
        k = mod(k + pi, 2*pi) - pi
        dist = min(abs(k₀ - k), 2*pi - abs(k₀ - k))
        array[i] = exp(-im*k*x₀) * exp(-((dist)/(2*σ))^2)
        normalization += (exp(-((dist)/(2*σ))^2))^2
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

function get_energy_matrices_right_moving(mps, H, N, m, σ, x₀; datapoints = N, Delta_g = 0.0, v = 0.0)
    # This function is tested in the file "get_resolution_occupation_number"
    @load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
    @load "operators_new" S⁺swap S⁻swap S_z_symm

    spin = 1//2
    pspace = U1Space(i => 1 for i in (-spin):spin)
    Plus_space = U1Space(1 => 1)
    Min_space = U1Space(-1 => 1)
    trivspace = U1Space(0 => 1)
    S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], trivspace ⊗ pspace, pspace ⊗ Plus_space)
    S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace ⊗ trivspace)
    S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)
    S⁺2 = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Min_space ⊗ pspace, pspace ⊗ trivspace)
    S⁻2 = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], trivspace ⊗ pspace, pspace ⊗ Min_space)    

    corr = zeros(ComplexF64, 2*N, 2*N)
    for i = 2:2*N+1
        corr_bigger = correlator_tf(mps, H, S⁺, S⁻, (2*im)*S_z_symm, i, 2*N+2)
        corr[i-1,:] = corr_bigger[2:2*N+1]
    end
    corr = corr + adjoint(corr)
    
    println("corr is $(corr)")
    for i = 1:2*N
        println("corr[$(i),$(i)] = $(corr[i,i])")
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

function get_occupation_matrix(mps, N, m; datapoints = N, branch = "velocity")
    @load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
    corr = zeros(ComplexF64, 2*N, 2*N)
    for i = 2:2*N+1
        corr_bigger = correlator(mps, S⁺, S⁻, S_z_symm, i, 2*N+2)
        corr[i-1,:] = corr_bigger[2:2*N+1]
    end
    
    X = [(2*pi)/N*i - pi for i = 0:N-1]
    X_finer = [(2*pi)/datapoints*i - pi for i = 0:datapoints-1]
    if branch == "velocity"
        (V₊,V₋) = V_matrix(X, m)
    elseif branch == "energy"
        (V₊,V₋) = V_matrix_pos_neg_energy(X, m)
    end

    occ_matrix₊ = adjoint(V₊)*corr*(V₊)
    occ_matrix₋ = adjoint(V₋)*corr*(V₋)
    return (occ_matrix₊, occ_matrix₋, X, X_finer)
end

function get_occupation_matrix_bogoliubov(mps, N, m; datapoints = N, bogoliubov = true)
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

    Bog_matrix = zeros(ComplexF64, 2*N, 2*N)
    for i = 1:N
        Diag = get_2D_matrix(i, diag_free)
        (eigval, A) = eigen(Diag)
        Bog_matrix[2*i-1:2*i,2*i-1:2*i] = A
    end

    if bogoliubov
        V = adjoint(Bog_matrix) * V * Bog_matrix
    end
    
    Vpermuted = permute_left_right(X, N, V)
    V₋ = Vpermuted[:,1:N]
    V₊ = Vpermuted[:,N+1:2*N]

    occ_matrix₊ = adjoint(V₊)*corr*(V₊)
    occ_matrix₋ = adjoint(V₋)*corr*(V₋)
    return (occ_matrix₊, occ_matrix₋, X, X_finer)
end

function V_matrix_bogoliubov(mps, N, m; datapoints = N, symmetric = true)
    if symmetric
        @load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
    else
        S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^1 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^1)
        S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], ℂ^1 ⊗ ℂ^2, ℂ^2 ⊗ ℂ^1)
        S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], ℂ^2, ℂ^2)
        end 
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

    Bog_matrix = zeros(ComplexF64, 2*N, 2*N)
    for i = 1:N
        Diag = get_2D_matrix(i, diag_free)
        (eigval, A) = eigen(Diag)
        Bog_matrix[2*i-1:2*i,2*i-1:2*i] = A
    end

    V = adjoint(Bog_matrix) * V * Bog_matrix
    
    Vpermuted = permute_left_right(X, N, V)
    V₋ = Vpermuted[:,1:N]
    V₊ = Vpermuted[:,N+1:2*N]
    return (V₊, V₋)
end

function get_F_matrix(X, N)
    F = zeros(ComplexF64, 2*N, 2*N)
    for i = 0:N-1
        k = X[i+1]
        for n = 0:N-1
            F[2*n+1, 2*i+1] = 1/sqrt(N) * exp(-im*k*(n+1))
            F[2*n+2, 2*i+2] = 1/sqrt(N) * exp(-im*k*(n+1))
        end
    end
    return F
end

function permute_left_right(X, N, V)
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
        end
    end
    return Vpermuted
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

function get_occupation_number_bogoliubov_right_moving(mps, N, m, σ, x₀; datapoints = N, bogoliubov = true)
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

    Bog_matrix = zeros(ComplexF64, 2*N, 2*N)
    for i = 1:N
        Diag = get_2D_matrix(i, diag_free)
        (eigval, A) = eigen(Diag)
        Bog_matrix[2*i-1:2*i,2*i-1:2*i] = A
    end

    if bogoliubov
        V = adjoint(Bog_matrix) * V * Bog_matrix
    end
    
    Vpermuted = permute_left_right(X, N, V)
    V₋ = Vpermuted[:,1:N]
    V₊ = Vpermuted[:,N+1:2*N]

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
