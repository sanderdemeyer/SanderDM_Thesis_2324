using LinearAlgebra
# using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit

include("get_thirring_hamiltonian_symmetric.jl")

function V_matrix_OLD(X, m)
    N = length(X)
    V₊ = zeros(ComplexF64, N, 2*N)
    V₋ = zeros(ComplexF64, N, 2*N)

    for (iₖ, k) in enumerate(X)
        if (m == 5.0)
            # v₊¹ = -1 /sqrt(2)# a
            # v₊² = exp(-im*k) /sqrt(2) # c
            # v₋¹ = -1 / sqrt(2) # b 
            # v₋² = -exp(-im*k) / sqrt(2) # d
            v₊¹ = -exp(im*k) /sqrt(2)# a
            v₊² = 1.0/sqrt(2) # c
            v₋¹ = exp(im*k) / sqrt(2) # b 
            v₋² = 1 / sqrt(2) # d
            M = [v₊¹ v₋¹; v₊² v₋²]
        else
            if (k == 0.0)
                k = 1e-10
            elseif (k == -pi)
                k = -pi + 1.0
            end
            if (k < 0.0)
                sign_lambda = -1.0
            else
                sign_lambda = 1.0
            end
            # λ = sqrt(m^2+(sin(k))^2)
            # a = -1
            # c = (λ + sign_lambda*m)/((im/2)*(1-exp(im*2*k)))
            # b = -1
            # d =  -(λ + sign_lambda*m)/((im/2)*(1-exp(im*2*k)))
            if (k > 0.0)
                λ = sqrt(m^2+(sin(k))^2)
                a = -1
                c = sign_lambda*(λ - sign_lambda*m)/((im/2)*(1-exp(im*2*k)))
                b = -1
                d = -sign_lambda*(λ + sign_lambda*m)/((im/2)*(1-exp(im*2*k)))

                a = ((im/2)*(1-exp(im*2*k)))/((λ - m)*sqrt(2))
                c = -1/sqrt(2)
                b =  ((im/2)*(1-exp(im*2*k)))/((λ + m)*sqrt(2))
                d = 1/sqrt(2)
            else
                λ = sqrt(m^2+(sin(k))^2)
                a = ((im/2)*(1-exp(im*2*k)))/((λ - m)*sqrt(2))
                c = 1/sqrt(2)
                b = -((im/2)*(1-exp(im*2*k)))/((λ + m)*sqrt(2))
                d = 1/sqrt(2)
            end

            U = [a b; c d]
            # U = (U + exp(im * angle(U[1])) * adjoint(U)) / sqrt(2)
            # a = U[1,1]
            # b = U[1,2]
            # c = U[2,1]
            # d = U[2,2]

            # if (1.0 > 0.0)
            #     v₊¹ = a #/ (abs(a)^2 + abs(c)^2)
            #     v₊² = c #/ (abs(a)^2 + abs(c)^2)
            #     v₋¹ = b #/ (abs(b)^2 + abs(d)^2)
            #     v₋² = d #/ (abs(b)^2 + abs(d)^2)
            # else
            #     v₊¹ = b / (abs(b)^2 + abs(d)^2)
            #     v₊² = d / (abs(b)^2 + abs(d)^2)
            #     v₋¹ = a / (abs(a)^2 + abs(c)^2)
            #     v₋² = c / (abs(a)^2 + abs(c)^2)
            # end
            # v₋² = (-conj(v₊¹)*v₊²/conj(v₋¹))

            # v₊¹ = -1
            # v₊² = -exp(-im*k/2)*(m-sign*λ)/sin(k/2)
            # v₋¹ = -1
            # v₋² =  -exp(-im*k/2)*(m+λ)/sin(k/2)
            # M = [v₊¹ v₋¹; v₊² v₋²]
            # println("M*M* = $(M * adjoint(M))")

            A = [m -(im/2)*(1-exp(2*im*k)); (im/2)*(1-exp(-2*im*k)) -m]
            eigen_result = eigen(A)
            eigenvectors_matrix = eigen_result.vectors
            eigenvalues_array = eigen_result.values
            P = eigenvectors_matrix
            D = Diagonal(eigenvalues_array)
            verification = inv(P) * A * P


            # Compute the eigenvectors and eigenvalues
            eigenvals, eigenvects = eigen(A)
            # Construct the unitary transformation matrix U
            # U = (eigenvects + exp(im * angle(eigenvects[1])) * adjoint(eigenvects)) / sqrt(2)
            # println("U*adjoint(U) = $(U*adjoint(U))")
            # if (k < 0.0)

            # good for k < 0.0
            if (k < 0.0)
                v₊¹ = -eigenvectors_matrix[1,2] # good ones
                v₊² = eigenvectors_matrix[2,2] # good ones
                v₋¹ = eigenvectors_matrix[1,1] # good ones
                v₋² = -eigenvectors_matrix[2,1] # good ones
                # v₊¹ = -eigenvectors_matrix[1,1]
                # v₊² = -eigenvectors_matrix[2,1]
                # v₋¹ = eigenvectors_matrix[1,2]
                # v₋² = eigenvectors_matrix[2,2]
            else
                v₊¹ = eigenvectors_matrix[1,1]
                v₊² = -eigenvectors_matrix[2,1]
                v₋¹ = -eigenvectors_matrix[1,2]
                v₋² = eigenvectors_matrix[2,2]
            end
            
            # not good for k > 0.0
            # v₊¹ = eigenvectors_matrix[1,1]
            # v₊² = eigenvectors_matrix[2,1]
            # v₋¹ = eigenvectors_matrix[1,2]
            # v₋² = eigenvectors_matrix[2,2]
            # else
            #     v₊¹ = U[1,2]
            #     v₊² = U[2,2]
            #     v₋¹ = U[1,1]
            #     v₋² = U[2,1]
            # end
            M2 = [v₊¹ v₋¹; v₊² v₋²]
            # v₊¹ = a
            # v₊² = c
            # v₋¹ = b
            # v₋² = d

        end
        for n = 0:N-1
            # V₊[iₖ,2*n+1] = exp(im*k*n)*v₊¹
            # V₊[iₖ,2*n+2] = exp(im*k*n)*v₊²
            # V₋[iₖ,2*n+1] = exp(im*k*n)*v₋¹
            # V₋[iₖ,2*n+2] = exp(im*k*n)*v₋²

            V₊[iₖ,2*n+1] = exp(-2*im*k*n)*v₊¹ / sqrt(2*N)
            V₊[iₖ,2*n+2] = exp(-2*im*k*n)*v₊² / sqrt(2*N)
            V₋[iₖ,2*n+1] = exp(-2*im*k*n)*v₋¹ / sqrt(2*N)
            V₋[iₖ,2*n+2] = exp(-2*im*k*n)*v₋² / sqrt(2*N)
        end
    end
    return (V₊,V₋)
end

function V_matrix(X, m)
    N = length(X)
    V₊ = zeros(ComplexF64, N, 2*N)
    V₋ = zeros(ComplexF64, N, 2*N)

    for (iₖ, k) in enumerate(X)
        A = [m -(im/2)*(1-exp(2*im*k)); (im/2)*(1-exp(-2*im*k)) -m]
        eigen_result = eigen(A)
        eigenvectors_matrix = eigen_result.vectors

        if (k < 0.0)
            v₊¹ = -eigenvectors_matrix[1,2]
            v₊² = eigenvectors_matrix[2,2]
            v₋¹ = eigenvectors_matrix[1,1]
            v₋² = -eigenvectors_matrix[2,1]
        else
            v₊¹ = eigenvectors_matrix[1,1]
            v₊² = -eigenvectors_matrix[2,1]
            v₋¹ = -eigenvectors_matrix[1,2]
            v₋² = eigenvectors_matrix[2,2]
        end
        # V_eigenvectors = [v₊¹ v₋¹; v₊² v₋²]

        for n = 0:N-1
            V₊[iₖ,2*n+1] = exp(-2*im*k*n)*v₊¹ / sqrt(N)
            V₊[iₖ,2*n+2] = exp(-2*im*k*n)*v₊² / sqrt(N)
            V₋[iₖ,2*n+1] = exp(-2*im*k*n)*v₋¹ / sqrt(N)
            V₋[iₖ,2*n+2] = exp(-2*im*k*n)*v₋² / sqrt(N)
        end
    end
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
    return array / sqrt(normalization)
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

function get_occupation_number_matrices_base(mps, L, m, σ, x₀)
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
    for (i,k₀) in enumerate(X)
        array = gaussian_array(X, k₀, σ, x₀)
        occupation_number = adjoint(array)*occ_matrix*array
        if (abs(imag(occupation_number)) > 1e-3)
            println("Warning, complex number for occupation number: $(occupation_number)")
        end
        occ[i] = real(occupation_number)
    end

    return occ
end

function get_occupation_number_matrices(mps, L, m, σ, x₀; datapoints = div(L,2)-1)
    N = div(L,2)-1
    @load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
    corr = zeros(ComplexF64, 2*N, 2*N)
    for i = 2:2*N+1
        corr_bigger = correlator(mps, S⁺, S⁻, S_z_symm, i, 2*N+2)
        corr[i-1,:] = corr_bigger[2:2*N+1]
    end
    
    X = [(2*pi)/N*i - pi for i = 0:N-1]
    X_finer = [(2*pi)/datapoints*i - pi for i = 0:datapoints-1]

    (V₊,_) = V_matrix(X, m)

    occ_matrix = V₊*corr*adjoint(V₊)

    occ = zeros(Float64, datapoints)
    for (i,k₀) in enumerate(X_finer)
        array = gaussian_array(X, k₀, σ, x₀)
        occupation_number = adjoint(array)*occ_matrix*array
        if (abs(imag(occupation_number)) > 1e-3)
            println("Warning, complex number for occupation number: $(occupation_number)")
        end
        occ[i] = real(occupation_number)
    end

    return (X_finer, occ)
end

# L = 70
# m = 0.0
# Delta_g = 0.0
# RAMPING_TIME = 5
# dt = 0.01
# v_max = 1.5
# spatial_sweep = 10
# truncation = 1.5
# nr_steps = 1500
# kappa = 0.6
# frequency_of_saving = 5


# @load "SanderDM_Thesis_2324/gs_mps_trunc_2.5_mass_$(m)_v_0.0_Delta_g_0.0" mps

# @load "SanderDM_Thesis_2324/window_time_evolution_v_sweep_N_$(L)_mass_$(m)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(nr_steps)_vmax_$(v_max)_kappa_$(kappa)_trunc_$(truncation)_savefrequency_$(frequency_of_saving)" MPSs Es

# N = div(L,2)-1
# X = [(2*pi)/N*i - pi for i = 0:N-1]
# # X = [2*((2*pi)/N*i - pi) for i = 0:N-1]

# σ = 2*(2*pi/L)
# x₀ = L-5

# dilution = 60
# data_points = div(nr_steps, frequency_of_saving*dilution)
# occ_ifo_time = zeros(Float64, data_points, N)

# for i = 1:data_points
#     occ_ifo_time[i,:] = get_occupation_number_matrices(MPSs[i*dilution].window, L, m, σ, x₀)
# end

# occ_data = zeros(Float64, data_points-1,N)
# for i = 2:data_points
#     occ_data[i-1,:] = occ_ifo_time[i,:] - occ_ifo_time[1,:]
# end

# plt = plot(X, occ_numbers, xlabel = "k")
# title!("Occupation number for N = $(N)")
# display(plt)


# checks
# V₊*adjoint(V₊) # should be unit
# V₋*adjoint(V₋) # should be unit
# adjoint(V₊)*V₊ + adjoint(V₋)*V₋ # should be unit
