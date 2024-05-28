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
    