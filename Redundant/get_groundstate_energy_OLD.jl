function get_groundstate_energy(am_tilde_0, Delta_g, v, D, symmetric = true)
    if symmetric
        state = get_initial_mps(D)
        hamiltonian = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v)
    else
        state = get_initial_mps(D, symmetric = false)
        hamiltonian = get_thirring_hamiltonian(am_tilde_0, Delta_g, v)
    end

    (groundstate,_) = find_groundstate(state,hamiltonian,VUMPS(maxiter = 30))

    println("Done with VUMPS")
    # one = expectation_value(groundstate, identity_op)

    # xi = expectation_value(groundstate, xi_operator)


    gs_energy = expectation_value(groundstate, hamiltonian)
    gs_energy = (gs_energy[1]+gs_energy[2])/2

    println("Gs_energy is $gs_energy")

    S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)

    magnetization = expectation_value(groundstate, S_z_symm)
    magnetization = (magnetization[1]+magnetization[2])/2
    println("Magnetization is $magnetization")

    v_term_energy = expectation_value(groundstate, v_term)
    println("v_term energy is $v_term_energy")

    # S⁺ = TensorMap([0.0 1.0; 0.0 0.0], ℂ^2, ℂ^2)
    # S⁻ = TensorMap([0.0 0.0; 1.0 0.0], ℂ^2, ℂ^2)

    # max_dist = 200
    # sz_mpo = TensorMap([1.0 0; 0 -1.0], ℂ^2, ℂ^2)
    # corr_function_string1 = correlation_function(groundstate, S⁺, sz_mpo, S⁻, max_dist)
    # corr_function_string2 = correlation_function(groundstate, S⁻, sz_mpo, S⁺, max_dist)
    corr_function_string1 = 0

    return gs_energy, magnetization
end