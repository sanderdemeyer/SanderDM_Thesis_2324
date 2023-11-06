function get_groundstate_energy_dynamic(am_tilde_0, Delta_g, v, trunc, tolerance; number_of_loops = 4, final_iterations = 30, D_start = 3, mps_start = 0)
    if D_start == 0
        mps = mps_start
    else
        spin = 1//2
        pspace = U1Space(i => 1 for i in (-spin):spin)
        vspace_L = U1Space(1//2 => D_start, -1//2 => D_start, 3//2 => D_start, -3//2 => D_start)
        vspace_R = U1Space(2 => D_start, 1 => D_start, 0 => D_start, -1 => D_start, -2 => D_start)
        mps = InfiniteMPS([pspace, pspace], [vspace_L, vspace_R])
    end

    println("parameters are $am_tilde_0, $Delta_g, $v")
    hamiltonian = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v)

    for i in 1:number_of_loops 
        (mps,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = 10, tol_galerkin = 10^(-tolerance)))
        (mps,envs) = changebonds(mps, hamiltonian, VUMPSSvdCut(trscheme = truncbelow(10^(-trunc))), envs)
    end 
    
    (groundstate,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = final_iterations, tol_galerkin = 1e-12))

    tot_bonddim = dims((groundstate.AL[1]).codom)[1] + dims((groundstate.AL[1]).dom)[1]
    print(tot_bonddim)

    gs_energy = expectation_value(groundstate, hamiltonian)
    gs_energy = real(gs_energy[1]+gs_energy[2])/2

    spectrum = transfer_spectrum(groundstate)
    lambdas = -log.(abs.(spectrum))
    correlation_length = 1/(lambdas[2] - lambdas[1])    
    
    println("ground_state_energy is $gs_energy and correlation length is $correlation_length")
    return gs_energy, correlation_length, tot_bonddim, groundstate
end