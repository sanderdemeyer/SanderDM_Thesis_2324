include("get_initial_mps.jl")
include("get_thirring_hamiltonian_symmetric.jl")

function get_groundstate_energy(am_tilde_0, Delta_g, v, iterations, trunc, tolerance; number_of_loops = 4, D_start = 3, mps_start = 0)
    if D_start == 0
        mps = mps_start
    else
        mps = get_initial_mps(D_start)
    end
    hamiltonian = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v)

    for _ in 1:number_of_loops
        (mps,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = iterations[1], tol_galerkin = 10^(-tolerance)))
        (mps,envs) = changebonds(mps, hamiltonian, OptimalExpand(trscheme = truncbelow(10^(-trunc))), envs)
        #(mps,envs) = changebonds(mps, hamiltonian, VUMPSSvdCut(trscheme = truncbelow(10^(-trunc))), envs)
    end 
    
    (groundstate,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = iterations[2], tol_galerkin = 1e-12))

    tot_bonddim = dims((groundstate.AL[1]).codom)[1] + dims((groundstate.AL[1]).dom)[1]

    gs_energy = expectation_value(groundstate, hamiltonian)
    gs_energy = real(gs_energy[1]+gs_energy[2])/2

    spectrum = transfer_spectrum(groundstate)
    lambdas = -log.(abs.(spectrum))
    correlation_length = 1/(lambdas[2] - lambdas[1])    
    
    return gs_energy, correlation_length, tot_bonddim, groundstate
end