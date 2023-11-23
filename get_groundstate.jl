include("get_thirring_hamiltonian_symmetric.jl")
include("get_initial_mps.jl")
function get_groundstate(am_tilde_0, Delta_g, v, iterations, trunc, tolerance; number_of_loops = 4, D_start = 3, mps_start = 0)
    if D_start == 0
        mps = mps_start
    else
        mps = get_initial_mps(D_start)
    end
    hamiltonian = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v)

    for _ in 1:number_of_loops 
        (mps,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = iterations[1], tol_galerkin = 10^(-tolerance)))
        (mps,envs) = changebonds(mps, hamiltonian, VUMPSSvdCut(trscheme = truncbelow(10^(-trunc))), envs)
    end 

    (groundstate,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = iterations[2], tol_galerkin = 1e-12))
    return (groundstate, envs)
end