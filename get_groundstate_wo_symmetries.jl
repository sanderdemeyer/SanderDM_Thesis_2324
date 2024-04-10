include("get_thirring_hamiltonian.jl")

function get_groundstate_wo_symmetries(mass, Delta_g, v, iterations, trunc, tolerance; number_of_loops = 4, D_start = 3)
    mps = InfiniteMPS([ℂ^2, ℂ^2],[ℂ^D_start, ℂ^D_start])
    hamiltonian = get_thirring_hamiltonian(mass, Delta_g, v)

    for _ in 1:number_of_loops
        (mps,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = iterations[1], tol_galerkin = 10^(-tolerance)))
        (mps,envs) = changebonds(mps, hamiltonian, OptimalExpand(trscheme = truncbelow(10^(-trunc))), envs)
        #(mps,envs) = changebonds(mps, hamiltonian, VUMPSSvdCut(trscheme = truncbelow(10^(-trunc))), envs)
    end 

    (groundstate,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = iterations[2], tol_galerkin = 1e-12))
    return (groundstate, envs)
end