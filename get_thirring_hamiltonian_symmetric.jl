function get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v, mu)
    spin = 1//2
    pspace = U1Space(i => 1 for i in (-spin):spin)

    operator = TensorMap([0.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace)
    S_xx_S_yy = 0.5*TensorMap([0.0+0.0im 0.0+0.0im 0.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im 1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im 0.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im 0.0+0.0im 0.0+0.0im], pspace ⊗ pspace, pspace ⊗ pspace)

    Hopping_term = (1) * @mpoham sum(S_xx_S_yy{i, i + 1} for i in vertices(InfiniteChain(2)))
    # Change above from 1 to -1 !!!!!

    S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)
    J = [1.0 -1.0]
    Sz_plus_12 = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace)
    Mass_term = am_tilde_0 * @mpoham sum((J[i] * Sz_plus_12){i} for i in vertices(InfiniteChain(2)))
    
    #Sz_plus_12 = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace)
    Interaction_term = @mpoham Delta_g * sum(Sz_plus_12{i}*Sz_plus_12{i+1} for i in vertices(InfiniteChain(2)))

    return Hopping_term + Mass_term + Interaction_term


end