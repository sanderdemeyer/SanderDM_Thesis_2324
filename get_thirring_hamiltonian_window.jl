function get_thirring_hamiltonian_window(am_tilde_0, Delta_g, v, N, lijst)
    my_next_nearest_neighbours(chain::InfiniteChain) = map(v -> (v, v + 1, v + 2), vertices(chain))
 
    spin = 1//2
    pspace = U1Space(i => 1 for i in (-spin):spin)

    S_xx_S_yy = 0.5*TensorMap([0.0+0.0im 0.0+0.0im 0.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im 1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im 0.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im 0.0+0.0im 0.0+0.0im], pspace ⊗ pspace, pspace ⊗ pspace)

    J = [1.0 -1.0]
    Sz_plus_12 = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace)
        
    Plus_space = U1Space(1 => 1)
    Min_space = U1Space(-1 => 1)
    trivspace = U1Space(0 => 1)

    S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace ⊗ Plus_space)
    S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace)
    S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], Plus_space ⊗ pspace, pspace ⊗ Plus_space)
    S⁺2 = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Min_space ⊗ pspace, pspace ⊗ trivspace)
    S⁻2 = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], trivspace ⊗ pspace, pspace ⊗ Min_space)
    S_z_symm2 = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], Min_space ⊗ pspace, pspace ⊗ Min_space)


    @tensor operator_threesite[-1 -2 -3; -4 -5 -6] := S⁺[-1; -4 1] * S_z_symm[1 -2; -5 2] * S⁻[2 -3; -6]
    @tensor operator_threesite_final[-1 -2 -3; -4 -5 -6] := operator_threesite[-1 -2 -3; -4 -5 -6] - operator_threesite[-3 -2 -1; -6 -5 -4]

    Interaction_v_term = @mpoham (im*v*0.5) * sum(operator_threesite_final{i, i + 1, i + 2} for i in vertices(InfiniteChain(2)))
    Interaction_v_term_window = @mpoham (im*v*0.5) * sum(lijst[i]*operator_threesite_final{i, j, k} for (i,j,k) in my_next_nearest_neighbours(InfiniteChain(N)))
    
    # println("before terms")
    # term1 = (im*0.5*v) * (repeat(MPOHamiltonian([S⁺, S_z_symm, S⁻]),N));
    # println("after term1")
    # term2 = -(im*0.5*v) * (repeat(MPOHamiltonian([S⁻2, S_z_symm2, S⁺2]),N));
    # println("after terms")
    # Interaction_v_term = (im*0.5) * (repeat(MPOHamiltonian([S⁺, S_z_symm, S⁻]),2)-repeat(MPOHamiltonian([S⁻2, S_z_symm2, S⁺2]),2));
    # Interaction_v_term_window = term1 + term2
    # println("added")
    
    # for i = 1:N
    #     for k = 1:Interaction_v_term_window.odim-1
    #         Interaction_v_term_window[i][k,Interaction_v_term_window.odim] *= lijst_ramping[i]
    #     end
    # end
    
    
    # @tensor operator_threesite[-1 -2 -3; -4 -5 -6] := S⁺[-1; -4 1] * S_z_symm[1 -2; -5 2] * S⁻[2 -3; -6]
    # @tensor operator_threesite_final[-1 -2 -3; -4 -5 -6] := operator_threesite[-1 -2 -3; -4 -5 -6] - operator_threesite[-3 -2 -1; -6 -5 -4]

    S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)

    Hopping_term = (-1) * @mpoham sum(S_xx_S_yy{i, i + 1} for i in vertices(InfiniteChain(2)))
    Mass_term = am_tilde_0 * @mpoham sum(J[i] * (S_z_symm + 0.5*id(domain(S_z_symm))){i} for i in vertices(InfiniteChain(2)))
    
    Interaction_term = @mpoham Delta_g * sum(Sz_plus_12{i}*Sz_plus_12{i+1} for i in vertices(InfiniteChain(2)))

    # Interaction_v_term = @mpoham (im*v*0.5) * sum(S⁺{i}*S_z_symm{j}*S⁻{k} - S⁻{i}*S_z_symm{j}*S⁺{k} for i in vertices(InfiniteChain(2)))

    Mass_term_window = am_tilde_0 * @mpoham sum(lijst[i]*J[i] * (S_z_symm + 0.5*id(domain(S_z_symm))){i} for i in vertices(InfiniteChain(2)))

    return (Hopping_term, Mass_term, Interaction_term, Interaction_v_term, Interaction_v_term_window, Mass_term_window)
end