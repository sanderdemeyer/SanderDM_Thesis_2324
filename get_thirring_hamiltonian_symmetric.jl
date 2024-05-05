function get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v; new = false)
    spin = 1//2
    pspace = U1Space(i => 1 for i in (-spin):spin)

    S_xx_S_yy = 0.5*TensorMap([0.0+0.0im 0.0+0.0im 0.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im 1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im 0.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im 0.0+0.0im 0.0+0.0im], pspace ⊗ pspace, pspace ⊗ pspace)

    J = [1.0 -1.0]
    Sz_plus_12 = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace)
        
    Plus_space = U1Space(1 => 1)
    S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace ⊗ Plus_space)
    S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace)
    S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)

    Hopping_term = (-1) * @mpoham sum(S_xx_S_yy{i, i + 1} for i in vertices(InfiniteChain(2)))
    Mass_term = am_tilde_0 * @mpoham sum(J[i] * (S_z_symm + 0.5*id(domain(S_z_symm))){i} for i in vertices(InfiniteChain(2)))
    
    Interaction_term = @mpoham Delta_g * sum(Sz_plus_12{i}*Sz_plus_12{i+1} for i in vertices(InfiniteChain(2)))

    if new
        Min_space = U1Space(-1 => 1)
        trivspace = U1Space(0 => 1)
        S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], trivspace ⊗ pspace, pspace ⊗ Plus_space)
        S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace ⊗ trivspace)
        S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], Plus_space ⊗ pspace, pspace ⊗ Plus_space)
        S⁺2 = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Min_space ⊗ pspace, pspace ⊗ trivspace)
        S⁻2 = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], trivspace ⊗ pspace, pspace ⊗ Min_space)
        S_z_symm2 = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], Min_space ⊗ pspace, pspace ⊗ Min_space)
        Interaction_v_term = (im*v*0.5) * (repeat(MPOHamiltonian([S⁺, S_z_symm, S⁻]),2)-repeat(MPOHamiltonian([S⁻2, S_z_symm2, S⁺2]),2));
    else
        S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], Plus_space ⊗ pspace, pspace ⊗ Plus_space)
        @tensor operator_threesite[-1 -2 -3; -4 -5 -6] := S⁺[-1; -4 1] * S_z_symm[1 -2; -5 2] * S⁻[2 -3; -6]
        @tensor operator_threesite_final[-1 -2 -3; -4 -5 -6] := operator_threesite[-1 -2 -3; -4 -5 -6] - operator_threesite[-3 -2 -1; -6 -5 -4]
    
        Interaction_v_term = @mpoham (im*v*0.5) * sum(operator_threesite_final{i, i + 1, i + 2} for i in vertices(InfiniteChain(2)))
    end

    return Hopping_term + Mass_term + Interaction_term + Interaction_v_term
end

#=
data = Array{ComplexF64, 2}(undef, 8, 8)
data[2,5] = 0.5 + 0.0im
data[4,7] = -0.5 + 0.0im
data[5,2] = -0.5 + 0.0im
data[7,4] = 0.5 + 0.0im
=#