function get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v)
    spin = 1//2
    pspace = U1Space(i => 1 for i in (-spin):spin)

    S_xx_S_yy = 0.5*TensorMap([0.0+0.0im 0.0+0.0im 0.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im 1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im 0.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im 0.0+0.0im 0.0+0.0im], pspace ⊗ pspace, pspace ⊗ pspace)

    #=
    display((blocks(S_xx_S_yy)[U1Irrep(1)]));
    display((blocks(S_xx_S_yy)[U1Irrep(-1)]));
    display((blocks(S_xx_S_yy)[U1Irrep(0)]));
    println("done1")
    =#

    S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)
    J = [1.0 -1.0]
    Sz_plus_12 = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace)
        
    #=
    display((blocks(Sz_plus_12)[U1Irrep(1//2)]));
    display((blocks(Sz_plus_12)[U1Irrep(-1//2)]));
    println("done2")
    =#
    Hopping_term = (-1) * @mpoham sum(S_xx_S_yy{i, i + 1} for i in vertices(InfiniteChain(2)))
    Mass_term = am_tilde_0 * @mpoham sum(J[i] * (S_z_symm + 0.5*id(domain(S_z_symm))){i} for i in vertices(InfiniteChain(2)))
    Interaction_term = @mpoham Delta_g * sum(Sz_plus_12{i}*Sz_plus_12{i+1} for i in vertices(InfiniteChain(2)))
    #hamiltonian = Hopping_term + Mass_term + Interaction_term
    #return hamiltonian
    #Sz_plus_12 = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace)

    data = Array{ComplexF64, 2}(undef, 8, 8)
    data[2,5] = 0.5 + 0.0im
    data[4,7] = -0.5 + 0.0im
    data[5,2] = -0.5 + 0.0im
    data[7,4] = 0.5 + 0.0im

    #three_site_operator = TensorMap(data, pspace ⊗ pspace ⊗ pspace, pspace ⊗ pspace ⊗ pspace)
    #Interaction_v_term = @mpoham (im*v/2) * sum(three_site_operator{i, i + 1, i + 2} for i in vertices(InfiniteChain(2)))

    Plus_space = U1Space(1 => 1)
    Min_space = U1Space(-1 => 1)
    
    S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace ⊗ Plus_space)
    S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace)
    S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], Plus_space ⊗ pspace, pspace ⊗ Plus_space)

    @tensor operator_threesite[-1 -2 -3; -4 -5 -6] := S⁺[-1; -4 1] * S_z_symm[1 -2; -5 2] * S⁻[2 -3; -6]
    @tensor operator_threesite_final[-1 -2 -3; -4 -5 -6] := operator_threesite[-1 -2 -3; -4 -5 -6] - operator_threesite[-3 -2 -1; -6 -5 -4]

    S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)

    Hopping_term = (-1) * @mpoham sum(S_xx_S_yy{i, i + 1} for i in vertices(InfiniteChain(2)))
    Mass_term = am_tilde_0 * @mpoham sum(J[i] * (S_z_symm + 0.5*id(domain(S_z_symm))){i} for i in vertices(InfiniteChain(2)))
    
    M = Mass_term[1]
    println(collect(keys(M))) 

    Interaction_term = @mpoham Delta_g * sum(Sz_plus_12{i}*Sz_plus_12{i+1} for i in vertices(InfiniteChain(2)))
    Interaction_v_term = @mpoham (im*v*0.5) * sum(operator_threesite_final{i, i + 1, i + 2} for i in vertices(InfiniteChain(2)))
    hamiltonian = Hopping_term + Mass_term + Interaction_term + Interaction_v_term
    return hamiltonian

    display(reshape(convert(Array, operator_threesite_final), 8, 8))
    println("done")

    println(blocks(operator_threesite_final))

    display(blocks(operator_threesite_final)[U1Irrep(-3//2)])
    display(blocks(operator_threesite_final)[U1Irrep(-1//2)])
    display(blocks(operator_threesite_final)[U1Irrep(1//2)])
    display(blocks(operator_threesite_final)[U1Irrep(3//2)])
        # @tensor Interaction_O[-1 -2; -3 -4] := S⁺[1; -3] * S⁻[-1; 1] * S⁺[2; -4] * S⁻[-2 2]
    # @tensor Interaction_O[-1 -2; -3 -4] := S⁻[1; -3] * S⁺[-1; 1] * S⁻[2; -4] * S⁺[-2 2]

    #=
    println("Block sectors")
    println(blocksectors(three_site_operator))
    println("Blocks")
    println(blocks(three_site_operator))
    println("fusion trees")
    println(fusiontrees(three_site_operator))
    println("double")
    println(double(three_site_operator))
    =#
end