function get_thirring_hamiltonian(am_tilde_0, Delta_g, v)
    J = [1.0 -1.0]
    Sz_plus_12 = S_z() + 0.5*id(domain(S_z()))

    S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
    S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
    @tensor operator_threesite[-1 -2 -3; -4 -5 -6] := S⁺[-1; -4] * S_z()[-2; -5] * S⁻[-3; -6]
    @tensor operator_threesite_final[-1 -2 -3; -4 -5 -6] := operator_threesite[-1 -2 -3; -4 -5 -6] - operator_threesite[-3 -2 -1; -6 -5 -4]

    display((blocks(S_xx() + S_yy())));
    println("done1")

    display((blocks(Sz_plus_12)));
    println("done2")

    Hopping_term = (-1) * @mpoham sum(S_xx(){i, i + 1} + S_yy(){i, i + 1} for i in vertices(InfiniteChain(2)))
    Mass_term = am_tilde_0 * @mpoham sum((J[i] * Sz_plus_12){i} for i in vertices(InfiniteChain(2)))
    Interaction_term = @mpoham Delta_g * sum(Sz_plus_12{i}*Sz_plus_12{i+1} for i in vertices(InfiniteChain(2)))
    Interaction_v_term = @mpoham (im*v*0.5) * sum(operator_threesite_final{i, i + 1, i + 2} for i in vertices(InfiniteChain(2)))
    hamiltonian = Hopping_term + Mass_term + Interaction_term + Interaction_v_term
    return hamiltonian
#    Interaction_v_term_1 = @mpoham (im*v/2) * sum(S⁺{i}*S_z(){i+1}*S⁻{i+2} for i in vertices(InfiniteChain(2))) 
#    Interaction_v_term_2 = @mpoham (im*v/2) * sum(S⁻{i}*S_z(){i+1}*S⁺{i+2} for i in vertices(InfiniteChain(2))) 
#    Interaction_v_term = Interaction_v_term_1 - Interaction_v_term_2




    # change above to -1 !!!!!
    # @tensor S¹[-1 -2; -3 -4] := S⁺[-1; -3] * S⁻[-2; -4]
    # Hopping_term1 = (-1/2) * @mpoham sum(S¹{i, i + 1} for i in vertices(InfiniteChain(2)))
    # Hopping_term2 = (-1/2) * @mpoham sum(S¹{i + 1, i} for i in vertices(InfiniteChain(2)))
    #Hopping_term = Hopping_term1 + Hopping_term2
    # operator1 = S_z()
    # id(domain(operator1))
    #Mass_term = am_tilde_0 * @mpoham sum(S_z(){i} + 0.5 * id(domain(S_z())){i} for i in vertices(InfiniteChain(2)))

    # Interaction_term = Delta_g * @mpoham sum(S_zz(){i, i + 1} + S_z(){i}/2 + id(domain(S_z())){i}/4 for i in vertices(InfiniteChain(2)))
    # Interaction_term = @mpoham Delta_g * sum(S_zz(){i, i + 1} + (S_z() + 0.25*id(domain(S_z()))){i} for i in vertices(InfiniteChain(2)))


    # @tensor Interaction_O[-1 -2; -3 -4] := operator[-1; -3] * operator[-2; -4]

    # operator = @mpoham sum((σˣˣ() + σʸʸ()){i, i+1} for i in vertices(InfiniteChain(2)))
    # operator = @mpoham sum(σˣ(){i}*σˣ(){i+1} + σʸ(){i}*σʸ(){i+1} for i in vertices(InfiniteChain(2)))

    # Interaction_term = @mpoham sum(S_z_plus_1_2{i}*S_z_plus_1_2{i+1} for i in vertices(InfiniteChain(2))) # Werkt niet

    # @tensor Interaction_O[-1 -2; -3 -4] := S⁺[1; -3] * S⁻[-1; 1] * S⁺[2; -4] * S⁻[-2 2]
    # @tensor Interaction_O[-1 -2; -3 -4] := S⁻[1; -3] * S⁺[-1; 1] * S⁻[2; -4] * S⁺[-2 2]
    # Interaction_term = @mpoham Delta_g * sum(Interaction_O{i, i + 1} for i in vertices(InfiniteChain(2)))

    
    # chemical_potential = @mpoham mu * sum(S_z(){i} for i in vertices(InfiniteChain(2)))

    #Interaction_term = @mpoham Delta_g * sum(S_zz(){i, i + 1} + (0.5*S_z() + 0.25*id(domain(S_z()))){i} + (0.5*S_z() + 0.25*id(domain(S_z()))){i+1} for i in vertices(InfiniteChain(2)))
# +- -1                                                          0.25*id(domain(S_z()))){i}

    #Interaction_v_term = @mpoham (Delta_g/2) * v*im * sum(S_xx(){i, i + 1} + S_yy(){i, i + 1} for i in vertices(InfiniteChain(2)))
    #Hopping_v_term = -im*v * @mpoham sum(((S_xx() + S_yy()){i+2, i})*(S_z(){i+1}) - ((S_xx() + S_yy()){i, i+2})*(S_z(){i+1}) for i in vertices(InfiniteChain(2)))


    println(typeof(Hopping_term))
    println(typeof(Mass_term))
    println(typeof(Interaction_term))
end