function get_thirring_hamiltonian(am_tilde_0, Delta_g, v, mu)
    
    Hopping_term = (-1) * @mpoham sum(S_xx(){i, i + 1} + S_yy(){i, i + 1} for i in vertices(InfiniteChain(2)))
    # change above to -1 !!!!!
    S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
    S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
    @tensor S¹[-1 -2; -3 -4] := S⁺[-1; -3] * S⁻[-2; -4]
    Hopping_term1 = (-1/2) * @mpoham sum(S¹{i, i + 1} for i in vertices(InfiniteChain(2)))
    Hopping_term2 = (-1/2) * @mpoham sum(S¹{i + 1, i} for i in vertices(InfiniteChain(2)))
    #Hopping_term = Hopping_term1 + Hopping_term2
    operator1 = S_z()
    id(domain(operator1))
    #Mass_term = am_tilde_0 * @mpoham sum(S_z(){i} + 0.5 * id(domain(S_z())){i} for i in vertices(InfiniteChain(2)))
    J = [1.0 -1.0]
    Mass_term = am_tilde_0 * @mpoham sum((J[i] * S_z() + 0.5*J[i] * id(domain(S_z()))){i} for i in vertices(InfiniteChain(2)))
    #Interaction_term = Delta_g * @mpoham sum(S_zz(){i, i + 1} + S_z(){i}/2 + id(domain(S_z())){i}/4 for i in vertices(InfiniteChain(2)))
    Interaction_term = @mpoham Delta_g * sum(S_zz(){i, i + 1} + (S_z() + 0.25*id(domain(S_z()))){i} for i in vertices(InfiniteChain(2)))

    Sz_plus_12 = S_z() + 0.5*id(domain(S_z()))
    # @tensor Interaction_O[-1 -2; -3 -4] := operator[-1; -3] * operator[-2; -4]

    operator = @mpoham sum((σˣˣ() + σʸʸ()){i, i+1} for i in vertices(InfiniteChain(2)))
    operator = @mpoham sum(σˣ(){i}*σˣ(){i+1} + σʸ(){i}*σʸ(){i+1} for i in vertices(InfiniteChain(2)))

    # Interaction_term = @mpoham sum(S_z_plus_1_2{i}*S_z_plus_1_2{i+1} for i in vertices(InfiniteChain(2))) # Werkt niet

    # @tensor Interaction_O[-1 -2; -3 -4] := S⁺[1; -3] * S⁻[-1; 1] * S⁺[2; -4] * S⁻[-2 2]
    # @tensor Interaction_O[-1 -2; -3 -4] := S⁻[1; -3] * S⁺[-1; 1] * S⁻[2; -4] * S⁺[-2 2]
    # Interaction_term = @mpoham Delta_g * sum(Interaction_O{i, i + 1} for i in vertices(InfiniteChain(2)))

    Interaction_term = @mpoham Delta_g * sum(Sz_plus_12{i}*Sz_plus_12{i+1} for i in vertices(InfiniteChain(2)))
    
    chemical_potential = @mpoham mu * sum(S_z(){i} for i in vertices(InfiniteChain(2)))

    #Interaction_term = @mpoham Delta_g * sum(S_zz(){i, i + 1} + (0.5*S_z() + 0.25*id(domain(S_z()))){i} + (0.5*S_z() + 0.25*id(domain(S_z()))){i+1} for i in vertices(InfiniteChain(2)))
# +- -1                                                          0.25*id(domain(S_z()))){i}

    #Interaction_v_term = @mpoham (Delta_g/2) * v*im * sum(S_xx(){i, i + 1} + S_yy(){i, i + 1} for i in vertices(InfiniteChain(2)))
    Interaction_v_term = 0
    #Hopping_v_term = -im*v * @mpoham sum(((S_xx() + S_yy()){i+2, i})*(S_z(){i+1}) - ((S_xx() + S_yy()){i, i+2})*(S_z(){i+1}) for i in vertices(InfiniteChain(2)))


    println(typeof(Hopping_term))
    println(typeof(Mass_term))
    println(typeof(Interaction_term))
    hamiltonian = Hopping_term + Mass_term + Interaction_term + chemical_potential
    return hamiltonian
end