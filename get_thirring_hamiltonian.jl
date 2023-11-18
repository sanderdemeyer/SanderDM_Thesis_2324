function get_thirring_hamiltonian(am_tilde_0, Delta_g, v)
    J = [1.0 -1.0]
    Sz_plus_12 = S_z() + 0.5*id(domain(S_z()))

    S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
    S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
    @tensor operator_threesite[-1 -2 -3; -4 -5 -6] := S⁺[-1; -4] * S_z()[-2; -5] * S⁻[-3; -6]
    @tensor operator_threesite_final[-1 -2 -3; -4 -5 -6] := operator_threesite[-1 -2 -3; -4 -5 -6] - operator_threesite[-3 -2 -1; -6 -5 -4]

    Hopping_term = (-1) * @mpoham sum(S_xx(){i, i + 1} + S_yy(){i, i + 1} for i in vertices(InfiniteChain(2)))
    Mass_term = am_tilde_0 * @mpoham sum((J[i] * Sz_plus_12){i} for i in vertices(InfiniteChain(2)))
    Interaction_term = @mpoham Delta_g * sum(Sz_plus_12{i}*Sz_plus_12{i+1} for i in vertices(InfiniteChain(2)))
    Interaction_v_term = @mpoham (im*v*0.5) * sum(operator_threesite_final{i, i + 1, i + 2} for i in vertices(InfiniteChain(2)))
    hamiltonian = Hopping_term + Mass_term + Interaction_term + Interaction_v_term
    return hamiltonian
end