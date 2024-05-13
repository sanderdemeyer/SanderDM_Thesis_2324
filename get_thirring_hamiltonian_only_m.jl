function get_thirring_hamiltonian_only_m(am_tilde_0)
    J = [1.0 -1.0]
    Sz_plus_12 = S_z() + 0.5*id(domain(S_z()))

    Hopping_term = (-1) * @mpoham sum(S_xx(){i, i + 1} + S_yy(){i, i + 1} for i in vertices(InfiniteChain(2)))
    Mass_term = am_tilde_0 * @mpoham sum((J[i] * Sz_plus_12){i} for i in vertices(InfiniteChain(2)))
    hamiltonian = Hopping_term + Mass_term
    return hamiltonian
end