function get_groundstate_energy(am_tilde_0, Delta_g, v, D)
    # For a given bond dimension, calculate the ground state energy of the Thirring model with given parameters
    state = InfiniteMPS([ℂ^2, ℂ^2],[ℂ^D, ℂ^D])

    hamiltonian = get_thirring_hamiltonian(am_tilde_0, Delta_g, v)

    J = [1.0 -1.0]
    xi_operator = @mpoham sum(J[i] * S_z(){i} for i in vertices(InfiniteChain(2)))

    (groundstate,_) = find_groundstate(state,hamiltonian,VUMPS(maxiter = 6))

    xi = expectation_value(groundstate, xi_operator)
    println("xi = ")
    println(xi)

    operator10 = @mpoham sum(S_z(){i} for i in vertices(InfiniteChain(2))) 
    operator2 = S_z(ComplexF64, Trivial, spin=1//2);
    operator2 = S_z(ComplexF64, Trivial, spin=1//2);
    
    sz_mpo = TensorMap([1.0 0; 0 -1.0], ℂ^1 * ℂ^2, ℂ^2 * ℂ^1)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)

    sz_ham = @mpoham sum(S_z(){i} for i in vertices(InfiniteChain(2)))


#    sz_mpo1 = @mpoham sum(sz_mpo{i} for i in vertices(InfiniteChain(2))) 

    w = 2
    max_dist = 200
    corr_function = zeros(ComplexF64, max_dist)
    for i = 1:1
        corr_function += correlator(groundstate, sz_mpo, sz_mpo, i, i+1:i+max_dist)
    end
    corr_function ./= w
    disconnected_base = expectation_value(groundstate, sz_ham)
    disconnected = [disconnected_base[(i)%2+1] for i in 1:length(corr_function)]
    println(disconnected_base[1])
    connected_corr_function = corr_function .- disconnected_base[1] * disconnected

    odd_x = [connected_corr_function[2*n+1] for n in 1:floor(Int, length(connected_corr_function)/2)-1]
    println(odd_x)
    #=
    f = S_z;
    println(typeof(f))
    println(typeof(S_z))
    println("This was the type")
    # f should be MPOTensor
    corr_function = correlator(groundstate, operator10, operator10, 1, 20)
    println("fdjklqsm")
    corr_function = correlator(groundstate, f, f, 1, 1:20)
    println(corr_function)
    
    spectrum = transfer_spectrum(groundstate)
    gs_correlation_length = -1/log(abs(spectrum[2]))

    =#
    gs_energy = expectation_value(groundstate, hamiltonian)
    gs_correlation_length = correlation_length(groundstate)
    # return gs_energy
    return gs_energy, gs_correlation_length
end