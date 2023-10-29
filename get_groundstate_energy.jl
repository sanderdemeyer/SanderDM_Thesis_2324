include("correlation_function.jl")
using Dates

function get_groundstate_energy(am_tilde_0, Delta_g, v, mu, D)
    spin = 1//2
    # For a given bond dimension, calculate the ground state energy of the Thirring model with given parameters
    state = InfiniteMPS([ℂ^2, ℂ^2],[ℂ^D, ℂ^D])

    #Symmetric:
    pspace = U1Space(i => 1 for i in (-spin):spin)
    vspace_L = U1Space(1//2 => D, -1//2 => D)
    vspace_R = U1Space(1 => D, 0 => D, -1 => D)
    state = InfiniteMPS([pspace, pspace], [vspace_L, vspace_R])
    println(typeof(pspace))
    println(typeof(:U1))

    # hamiltonian = get_thirring_hamiltonian(am_tilde_0, Delta_g, v, mu)
    hamiltonian = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v, mu)

    # identity_op = @mpoham sum(id(domain(S_z())){i} for i in vertices(InfiniteChain(2)))

    # J = [1.0 -1.0]
    # xi_operator = @mpoham sum(J[i] * S_z(){i} for i in vertices(InfiniteChain(2)))
    println(typeof(state))
    println(typeof(hamiltonian))

    (groundstate,_) = find_groundstate(state,hamiltonian,VUMPS(maxiter = 30))

    println("Done with VUMPS")
    # one = expectation_value(groundstate, identity_op)

    # xi = expectation_value(groundstate, xi_operator)


    gs_energy = expectation_value(groundstate, hamiltonian)
    gs_energy = (gs_energy[1]+gs_energy[2])/2

    println("Gs_energy is $gs_energy")

    S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)

    magnetization = expectation_value(groundstate, S_z_symm)
    magnetization = (magnetization[1]+magnetization[2])/2
    println("Magnetization is $magnetization")

    # S⁺ = TensorMap([0.0 1.0; 0.0 0.0], ℂ^2, ℂ^2)
    # S⁻ = TensorMap([0.0 0.0; 1.0 0.0], ℂ^2, ℂ^2)

    # max_dist = 200
    # sz_mpo = TensorMap([1.0 0; 0 -1.0], ℂ^2, ℂ^2)
    # corr_function_string1 = correlation_function(groundstate, S⁺, sz_mpo, S⁻, max_dist)
    # corr_function_string2 = correlation_function(groundstate, S⁻, sz_mpo, S⁺, max_dist)
    corr_function_string1 = 0

    return gs_energy, magnetization
    #=

    operator10 = @mpoham sum(S_z(){i} for i in vertices(InfiniteChain(2))) 
    operator2 = S_z(ComplexF64, Trivial, spin=1//2);
    operator2 = S_z(ComplexF64, Trivial, spin=1//2);
    
    sz_mpo = TensorMap([1.0 0; 0 -1.0], ℂ^1 * ℂ^2, ℂ^2 * ℂ^1)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)

    sz_ham = @mpoham sum(S_z(){i} for i in vertices(InfiniteChain(2)))


#    sz_mpo1 = @mpoham sum(sz_mpo{i} for i in vertices(InfiniteChain(2))) 

    println("first corr function")
    t₀ = now()
    w = 2
    max_dist = 100
    corr_function = zeros(ComplexF64, max_dist)
    for i = 1:w
        corr_function += correlator(groundstate, sz_mpo, sz_mpo, i, i+1:i+max_dist)
    end
    corr_function ./= w
    t₁ = now()
    sz_mpo = TensorMap([1.0 0; 0 -1.0], ℂ^2, ℂ^2)

    corr_function2 = correlation_function(groundstate, sz_mpo, sz_mpo, max_dist)
    t₂ = now()    

    println(norm(corr_function2 - corr_function)/sqrt(max_dist))

    println("MPSKit took $(t₁-t₀) seconds, own implementation took $(t₂-t₁) seconds")

    S⁺ = TensorMap([0.0 1.0; 0.0 0.0], ℂ^2, ℂ^2)
    S⁻ = TensorMap([0.0 0.0; 1.0 0.0], ℂ^2, ℂ^2)

    sz_mpo = TensorMap([1.0 0; 0 -1.0], ℂ^2, ℂ^2)
    corr_function_string1 = correlation_function(groundstate, S⁺, sz_mpo, S⁻, max_dist)
    corr_function_string2 = correlation_function(groundstate, S⁻, sz_mpo, S⁺, max_dist)

    println("string")
    println(corr_function_string2)

    corr = correlator(groundstate, S_xx(), 1, 2:1+max_dist)
    println("fjdskqlmfjklm")
    println(corr)
    =#

    #=
    disconnected_base = expectation_value(groundstate, sz_ham)
    disconnected = [disconnected_base[(i)%2+1] for i in 1:length(corr_function)]
    println(disconnected_base[1])
    connected_corr_function = corr_function .- disconnected_base[1] * disconnected

    odd_x = [connected_corr_function[2*n+1] for n in 1:floor(Int, length(connected_corr_function)/2)-1]
    println(odd_x)

    S⁺ = TensorMap([0.0 1.0; 0.0 0.0], ℂ^2, ℂ^2)
    S⁻ = TensorMap([0.0 0.0; 1.0 0.0], ℂ^2, ℂ^2)

    sz_mpo = TensorMap([1.0 0; 0 -1.0], ℂ^2, ℂ^2)
    corr_function_string = correlation_function(groundstate, S⁺, sz_mpo, S⁻, max_dist)

    println("string")
    println(corr_function_string)
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
    =#
    
    # gs_correlation_length = 0
    # return gs_energy, gs_correlation_length
end