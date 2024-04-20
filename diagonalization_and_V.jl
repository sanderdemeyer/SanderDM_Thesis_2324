using LinearAlgebra

function V_matrix(X, m)
    N = length(X)

    F = zeros(ComplexF64, 2*N, 2*N)
    for i = 0:N-1
        k = X[i+1]
        for n = 0:N-1
            F[2*n+1, 2*i+1] = 1/sqrt(N) * exp(-im*k*(n+1))
            F[2*n+2, 2*i+2] = 1/sqrt(N) * exp(-im*k*(n+1))
        end
    end

    PN = zeros(ComplexF64, 2*N, 2*N)
    for i = 0:N-1
        k = X[i+1]
        A = [m -(im/2)*(1-exp(im*k)); (im/2)*(1-exp(-im*k)) -m]
        eigen_result = eigen(A)
        eigenvectors_matrix = eigen_result.vectors
        PN[2*i+1:2*i+2,2*i+1:2*i+2] = eigenvectors_matrix
    end
    
    V = F * PN

    Vp_indices = []
    Vm_indices = []
    for iₖ = 0:N-1
        k = X[iₖ+1]
        if (k < 0.0)
            push!(Vp_indices, 2*iₖ+1)
            push!(Vm_indices, 2*iₖ+2)
        else
            push!(Vp_indices, 2*iₖ+2)
            push!(Vm_indices, 2*iₖ+1)
        end
    end
    permutation = vcat(Vp_indices, Vm_indices)

    Vpermuted = zeros(ComplexF64, 2*N, 2*N)
    for i = 1:2*N
        for j = 1:2*N
            Vpermuted[i,j] = V[i,permutation[j]]
            # Vpermuted[i,permutation[j]] = V[i,j]
        end
    end
    
    V₋ = Vpermuted[:,1:N]
    V₊ = Vpermuted[:,N+1:2*N]

    return (V₊,V₋)
end

N = 10
X = [(2*pi)/N*i - pi for i = 0:N-1]
m = 0.3

for i = 0:N-1
    k = X[i+1]
    A = [m -(im/2)*(1-exp(im*k)); (im/2)*(1-exp(-im*k)) -m]
    lambda = sqrt(m^2+sin(k/2)^2)
    eigen_result = eigen(A)
    eigenvectors_matrix = eigen_result.vectors
    # println("for i = $(i), \n: $eigenvectors_matrix")
    c1p = eigenvectors_matrix[1,1]
    c2p = eigenvectors_matrix[2,1]
    c1m = eigenvectors_matrix[1,2]
    c2m = eigenvectors_matrix[2,2]
    
    # println((c1m/c2m))
    # println(-((m+lambda)/sin(k/2))*exp(im*k/2))


    c1p_zelf = exp(im*k/2)
    c2p_zelf = -sin(k/2)/(m-lambda)
    NORM = sqrt(abs(c1p_zelf)^2+abs(c2p_zelf)^2)
    c1p_zelf = c1p_zelf/NORM
    c2p_zelf = c2p_zelf/NORM

    c1m_zelf = exp(im*k/2)
    c2m_zelf = -sin(k/2)/(m+lambda)
    NORM = sqrt(abs(c1m_zelf)^2+abs(c2m_zelf)^2)
    c1m_zelf = c1m_zelf/NORM
    c2m_zelf = c2m_zelf/NORM

    
    println(c1p/c1p_zelf)
    println(c2p/c2p_zelf)
    println(c1m/c1m_zelf)
    println(c2m/c2m_zelf)

    # println("k = $(k), phase = $(angle(c1p))")
    # println("k = $(k), phase2 = $(angle(exp(im*k/2)))")
end

