using LinearAlgebra
# using Base
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_occupation_number_matrices.jl")

function get_energy(k, m, v)
    return (v/2*sin(k) + sqrt(m^2 + sin(k/2)^2), v/2*sin(k) - sqrt(m^2 + sin(k/2)^2))
end

mass = 0.3
Delta_g = 0.0
v = 0.0

N = 10

X = [(2*pi)/N*i - pi for i = 0:N-1]
X_new = [(2*pi)/N*i for i = 0:N-1]



k₀ = pi/5
σ = 2*(2*pi/N)
x₀ = div(N,2)

H = zeros(ComplexF64, 2*N, 2*N)

for i = 1:2*N
    H[i,i] = (-1)^(i-1) * (mass/2)
    H[i,mod1(i+1,2*N)] = -im/2
    H[i,mod1(i+2,2*N)] = (im*v/4)
end

H = H + adjoint(H)



# for i = 1:2*N
#     for j = 1:2*N
#         if ((i) % 2 == 0)
#             F[i,j] = 1/sqrt(2*N) * exp(-im*X[div(j,2)+1]*j)


#         end
#     end
# end

F = zeros(ComplexF64, 2*N, 2*N)
for i = 0:N-1
    k = X[i+1]
    for n = 0:N-1
        # F[2*i+1, 2*n+1] = 1/sqrt(N) * exp(-im*k*n)
        # F[2*i+2, 2*n+2] = 1/sqrt(N) * exp(-im*k*n)
        F[2*n+1, 2*i+1] = 1/sqrt(N) * exp(-im*k*n)
        F[2*n+2, 2*i+2] = 1/sqrt(N) * exp(-im*k*n)
    end
end

D = adjoint(F) * H * F

PN = zeros(ComplexF64, 2*N, 2*N)
for i = 0:N-1
    k = X_new[i+1]
    A = [mass -(im/2)*(1-exp(im*k)); (im/2)*(1-exp(-im*k)) -mass]
    println(A)
    eigen_result = eigen(A)
    eigenvectors_matrix = eigen_result.vectors

    if (k < 0.0)
        v₊¹ = -eigenvectors_matrix[1,2]
        v₊² = eigenvectors_matrix[2,2]
        v₋¹ = eigenvectors_matrix[1,1]
        v₋² = -eigenvectors_matrix[2,1]
    else
        v₊¹ = eigenvectors_matrix[1,1]
        v₊² = -eigenvectors_matrix[2,1]
        v₋¹ = -eigenvectors_matrix[1,2]
        v₋² = eigenvectors_matrix[2,2]
    end
    # PN[2*i+1,2*i+1] = v₊¹
    # PN[2*i+1,2*i+2] = v₋¹
    # PN[2*i+2,2*i+1] = v₊²
    # PN[2*i+2,2*i+2] = v₋²

    # PN[2*i+1,2*i+1] = eigenvectors_matrix[1,1]
    # PN[2*i+1,2*i+2] = eigenvectors_matrix[1,2]
    # PN[2*i+2,2*i+1] = eigenvectors_matrix[2,1]
    # PN[2*i+2,2*i+2] = eigenvectors_matrix[2,2]

    # PN[2*i+1,i+1] = eigenvectors_matrix[1,1]
    # PN[2*i+1,i+1+N] = eigenvectors_matrix[1,2]
    # PN[2*i+2,i+1] = eigenvectors_matrix[2,1]
    # PN[2*i+2,i+1+N] = eigenvectors_matrix[2,2]

    if (k < 0.0)
        PN[2*i+1,i+1] = -eigenvectors_matrix[2,1]
        PN[2*i+1,i+1+N] = eigenvectors_matrix[2,2]
        PN[2*i+2,i+1] = eigenvectors_matrix[1,1]
        PN[2*i+2,i+1+N] = -eigenvectors_matrix[1,2]
    else
        PN[2*i+1,i+1] = eigenvectors_matrix[2,2]
        PN[2*i+1,i+1+N] = -eigenvectors_matrix[2,1]
        PN[2*i+2,i+1] = -eigenvectors_matrix[1,2]
        PN[2*i+2,i+1+N] = eigenvectors_matrix[1,1]
    end
end

# PN[2*i+1,i+1] = eigenvectors_matrix[2,1]
# PN[2*i+1,i+1+N] = eigenvectors_matrix[2,2]
# PN[2*i+2,i+1] = eigenvectors_matrix[1,1]
# PN[2*i+2,i+1+N] = eigenvectors_matrix[1,2]

#NEW checks
count = 0
for i=1:2*N
    for j=1:2*N
        if abs(D[i,j]) > 10^(-10)
            global count
            count += 1
            println("i = $(i), j = $(j)")
        end
    end
end

for i = 0:N-1
    A2 = D[2*i+1:2*i+2,2*i+1:2*i+2]
    k = X[i+1]
    A = [mass -(im/2)*(1-exp(im*k)); (im/2)*(1-exp(-im*k)) -mass]
    println(norm(A2-A))
end


for i = 1:N
    k = X[i]
    println((-im/2)*(1-exp(im*k)))
end

for i = 0:N-1
    println(D[2*i+1,2*i+2])
end

D_diag = adjoint(PN) * D * PN

for i = 1:2*N
    println(D_diag[i,i])
end

for k in X
    println(sqrt(mass^2 + sin(k/2)^2))
end

eig = eigen(H)
for e in eig.values
    if e > 0.3
        println("for k = $(2*asin(sqrt(e^2-mass^2)))")
    end
end


PN = zeros(ComplexF64, 2*N, 2*N)
for i = 0:N-1
    k = X[i+1]
    A = [mass -(im/2)*(1-exp(im*k)); (im/2)*(1-exp(-im*k)) -mass]
    eigen_result = eigen(A)
    eigenvectors_matrix = eigen_result.vectors
    PN[2*i+1:2*i+2,2*i+1:2*i+2] = eigenvectors_matrix
end

D_blocks_to_diag = adjoint(PN) * D * PN

Vnew = F * PN
D_blocks_to_diag2 = adjoint(Vnew) * H * Vnew

for i = 1:2*N
    println(D_blocks_to_diag[i,i])
    println("diff is $(D_blocks_to_diag[i,i]-D_blocks_to_diag2[i,i])")
end

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
        Vpermuted[i,j] = Vnew[i,permutation[j]]
    end
end

newest = adjoint(Vpermuted) * H * Vpermuted

for i = 1:2*N
    println(newest[i,i])
end

V₊ = Vpermuted[:,1:N]
V₋ = Vpermuted[:,N+1:2*N]

D₊ = adjoint(V₊) * H * V₊
D₋ = adjoint(V₋) * H * V₋

for i = 1:N
    println(D₊[i,i])
end
for i = 1:N
    println(D₋[i,i])
end

break

D_new = adjoint(PN) * D * PN


EV = F * PN

D_new2 = adjoint(EV) * H * EV

EV₊ = EV[:,1:N]
EV₋ = EV[:,N+1:2*N]

D₊ = adjoint(EV₊) * H * EV₊
D₋ = adjoint(EV₋) * H * EV₋


for i = 1:2*N
    # println(D_new2[i,i])
    if real(D_new2[i,i]) > 0.0
        println("for k = $(asin(real(sqrt(D_new2[i,i]^2-mass^2))))")
    end
end


eigen_result = eigen(H)
eigenvectors_matrix = eigen_result.vectors
for el in eigen_result.values
    # println(D_new2[i,i])
    if el > 0.3
        println(el)
        println("for k = $(asin(real(sqrt(el^2-mass^2))))")
    end
end



for k in X_new
    println(sqrt(mass^2+sin(k)^2))
end

D_new3 = V*H*adjoint(V)
for i = 1:2*N
    println(D_new3[i,i])
end


for i = 1:2*N
    # println(D_new2[i,i])
    if real(D_new3[i,i]) > 0.0
        println("for k = $(asin(real(sqrt(D_new[i,i]^2-mass^2))))")
    end
end

break

(V₊,V₋) = V_matrix(X, mass)
V = vcat(V₊,V₋)
gaussian = gaussian_array(X, pi/4, σ, x₀)

D = V * H * adjoint(V)
diagonals = []
for i = 1:2*N
    push!(diagonals,D[i,i])
end

E_X_pos = [get_energy(k,mass,v)[1] for k in X]
E_X_neg = [get_energy(k,mass,v)[2] for k in X]

break

# hamiltonian = get_thirring_hamiltonian(mass, Delta_g, v)

A = (typeof(mps)).parameters[1]
S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)

O = S⁺
Xs = fill(id(domain(S_z())), N)

G = gaussian_array(X, k₀, σ, x₀)
(V₊,V₋) = V_matrix(X, mass)

wi = G * V₊
