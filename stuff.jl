using TensorOperations
using TensorKit
using MPSKit
using MPSKitModels
using LinearAlgebra: norm

using LinearAlgebra
using Base
using JLD2
# using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
include("get_groundstate_energy.jl")
include("get_thirring_hamiltonian.jl")
include("get_thirring_hamiltonian_symmetric.jl")


spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
S_z_symm2 = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)
H = @mpoham sum((S_z_symm2){i} for i in vertices(InfiniteChain(2)))
Plus_space = U1Space(1 => 1)
Triv_space = U1Space(0 => 1)
S⁺2 = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Triv_space ⊗ pspace, pspace ⊗ Plus_space)
S⁻2 = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace ⊗ Triv_space)

@load "operators_for_occupation_number" S⁺ S⁻ S_z_symm


break

x = 1:10
y1 = rand(10)
y2 = rand(10)
plt = plot(x, y1, label="Plot 1", xlabel="X-axis", ylabel="Y-axis", linewidth=2)
plot!(x, y2, label="Plot 2", linestyle=:dash, linewidth=2)
display(plt)

break

function M_H(m, v, k)
    return [m + (v/2)*sin(k) -(im/2)*(1-exp(im*k)) ; (im/2)*(1-exp(-im*k)) -m+(v/2)*sin(k)]
end

function P_H(m, v, k)
    λ = sqrt(m^2+sin(k/2)^2)
    # c₊¹ = -1
    # c₊² = -exp(-im*k/2)*(m-λ)/sin(k/2)
    return [-1 -1; -exp(-im*k/2)*(m-λ)/sin(k/2) -exp(-im*k/2)*(m+λ)/sin(k/2)]
end

A = [1 1; 1 -1]
P = [1 -1; sqrt(2)-1 sqrt(2)+1]

println(A)
println(P)

for i = 2:15
    println(norm(MPSs[i].window.AC[1]-MPSs[i-1].window.AC[1]))
end

for i = 2:15
    println(norm(WindowMPSs[i].AC[1]-WindowMPSs[i-1].AC[1]))
end

i = 2
N = 7
X = [(2*pi)/N*i - pi for i = 0:N-1]
plt = plot(X, occ_numbers[i,:], xlabel = "k")
title!("Occupation number for N = $(N)")
display(plt)



testt = 0.0

for i = 1:10
    global testt

    testt += 0.2
    println(testt)
end


N = 20
a = 7
b = 11
lijst = [i < a ? 0 : (i <= b ? (i - a) / (b - a) : 1) for i in 1:N]
println(lijst)

print(q)

factor_max = 48
list_of_indices = [0]

for factor = 1:factor_max
    for j = 1:4
        if !(j*factor in list_of_indices)
            push!(list_of_indices, j*factor)
            push!(list_of_indices, -j*factor)
        end
    end
end
println(list_of_indices)

a = Array{ComplexF64, 2}(undef, 8, 8)
a[2,5] = 0.5 + 0.0im
a[4,7] = -0.5 + 0.0im
a[5,2] = -0.5 + 0.0im
a[7,4] = 0.5 + 0.0im

display(a);