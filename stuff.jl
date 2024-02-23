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

include("get_groundstate_energy.jl")
include("get_thirring_hamiltonian.jl")
include("get_thirring_hamiltonian_symmetric.jl")

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