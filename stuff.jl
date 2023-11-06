using TensorOperations
using TensorKit
using MPSKit
using MPSKitModels
using LinearAlgebra: norm

using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_groundstate_energy.jl")
include("get_thirring_hamiltonian.jl")
include("get_thirring_hamiltonian_symmetric.jl")


a = Array{ComplexF64, 2}(undef, 8, 8)
a[2,5] = 0.5 + 0.0im
a[4,7] = -0.5 + 0.0im
a[5,2] = -0.5 + 0.0im
a[7,4] = 0.5 + 0.0im

display(a);