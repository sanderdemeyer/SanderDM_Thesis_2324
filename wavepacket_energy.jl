using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots

include("get_thirring_hamiltonian_symmetric.jl")

@load "SanderDM_Thesis_2324/gs_mps_trunc_2.5_mass_0.0_v_0.0_Delta_g_0.0" mps

N = 50
m = 0.0
delta_g = 0.0
v = 0.0

tensors = [i%2 == 0 ? mps.AR[2] : mps.AR[1] for i in 1:N]
tensors[1] = mps.AC[1]

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)

S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace)
# S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace ⊗ Triv_space)

@tensor tensors[1][-1 -2; -3] = tensors[1][-1 1; -3] * S_z_symm[-2; 1]
@tensor tensors[N][-1 -2; -3] = tensors[N][-1 1; -3] * S_z_symm[-2; 1]

println(typeof(mps.AC[1]))

Ψ = FiniteMPS(tensors)

H = get_thirring_hamiltonian_symmetric(m, delta_g, v)

E = expectation_value(Ψ, H)
println(E)

