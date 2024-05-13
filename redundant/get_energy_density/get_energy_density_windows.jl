using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using LaTeXStrings
using Random
using QuadGK

include("get_X_tensors.jl")
include("get_thirring_hamiltonian_symmetric.jl")
include("get_occupation_number_matrices.jl")

function get_left_mps(mps, middle)
    AL_new = copy(mps.AL)
    AC_new = copy(mps.AC)
    AR_new = copy(mps.AR)
    for w = 1:length(mps)
        @tensor new_tensor_AL[-1 -2; -3] := mps.AL[w][-1 1; -3] * middle[-2; 1]
        AL_new[w] = copy(new_tensor_AL)
        @tensor new_tensor_AR[-1 -2; -3] := mps.AR[w][-1 1; -3] * middle[-2; 1]
        AR_new[w] = copy(new_tensor_AR)
        @tensor new_tensor_AC[-1 -2; -3] := mps.AC[w][-1 1; -3] * middle[-2; 1]
        AC_new[w] = copy(new_tensor_AC)
    end

    return InfiniteMPS(AL_new, AR_new, mps.CR, AC_new)
end

function WindowMPS(left::InfiniteMPS{A,B}, right::InfiniteMPS{A,B}, ψ::InfiniteMPS{A,B}, L::Int; copyright=false, kwargs...) where {A,B}
    CLs = Vector{Union{Missing,B}}(missing, L + 1)
    ALs = Vector{Union{Missing,A}}(missing, L)
    ARs = Vector{Union{Missing,A}}(missing, L)
    ACs = Vector{Union{Missing,A}}(missing, L)

    ALs .= ψ.AL[1:L]
    ARs .= ψ.AR[1:L]
    ACs .= ψ.AC[1:L]
    CLs .= ψ.CR[0:L]

    return WindowMPS(left, FiniteMPS(ALs, ARs, ACs, CLs), right, fixleft = false, fixright = false)
end

N = 2
mass = 0.0
Delta_g = 0.0
v = 0.0
truncation = 2.5

@load "SanderDM_Thesis_2324/gs_mps_trunc_$(truncation)_mass_$(mass)_v_0.0_Delta_g_$(Delta_g)" mps

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
Plus_space = U1Space(1 => 1)
Min_space = U1Space(-1 => 1)
trivspace = U1Space(0 => 1)
S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace ⊗ Plus_space)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], pspace, pspace ⊗ Min_space)
S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)


middle = (2*im)*S_z_symm

left_gs = get_left_mps(mps, middle)
right_gs = copy(mps)

H0 = get_thirring_hamiltonian_symmetric(mass, Delta_g, v)

WindowH = Window(H0, repeat(H0, div(N,2)), H0)
Ψ = WindowMPS(right_gs, mps, right_gs, 2)

E = expectation_value(Ψ, WindowH, Ψ)