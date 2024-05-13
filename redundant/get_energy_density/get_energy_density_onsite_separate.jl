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
include("get_thirring_hamiltonian.jl")
include("get_occupation_number_matrices.jl")
include("get_thirring_hamiltonian_only_m.jl")
include("get_thirring_hamiltonian_symmetric.jl")
include("get_groundstate_energy.jl")


S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
mass_term = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)

@tensor hopping[-1 -2; -3 -4] := (-2)*(S⁺[-1; -3]*S⁻[-2; -4] + S⁻[-1; -3]*S⁺[-2; -4])

# @load "SanderDM_Thesis_2324/gs_mps_trunc_$(truncation)_mass_$(mass)_v_0.0_Delta_g_$(Delta_g)" mps
Xs = get_X_tensors(mps.AL) # geeft hetzelfde indiend AC of AR

@tensor check11[-1 -2; -3] := inv(Xs[1])[-1; 1] * mps.AC[1][1 -2; 2] * Xs[2][2; -3]
@tensor check12[-1 -2; -3] := mps.AC[1][-1 1; -3] * (2*im)*S_z_symm[-2; 1]
@tensor check21[-1 -2; -3] := inv(Xs[2])[-1; 1] * mps.AC[2][1 -2; 2] * Xs[1][2; -3]
@tensor check22[-1 -2; -3] := mps.AC[2][-1 1; -3] * (2*im)*S_z_symm[-2; 1]
@assert norm(check11-check12) < 1e-10
@assert norm(check21-check22) < 1e-10

mps_asymm = remove_symmetries(mps)
Xs_asymm = convert_to_array(Xs)
S_z_asymm = convert_to_array(S_z_symm)


@tensor check11[-1 -2; -3] := inv(Xs_asymm[1])[-1; 1] * mps_asymm.AC[1][1 -2; 2] * Xs_asymm[2][2; -3]
@tensor check12[-1 -2; -3] := mps_asymm.AC[1][-1 1; -3] * (2*im)*S_z_asymm[-2; 1]
@tensor check21[-1 -2; -3] := inv(Xs_asymm[2])[-1; 1] * mps_asymm.AC[2][1 -2; 2] * Xs_asymm[1][2; -3]
@tensor check22[-1 -2; -3] := mps_asymm.AC[2][-1 1; -3] * (2*im)*S_z_asymm[-2; 1]
@assert norm(check11-check12) < 1e-10
@assert norm(check21-check22) < 1e-10

println("Started with calculating the correlation matrix")

term_mass1 = mass * @tensor Xs_asymm[1][1; 2] * mps_asymm.AC[1][2 3; 8] * S⁺[4; 3] * mass_term[5; 4] * S⁻[6; 5] * conj(Xs_asymm[1][1; 7]) * conj(mps_asymm.AC[1][7 6; 8])
term_mass2 = mass * @tensor Xs_asymm[2][1; 2] * mps_asymm.AC[2][2 3; 8] * S⁺[4; 3] * mass_term[5; 4] * S⁻[6; 5] * conj(Xs_asymm[2][1; 7]) * conj(mps_asymm.AC[2][7 6; 8])
term_hop_left1 = @tensor mps_asymm.AL[2][1 2; 3] * Xs_asymm[1][3; 4] * mps_asymm.AC[1][4 5; 12] * S⁺[6; 5] * hopping[7 9; 2 6] * S⁻[11; 9] * conj(Xs_asymm[1][8; 10]) * conj(mps_asymm.AL[2][1 7; 8]) * conj(mps_asymm.AC[1][10 11; 12])
term_hop_right1 = @tensor Xs_asymm[1][1; 2] * mps_asymm.AC[1][2 3; 4] * mps_asymm.AR[2][4 6; 12] * S⁺[5; 3] * hopping[7 11; 5 6] * S⁻[8; 7] * conj(Xs_asymm[1][1; 9]) * conj(mps_asymm.AC[1][9 8; 10]) * conj(mps_asymm.AR[2][10 11; 12])

term_hop_left2 = @tensor mps_asymm.AL[1][1 2; 3] * Xs_asymm[2][3; 4] * mps_asymm.AC[2][4 5; 12] * S⁺[6; 5] * hopping[7 9; 2 6] * S⁻[11; 9] * conj(Xs_asymm[2][8; 10]) * conj(mps_asymm.AL[1][1 7; 8]) * conj(mps_asymm.AC[2][10 11; 12])
term_hop_right2 = @tensor Xs_asymm[2][1; 2] * mps_asymm.AC[2][2 3; 4] * mps_asymm.AR[1][4 6; 12] * S⁺[5; 3] * hopping[7 11; 5 6] * S⁻[8; 7] * conj(Xs_asymm[2][1; 9]) * conj(mps_asymm.AC[2][9 8; 10]) * conj(mps_asymm.AR[1][10 11; 12])
