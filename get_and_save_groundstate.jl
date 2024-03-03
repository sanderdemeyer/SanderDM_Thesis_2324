using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit

include("get_groundstate.jl")

trunc = 2.5
mass = 0.3
v = 0.0
Delta_g = 0.0

(mps, envs) = get_groundstate(mass, Delta_g, v, [20 50], trunc, 1e-10)

@save "SanderDM_Thesis_2324/gs_mps_trunc_$(trunc)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)" mps