using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit

include("get_groundstate.jl")

trunc = 1.5
mass = 0.3
v = 0.0
Delta_g = 0.0

(mps, envs) = get_groundstate(mass, Delta_g, v, [50 100], trunc, 1e-10; number_of_loops=7)

@save "SanderDM_Thesis_2324/gs_mps_trunc_$(trunc)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)" mps envs
H0 = get_thirring_hamiltonian_symmetric(mass, Delta_g, v)

N = 2
tot_bonddim = 0
for i = 1:N
    global tot_bonddim
    tot_bonddim += dims((mps.AL[i]).codom)[1] + dims((mps.AL[i]).dom)[1]
end
println("Bond dimension = $(tot_bonddim)")