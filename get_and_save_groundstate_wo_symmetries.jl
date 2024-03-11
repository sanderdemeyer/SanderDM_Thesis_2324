using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit

include("get_thirring_hamiltonian.jl")

mass = 0.3
v = 0.0
Delta_g = 0.0

trunc = 4.0
tolerance = trunc+3.0
number_of_loops = 2
iterations = [50 100]

D = 5
mps = InfiniteMPS([ℂ^2, ℂ^2],[ℂ^D, ℂ^D])
hamiltonian = get_thirring_hamiltonian(mass, Delta_g, v)

for _ in 1:number_of_loops
    global mps
    global envs
    (mps,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = iterations[1], tol_galerkin = 10^(-tolerance)))
    (mps,envs) = changebonds(mps, hamiltonian, OptimalExpand(trscheme = truncbelow(10^(-trunc))), envs)
    #(mps,envs) = changebonds(mps, hamiltonian, VUMPSSvdCut(trscheme = truncbelow(10^(-trunc))), envs)
end 

(mps,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = iterations[2], tol_galerkin = 1e-12))

@save "SanderDM_Thesis_2324/gs_mps_wo_symmetries_trunc_$(trunc)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)" mps envs

N = 2
tot_bonddim = 0
for i = 1:N
    global tot_bonddim
    tot_bonddim += dims((mps.AL[i]).codom)[1] + dims((mps.AL[i]).dom)[1]
end
println("Bond dimension = $(tot_bonddim)")