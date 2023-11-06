using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_groundstate_energy.jl")
include("get_thirring_hamiltonian_symmetric.jl")
include("get_thirring_hamiltonian.jl")

am_tilde_0 = 0.5
Delta_g = 0.0
v = 0.0
D = 20

#

println("Really started")

#=
pspace = U1Space(i => 1 for i in (-1//2):1//2)
vspace_L = U1Space(1//2 => D, -1//2 => D, 3//2 => D, -3//2 => D)
vspace_R = U1Space(2 => D, 1 => D, 0 => D, -1 => D, -2 => D)
state = InfiniteMPS([pspace, pspace], [vspace_L, vspace_R])

hamiltonian = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v)

(groundstate,envs,_) = find_groundstate(state,hamiltonian,VUMPS(maxiter = 25));
=#

number_of_loops = 4
final_iterations = 30
D_start = 3
trunc = 3.0
tolerance = 6.0

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
vspace_L = U1Space(1//2 => D_start, -1//2 => D_start, 3//2 => D_start, -3//2 => D_start)
vspace_R = U1Space(2 => D_start, 1 => D_start, 0 => D_start, -1 => D_start, -2 => D_start)
mps = InfiniteMPS([pspace, pspace], [vspace_L, vspace_R])

hamiltonian = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, v)


for i in 1:number_of_loops 
    global mps
    (mps,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = 10, tol_galerkin = 10^(-tolerance)))
    (mps,envs) = changebonds(mps, hamiltonian, VUMPSSvdCut(trscheme = truncbelow(10^(-trunc))), envs)
end 

(groundstate,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = final_iterations, tol_galerkin = 1e-12))

tot_bonddim = dims((groundstate.AL[1]).codom)[1] + dims((groundstate.AL[1]).dom)[1]
print(tot_bonddim)

println("Done with VUMPS")

gs_energy = expectation_value(groundstate, hamiltonian);

k_values = LinRange(-pi/2,pi/2,17)



(energies,Bs) = excitations(hamiltonian,QuasiparticleAnsatz(), k_values,groundstate,envs);

print("Done with excitations")

@save "2023_10_24_dispersion_relation_small_k_values_m_0_v_0p5_symmetric_better" gs_energy k_values energies Bs