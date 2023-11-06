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

#include("get_groundstate_energy.jl")
#include("get_thirring_hamiltonian.jl")
include("get_thirring_hamiltonian_symmetric.jl")

function change_bonds_custom(iter, mps, H, envs)
    #return changebonds(mps, H, VUMPSSvdCut(tol_eigenval = 1e-5), envs)
    return changebonds(mps, hamiltonian, VUMPSSvdCut(trscheme = truncbelow(1e-5)), envs)
end

D = 6
spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
vspace_L = U1Space(1//2 => D, -1//2 => D, 3//2 => D, -3//2 => D)
vspace_R = U1Space(2 => D, 1 => D, 0 => D, -1 => D, -2 => D)
mps = InfiniteMPS([pspace, pspace], [vspace_L, vspace_R])


hamiltonian = get_thirring_hamiltonian_symmetric(0.5, 0.5, 0.2)

print("Before changebonds")
println((mps.AL[1]).dom)
println((mps.AL[1]).codom)
println("gelukt")
println(entanglement_spectrum(mps, 1))

(mps,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = 30, tol_galerkin = 1e-12, finalize = change_bonds_custom))



(mps,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = 20, tol_galerkin = 1e-12))
(mps,envs) = changebonds(mps, hamiltonian, VUMPSSvdCut(trscheme = truncbelow(1e-5)), envs)

print("After changebonds1")
println((mps.AL[1]).dom)
println((mps.AL[1]).codom)
println("gelukt")
println(entanglement_spectrum(mps, 1))

(mps,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = 20, tol_galerkin = 1e-12))
(mps,envs) = changebonds(mps, hamiltonian, VUMPSSvdCut(trscheme = truncbelow(1e-5)), envs)

print("After changebonds2")
println((mps.AL[1]).dom)
println((mps.AL[1]).codom)
println("gelukt")
println(entanglement_spectrum(mps, 1))

(mps,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = 20, tol_galerkin = 1e-12))
(mps,envs) = changebonds(mps, hamiltonian, VUMPSSvdCut(trscheme = truncbelow(1e-5)), envs)

print("After changebonds3")
println((mps.AL[1]).dom)
println((mps.AL[1]).codom)
println("gelukt")
println(entanglement_spectrum(mps, 1))

(mps,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = 20, tol_galerkin = 1e-12))
(mps,envs) = changebonds(mps, hamiltonian, VUMPSSvdCut(trscheme = truncbelow(1e-5)), envs)

print("After changebonds4")
println((mps.AL[1]).dom)
println((mps.AL[1]).codom)
println("gelukt")

println(entanglement_spectrum(mps, 1))


break

(mps,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = 10, tol_galerkin = 1e-7, finalize = change_bonds_custom))


print("After 2nd changebonds")
println((mps.AL[1]).dom)
println((mps.AL[1]).codom)
println("gelukt")


println(typeof(mps))

(mps,envs,_) = find_groundstate(mps,hamiltonian,VUMPS(maxiter = 10))


println(entanglement_spectrum(mps, 1))
InfiniteMps()

