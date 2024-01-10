using LinearAlgebra
# using Base
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics

# Find the X tensor
spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)
H = @mpoham sum((S_z_symm){i} for i in vertices(InfiniteChain(2)))

AL1 = gs_mps.AC[1]

# @tensor T[-1 -3; -2 -4] := AL1[-1 1; -3] * S_z_symm[2; 1] * conj(AL1[-2; 2 -4])

# println(blocks(T).keys)

# D, V = eig(T)

# println(typeof(D))
# println(typeof(V))
# println(typeof(eigh(T)))
# println(norm(eigh(T)[2]))
# # for i in blocks(Ψ.AC[1]).keys
# println(typeof(TensorKit.eigen(T)))
# # println(typeof(eigvals(T)))
# # println(typeof(eigvecs(T)))

# algs = [VUMPS(; tol_galerkin=1e-5, verbose=false), GradientGrassmann(; verbosity=0)]
# alg = VUMPS(; tol_galerkin=1e-5, verbose=false)
# leading_boundary(gs_mps, H, alg)


# println(dims(T))
# Q, Λ = eigh(T)
# println(typeof(Q))
# println(typeof(Λ))

# println(Λ)

envs = environments(gs_mps, H)
AC1 = gs_mps.AC[1]
AR2 = gs_mps.AR[2]
AR1 = gs_mps.AR[1]
ρl = envs.lw[1]
ρl2 = envs.lw[2]
ρr1 = envs.rw[1]
ρr2 = envs.rw[2]
ρl = ρl/norm(ρl)



@tensor triv[-1; -2] := AC1[1 2; -1] * conj(AC1[1 2; -2])
@tensor triv2[-1; -2] := AR2[1 3; -1] * conj(AR2[2 3; -2]) * triv[1; 2]
@tensor triv3[-1; -2] := AR1[1 3; -1] * conj(AR1[2 3; -2]) * triv2[1; 2]

@tensor a[-1] := ρl[1 -1; 2] * triv2[1; 2]
@tensor b = ρl[1 3; 2] * conj(ρl[1 3; 2])
@tensor c[-1; -2] := ρl[1 -1; 2] * ρl[2 -2; 1]

@tensor ρl_new[-2 -3; -1] := ρl[2 -3; 1] * AR1[1 3; -1] * conj(AR1[2 3; -2])
@tensor ρl_new2[-2 -3; -1] := ρl_new[2 -3; 1] * AR2[1 3; -1] * conj(AR2[2 3; -2])

@tensor mid[-2 -3; -1] := ρl[2 -3; 1] * AR1[1 4; -1] * conj(AR1[2 3; -2]) * S_z_symm[3; 4]
@tensor mid2[-2 -3; -1] := mid[2 -3; 1] * AR2[1 4; -1] * conj(AR2[2 3; -2]) * S_z_symm[3; 4]

println(norm(triv))
println(norm(triv2))
println(norm(triv3))


println(norm(a))
println(norm(b))
println(norm(c))

println(norm(ρl_new2-ρl))
println(norm(mid2-ρl))

println(norm(mid2-ρl2))
println(norm(triv-triv3))
# @tensor rho_new[-1; -2] := ρl[3; 1] * AC1[1 2; -1] * conj(AC1[3 2; -2])