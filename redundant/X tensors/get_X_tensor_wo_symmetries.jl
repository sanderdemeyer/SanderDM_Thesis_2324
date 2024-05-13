using LinearAlgebra
# using Base
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics

function get_X_tensors(tensors)
    # tensors is for example mps.AL, mps.AC or mps.AR
    # this assumes the period of the unit cell to be 2.

    w = length(mps)
    spaces = [codomain(tensors[site])[1] for site = 1:w]

    data = [Dict(i => (-1)^i.charge * (1+im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = V1.dims) for site = 1:w]

    data = []
    for site = 1:w
        if (site % 2 == 1)
            push!(data, Dict(i => (-1)^i.charge * (1+im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = spaces[site].dims))
        else
            push!(data, Dict(i => (-1)^((numerator(i.charge)+1)/2) * (1-im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = spaces[site].dims))
        end
    end
    X_tensors = [TensorMap(data[site], spaces[site], spaces[site]) for site = 1:w]
    println("spaces are $(spaces[1])")
    println("spaces are $(spaces[2])")
    println("done")
    return X_tensors
    # data_X1 = Dict(i => (-1)^i.charge * (1+im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = V1.dims)
    # data_X2 = Dict(i => (-1)^((numerator(i.charge)+1)/2) * (1-im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = V2.dims)    
end

trunc = 4.0
mass = 0.3
v = 0.0
Delta_g = 0.0

@load "SanderDM_Thesis_2324/gs_mps_wo_symmetries_trunc_$(trunc)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)" mps envs

# Find the X tensor
S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], ℂ^2, ℂ^2)
O = (2)*S_z_symm

H = @mpoham sum((S_z_symm){i} for i in vertices(InfiniteChain(2)))


AL1 = mps.AL[1]
@tensor B[-1 -2; -3] := AL1[-1 1; -3] * O[-2; 1]


V1 = codomain(AL1)[1]
V2 = domain(AL1)[1]


TensorMap([1.0 0.0; 0.0 1.0], ℂ^2, ℂ^2)

X1 = TensorMap([(i==j)*(-1)^i *im for i in 1:12, j in 1:12], ℂ^12, ℂ^12)


@tensor B2[-1 -2; -3] := X1[-1; 1] * AL1[1 -2; 2] * adjoint(X1)[2 ; -3]
@tensor B1[-1 -2; -3] := AL1[-1 1; -3] * O[-2 ; 1]

break

# X1[U1Space(0 => 3)] = Matrix{Float64}(I,3,3)

# data_X2 = Dict(i => (-1)^((numerator(i.charge)+1)/2).charge * im * Matrix{Float64}(I,3,3) for i = keys(V2.dims))

data_X1 = Dict(i => (-1)^i.charge * (1+im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = V1.dims)
data_X2 = Dict(i => (-1)^((numerator(i.charge)+1)/2) * (1-im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = V2.dims)
X1 = TensorMap(data_X1, V1, V1)
X2 = TensorMap(data_X2, V2, V2)

(X1, X2) = get_X_tensors(mps.AL)

@tensor B2[-1 -2; -3] := X1[-1; 1] * AL1[1 -2; 2] * adjoint(X2)[2 ; -3]
@tensor B1[-1 -2; -3] := AL1[-1 1; -3] * O[-2 ; 1]


@tensor B3[-1 -2; -3] := X2[-1; 1] * mps.AL[2][1 -2; 2] * adjoint(X1)[2 ; -3]
@tensor B4[-1 -2; -3] := mps.AL[2][-1 1; -3] * O[-2 ; 1]

break
data = Dict(U1Irrep(0) => Matrix{Float64}(I,3,3), U1Irrep(1) => Matrix{Float64}(I,3,3), U1Irrep(-1) => Matrix{Float64}(I,3,3), U1Irrep(-2) => Matrix{Float64}(I,3,3), U1Irrep(2) => Matrix{Float64}(I,3,3))
TensorMap(data, V1, V1)


FusionTree(dom, cod)

function block_data(charge::Int)
    data = Dict(0 => Matrix{Float64}(I,3,3), 1 => Matrix{Float64}(I,3,3), -1 => Matrix{Float64}(I,3,3), -2 => Matrix{Float64}(I,3,3), 2 => Matrix{Float64}(I,3,3))
    data = Dict(U1Irrep(0) => Matrix{Float64}(I,3,3), U1Irrep(1) => Matrix{Float64}(I,3,3), U1Irrep(-1) => Matrix{Float64}(I,3,3), U1Irrep(-2) => Matrix{Float64}(I,3,3), U1Irrep(2) => Matrix{Float64}(I,3,3))
    data = [Matrix{Float64}(I,3,3); Matrix{Float64}(I,3,3); Matrix{Float64}(I,3,3); Matrix{Float64}(I,3,3); Matrix{Float64}(I,3,3)]
end

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