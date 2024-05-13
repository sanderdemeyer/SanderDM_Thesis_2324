using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using LaTeXStrings

include("get_thirring_hamiltonian_symmetric.jl")

@load "SanderDM_Thesis_2324/gs_mps_trunc_2.5_mass_0.0_v_0.0_Delta_g_0.0" mps

m = 0.0
delta_g = 0.0
v = 0.0

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)
H = @mpoham sum((S_z_symm){i} for i in vertices(InfiniteChain(2)))
Plus_space = U1Space(1 => 1)
Triv_space = U1Space(0 => 1)
S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Triv_space ⊗ pspace, pspace ⊗ Plus_space)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace ⊗ Triv_space)
# S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], Triv_space ⊗ pspace, pspace ⊗ Triv_space)

H = get_thirring_hamiltonian_symmetric(m, delta_g, v)

state = mps
envs = environments(state, H)
left_env = leftenv(envs, i, state)
right_env = rightenv(envs, i, state)

transf = TransferMatrix(mps.AC[i], H[i], mps.AC[i])
transf2 = TransferMatrix(mps.AC[i-1], H[i-1], mps.AC[i-1])

right = (transf * right_env)
left_env * transf
for (j,k) in keys(H[i])
    V = @tensor right[j][1 2; 3] * left_env[j][3 2; 1]
end
break

println(typeof(left_env)) # = PeriodicArray, 2, 1
println(typeof(right_env)) # PeriodicArray, 2, 1
println(typeof(transf)) # SingleTransferMatrix, 2, 1
println(typeof(left_env * transf)) # PeriodicArray, 2, 1
println(typeof(transf * right_env)) # PeriodicArray, 2, 1

test_right = transf2 * (transf * right_env)

for (j,k) in keys(H[i])
    println("j = $(j), k = $(k)")
    V = @tensor left_env[j][1 2; 3] * test_right[j][3 2; 1]
end


for (j,k) in keys(H[i])
    left_env = leftenv(envs, i, state)
    right_env = rightenv(envs, i+1, state)
    V = @tensor left_env[j][1 2; 3] * (TransferMatrix(mps.AC[i], H[i], mps.AC[i]) * TransferMatrix(mps.AC[i+1], H[i+1], mps.AC[i+1]) * right_env)[j][3 2; 1]
end

left_env = leftenv(envs, i, state)
tra = TransferMatrix(mps.AC[i], H[i], mps.AC[i])

# @tensor W[-1 -2; -3] := left_env[1 2; 3] * tra[1 2 3; -1 -2 -3] 

left_env = leftenv(envs, i, state)

@tensor new_left[-1 -2; -3] := left_env[5 3; 1] * mps.AC[i][1 2; -3] * H[i][3 4; 2 -2] * conj(mps.AC[i][5 4; -1])


for (j,k) in keys(H[i])
    tra = TransferMatrix(mps.AC[i+1], H[i+1], mps.AC[i+1])
    @tensor new_left[-1 -2; -3] := left_env[j][5 3; 1] * mps.AC[i][1 2; -3] * H[i][j,k][3 4; 2 -2] * conj(mps.AC[i][5 4; -1])
    new_left * tra
end

O₁ = S⁺
U_space₁ = Tensor(ones, space(O₁)[1])

@plansor AO[-1 -2; -3 -4] := mps.AC[i][-1 1; -4] * O₁[2 -2; 1 -3] * conj(U_space₁[2])

tra = TransferMatrix(mps.AC[i+1], H[i+1], mps.AC[i+1])

for (j,k) in keys(H[i])
    for (a,b) in keys(H[i+1])
        if (k == a)
            println("yes")
            @tensor new[-1 -2 -3; -4 -5 -6] :=  mps.AC[i][-3 1; 2] * H[i][j,k][-2 5; 1 3] * conj(mps.AC[i][-1 5; 6]) * tra.above[2 4; -4] * tra.middle[a,b][3 7; 4 -5] * conj(tra.below[6 7; -6])
        end
    end
end

sites = 2

O₁ = S⁺
O₂ = S⁻
U_space₁ = Tensor(ones, space(O₁)[1])
U_space₂ = Tensor(ones, space(O₂)[4])

left_env = leftenv(envs, i, state)
right_env = rightenv(envs, i+sites, state)
tra = TransferMatrix(mps.AR[i+1], H[i+1], mps.AR[i+1])

for (j₁, k₁) in keys(H[i])
    for (j₂, k₂) in keys(tra.middle)
        if (k₁ == j₂)
            for (j₃, k₃) in keys(H[i+sites])
                if (k₂ == j₃)
                    println("jow")
                    G = @tensor left_env[j₁][6 5; 1] * mps.AC[i][1 2; 8] * O₁[3 4; 2 18] * conj(U_space₁[3]) * H[i][j₁,k₁][5 7; 4 10] * conj(mps.AC[i][6 7; 11]) * tra.above[8 9; 13] * tra.middle[j₂,k₂][10 12; 9 15] * conj(tra.below[11 12; 19]) * mps.AR[i+sites][13 14; 23] * H[i+sites][j₃,k₃][15 16; 14 22] * O₂[18 20; 16 17] * conj(U_space₂[17]) * conj(mps.AR[i+sites][19 20; 21]) * right_env[k₃][23 22; 21]
                end
            end
        end
    end
end


tra = TransferMatrix(mps.AC[i+1], H[i+1], mps.AC[i+1])
for (j,k) in keys(tra.middle)
    println("j = $(j), k = $(k)")
end

break

left_env = leftenv(envs, i, state)
AC_above[-1 -2; -3] := @tensor mps.AC[i][-1 1; -3] * S
tra = TransferMatrix(mps.AC[i], H[i], mps.AR[i])
