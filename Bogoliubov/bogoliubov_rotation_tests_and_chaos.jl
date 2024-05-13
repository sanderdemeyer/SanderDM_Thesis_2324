using LinearAlgebra
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Base
using Plots

include("get_thirring_hamiltonian.jl")
include("get_thirring_hamiltonian_symmetric.jl")

mass = 0.3
v = 0.0
Delta_g = 0.0
truncation = 3.0

@load "SanderDM_Thesis_2324/gs_mps_trunc_$(truncation)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)" mps
@load "SanderDM_Thesis_2324/Dispersion_Delta_m_$(mass)_delta_g_$(Delta_g)_v_$(v)_trunc_$(truncation)_all_sectors" gs_energy bounds_k energies Bs

k_values = LinRange(-bounds_k, bounds_k,length(Bs))


spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
trivspace = U1Space(0 => 1)
middle = (-im) * TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im -1.0+0.0im], pspace, pspace)
# middle = (-im) * TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im -1.0+0.0im], trivspace ⊗ pspace, trivspace ⊗ pspace)
H_middle = @mpoham sum(middle{i} for i in vertices(InfiniteChain(2)))
unit_O = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], pspace, pspace)
unit = @mpoham sum(unit_O{i} for i in vertices(InfiniteChain(2)))

h = get_thirring_hamiltonian_symmetric(mass, Delta_g, v)
H_middle = h
i = 1

@tensor T[-1 -2; -3 -4] := mps.AR[i][-1 1; -3] * middle[2 1] * conj(mps.AR[i][-2 2; -4])

@tensor T1[-1 -2; -3 -4] := mps.AR[1][-1 1; 2] * mps.AR[2][2 3; -3] * middle[4; 1] * middle[6; 3] * conj(mps.AR[1][-2 4; 5]) * conj(mps.AR[2][5 6; -4])
@tensor T2[-1 -2; -3 -4] := mps.AR[2][-1 1; 2] * mps.AR[3][2 3; -3] * middle[4; 1] * middle[6; 3] * conj(mps.AR[2][-2 4; 5]) * conj(mps.AR[3][5 6; -4])

T1inv = pinv(T1)
T2inv = pinv(T2)

envs = environments(mps, H_middle)

envs = environments(mps, H_middle)
left_env = leftenv(envs, i, mps)
right_env = rightenv(envs, i, mps)


# TEST leftenv
AR = mps.AR
AC = mps.AC
AL = mps.AL

envs = environments(mps, H_middle)

left_env1 = leftenv(envs, i, mps)
left_env2 = leftenv(envs, i+1, mps)
U_space1 = Tensor(ones, space(left_env1[1])[2])
U_space2 = Tensor(ones, space(left_env2[1])[2])
U_space3 = Tensor(ones, space(right_env1[2])[2])
U_space4 = Tensor(ones, space(right_env2[2])[2])

break
#always choose left_env?[1] to target ending string of Sz tensors


result1 = @tensor AL[i][1 2; 4] * AC[i+1][4 5; 6] * middle[8; 2] * middle[10; 5] * conj(AL[i][7 8; 9]) * conj(AC[i+1][9 10; 6]) * conj(U_space1[3]) * left_env1[1][7 3; 1]
result2 = @tensor AC[i+1][1 2; 4] * middle[6; 2] * conj(AC[i+1][5 6; 4]) * left_env2[1][5 3; 1] * conj(U_space2[3])

i = i + 1

result3 = @tensor AL[i][1 2; 4] * AC[i+1][4 5; 6] * middle[8; 2] * middle[10; 5] * conj(AL[i][7 8; 9]) * conj(AC[i+1][9 10; 6]) * conj(U_space1[3]) * left_env2[1][7 3; 1]
result4 = @tensor AC[i+1][1 2; 4] * middle[6; 2] * conj(AC[i+1][5 6; 4]) * left_env1[1][5 3; 1] * conj(U_space2[3])

break
#with right_env


i = 1

right_env1 = rightenv(envs, i, mps)
right_env2 = rightenv(envs, i+1, mps)

U_space3 = Tensor(ones, space(right_env1[2])[2])
U_space4 = Tensor(ones, space(right_env2[2])[2])

result1 = 0.0
result2 = 0.0

for l = 1:6
    for a = 1:6
        for b = 1:6
            value = @tensor AL[i][1 2; 4] * AC[i+1][4 5; 6] * H_middle[i][l,a][3 8; 2 11] * H_middle[i+1][a,b][11 10; 5 13] * conj(AL[i][7 8; 9]) * conj(AC[i+1][9 10; 12]) * left_env1[l][7 3; 1] * right_env2[b][6 13; 12]
            println("($(l),$(a),$(b))")
            println(value)
            result1 += value
        end
    end
end

for a = 1:6
    for b = 1:6
        value = @tensor AC[i+1][1 2; 4] * H_middle[i+1][a,b][3 6; 2 7] * conj(AC[i+1][5 6; 8]) * left_env2[a][5 3; 1] * right_env2[b][4 7; 8]
        println("($(a),$(b))")
        println(value)
        result2 += value
    end
end

i = 2

result3 = @tensor AL[i][1 2; 4] * AC[i+1][4 5; 6] * middle[8; 2] * middle[10; 5] * conj(AL[i][7 8; 9]) * conj(AC[i+1][9 10; 6]) * conj(U_space1[3]) * left_env2[1][7 3; 1]
result4 = @tensor AC[i+1][1 2; 4] * middle[6; 2] * conj(AC[i+1][5 6; 4]) * left_env1[1][5 3; 1] * conj(U_space2[3])

break

result1 = @tensor AL[i][1 2; 4] * AC[i+1][4 5; 6] * middle[8; 2] * middle[10; 5] * conj(AL[i][7 8; 9]) * conj(AC[i+1][9 10; 6]) * conj(U_space1[3]) * left_env1[1][7 3; 1]
result2 = @tensor AL[i][1 2; 4] * AC[i+1][4 5; 6] * middle[8; 2] * middle[10; 5] * conj(AL[i][7 8; 9]) * conj(AC[i+1][9 10; 6]) * conj(U_space1[3]) * left_env1[2][7 3; 1]

result3 = @tensor AC[i+1][1 2; 4] * middle[6; 2] * conj(AC[i+1][5 6; 4]) * left_env2[1][5 3; 1] * conj(U_space2[3])
result4 = @tensor AC[i+1][1 2; 4] * middle[6; 2] * conj(AC[i+1][5 6; 4]) * left_env2[2][5 3; 1] * conj(U_space2[3])

i = i + 1

result5 = @tensor AL[i][1 2; 4] * AC[i+1][4 5; 6] * middle[8; 2] * middle[10; 5] * conj(AL[i][7 8; 9]) * conj(AC[i+1][9 10; 6]) * conj(U_space1[3]) * left_env2[1][7 3; 1]
result6 = @tensor AL[i][1 2; 4] * AC[i+1][4 5; 6] * middle[8; 2] * middle[10; 5] * conj(AL[i][7 8; 9]) * conj(AC[i+1][9 10; 6]) * conj(U_space1[3]) * left_env2[2][7 3; 1]

result7 = @tensor AC[i+1][1 2; 4] * middle[6; 2] * conj(AC[i+1][5 6; 4]) * left_env1[1][5 3; 1] * conj(U_space2[3])
result8 = @tensor AC[i+1][1 2; 4] * middle[6; 2] * conj(AC[i+1][5 6; 4]) * left_env1[2][5 3; 1] * conj(U_space2[3])


break




break

@tensor T_left[-1 -2; -3 -4] := AL[i-1][-1 1 3] * AL[i][3 4; -3] * middle[2; 1] * middle[5; 4] * conj(AL[i-1][-2 2; 6]) * conj(AL[i][6 5; -4])
eig = eigen(T_left)

break

test1 = @tensor AL[i-1][1 2; 3] * AC[i][3 4; 5] * conj(AL[i-1][1 2; 6]) * conj(AC[i][6 4; 5])
test2 = @tensor AC[i][1 2; 3] * conj(AC[i][1 2; 3])


test3 = @tensor AL[i-1][1 2; 3] * AC[i][3 4; 6] * conj(AL[i-1][1 2; 7]) * conj(AC[i][7 5; 6]) * middle[5; 4]
test4 = @tensor AC[i][1 2; 4] * conj(AC[i][1 3; 4]) * middle[3; 2]

break

U_space = Tensor(ones, ℂ^1)
B_tensors = []
for iₖ = 1:length(Bs)
    values = []
    VL = Bs[iₖ].VLs
    Xs = Bs[iₖ].Xs
    for a in keys(VL)
        @tensor new[-1 -2; -3] := VL[1][-1 -2; 1] * Xs[1][1 2; -3] * conj(U_space[2])
        push!(values, new)
        # B_base = VL[a] * Xs[a]
        # @tensor B_new[-1 -2; -3] := B_base[-1 -2; 1 -3] * conj(U_space[1])
        # push!(values, VL[a] * Xs[a]) # based on src/states/quasiparticle_state.jl line 12
        # push!(values, B_new) # based on src/states/quasiparticle_state.jl line 12
    end
    value = values[1]
    for i = 2:length(values)
        value += values[i]
    end
    push!(B_tensors, value)
end

Sz_plus_12 = S_z() + 0.5*id(domain(S_z()))
S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
Szₘ = TensorMap((-2*im)*[0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], ℂ^2, ℂ^2)
unit = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], ℂ^2, ℂ^2)
# S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^1⊗ℂ^2, ℂ^2⊗ℂ^1)
# S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], ℂ^1⊗ℂ^2, ℂ^2⊗ℂ^1)
# Szₘ = (-2*im)*S_z()
HSzₘ = @mpoham sum((Szₘ){i} for i in vertices(InfiniteChain(2)))
HS⁺ = @mpoham sum((S⁺){i} for i in vertices(InfiniteChain(2)))
Hunit = @mpoham sum((unit){i} for i in vertices(InfiniteChain(2)))
# Szₘ = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], ℂ^1⊗ℂ^2, ℂ^2⊗ℂ^1)
H = get_thirring_hamiltonian(mass, Delta_g, v)

J = expectation_value(mps, H)
H_ev = @mpoham sum((J[i]*unit){i} for i in vertices(InfiniteChain(2)))
H_con = H-H_ev

S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^1⊗ℂ^2, ℂ^2⊗ℂ^1)
Szₘ = TensorMap((-2*im)*[0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], ℂ^1⊗ℂ^2, ℂ^2⊗ℂ^1)

Os = [Szₘ S⁺; 0 unit]

SparseMPO(Os, [ℂ^1 ℂ^1 ℂ^1 ; ℂ^1 ℂ^1 ℂ^1], [ℂ^2 ℂ^2 ℂ^2])
SparseMPOSlice(Os, [ℂ^1 ℂ^1 ℂ^1], [ℂ^1 ℂ^1 ℂ^1], [ℂ^2 ℂ^2 ℂ^2])

#=
for i = 1:length(energies)
    E1 = @tensor mps.AC[1][1 2; 3] * conj(B_tensors[i][1 2; 3])
    E2 = @tensor mps.AC[2][1 2; 3] * conj(B_tensors[i][1 2; 3])
    E3 = @tensor mps.AL[1][1 2; 3] * conj(B_tensors[i][1 2; 3])
    E4 = @tensor mps.AL[2][1 2; 3] * conj(B_tensors[i][1 2; 3])
    E5 = @tensor B_tensors[i][1 2; 3] * conj(B_tensors[i][1 2; 3])
    E6 = @tensor B_tensors[i][1 2; 3] * conj(B_tensors[i][1 2; 3])
    envs = environments(mps, H)
    left_env = leftenv(envs, 1, mps)
    left = left_env * TransferMatrix(B_tensors[i], H[1], B_tensors[i])
    value = 0.0
    for a in keys(left)
        # global value
        value += @tensor left[a][3 2; 1] * rightenv(envs, 1, mps)[a][1 2; 3]
    end
    left_env = leftenv(envs, 2, mps)
    left = left_env * TransferMatrix(B_tensors[i], H[2], B_tensors[i])
    value2 = 0.0
    for a in keys(left)
        # global value
        value2 += @tensor left[a][3 2; 1] * rightenv(envs, 2, mps)[a][1 2; 3]
    end
    println("E from qp is $(energies[i]), own calculation gives")
    # println("value is $(value), value2 is $(value2)")
    println("there sum is $(value+value2)")
    # println(E1)
    # println(E2)
    # println(E3)
    # println(E4)
    println(E5)
    # println("Difference is $(energies[i]-E5)")
    println("--------------------------")
end

break
=#

break

# trying to calculate the energies

nodp = 50
Es = zeros(ComplexF64, nodp, nodp)

H_con = Hunit

envs = environments(mps, H_con)

iₖ = 18

for i = 1:nodp
    println("i = $(i)")
    # i = div(nodp,2)+iᵢ
    for j = 1:nodp
        if j > i
            left_env = leftenv(envs, i, mps)
            transf = TransferMatrix(B_tensors[iₖ], H_con[i], mps.AL[i])
            left = left_env * transf
            for a = i+1:j-1
                # global left
                left = left * TransferMatrix(mps.AR[a], H_con[a], mps.AL[a])
            end
            left = left * TransferMatrix(mps.AR[j], H_con[j], B_tensors[iₖ])
            value = 0.0
            for a in keys(left)
                # global value
                value += @tensor left[a][3 2; 1] * rightenv(envs, j, mps)[a][1 2; 3]
            end
        end
        if j == i
            left_env = leftenv(envs, i, mps)
            left = left_env * TransferMatrix(B_tensors[iₖ], H_con[i], B_tensors[iₖ])
            value = 0.0
            for a in keys(left)
                # global value
                value += @tensor left[a][3 2; 1] * rightenv(envs, i, mps)[a][1 2; 3]
            end
        end
        if j < i
            left_env = leftenv(envs, j, mps)
            left = left_env * TransferMatrix(mps.AL[j], H_con[j], B_tensors[iₖ])
            for a = j+1:i-1
                # global left
                left = left * TransferMatrix(mps.AL[a], H_con[a], mps.AR[a])
            end
            left = left * TransferMatrix(B_tensors[iₖ], H_con[i], mps.AR[i])
            value = 0.0
            for a in keys(left)
                # global value
                value += @tensor left[a][3 2; 1] * rightenv(envs, j, mps)[a][1 2; 3]
            end
        end
        Es[i,j] = value*exp(im*k_values[iₖ]*(i-j))
    end
end
E_final = sum(Es[div(nodp,2),:])+sum(Es[div(nodp,2)+1,:])
println(sum(Es[div(nodp,2),:])+sum(Es[div(nodp,2)+1,:]))


break

# i is the site of S⁺, j is the site of B
iₖ = 15
i = 1
j = 5
H = HSzₘ
envs = environments(mps, H)
left_env = leftenv(envs, i, mps)
transf = TransferMatrix(mps.AC[i], H[i], mps.AC[i])
left_env * transf
tra_S⁺ₗ = TransferMatrix(mps.AC[i], HS⁺[i], mps.AC[i])
tra_Sz = TransferMatrix(mps.AC[i], HSzₘ[i], mps.AC[i])
# env_current = left_env_Z * tra_S⁺ₗ
# env_current = left_env_Z * tra_Sz

nodp = 100
overlap = zeros(ComplexF64, 2, nodp)

for iᵢ = 0:1
    i = div(nodp,2)+iᵢ
    for j = 1:nodp
        println("j = $(j)")
        if j > i
            envs = environments(mps, HSzₘ)
            left_env = leftenv(envs, i, mps)
            tra_S⁺ = TransferMatrix(mps.AC[i], HS⁺[i], mps.AC[i])
            left = left_env * tra_S⁺
            for a = i+1:j-1
                # global left
                left = left * TransferMatrix(mps.AR[a], Hunit[a], mps.AR[a])
            end
            left = left * TransferMatrix(mps.AR[j], Hunit[j], B_tensors[iₖ])
            envs = environments(mps, Hunit)
            value = 0.0
            for a in keys(left)
                # global value
                value += @tensor left[a][3 2; 1] * rightenv(envs, j, mps)[a][1 2; 3]
            end
        end
        if j == i
            envs = environments(mps, HSzₘ)
            left_env = leftenv(envs, i, mps)
            left = left_env * TransferMatrix(mps.AC[i], HS⁺[i], B_tensors[iₖ])
            envs = environments(mps, Hunit)
            value = 0.0
            for a in keys(left)
                # global value
                value += @tensor left[a][3 2; 1] * rightenv(envs, j, mps)[a][1 2; 3]
            end
        end
        if j < i
            envs = environments(mps, HSzₘ)
            left_env = leftenv(envs, j, mps)
            left = left_env * TransferMatrix(mps.AC[j], HSzₘ[j], B_tensors[iₖ])
            for a = j+1:i-1
                # global left
                left = left * TransferMatrix(mps.AR[a], HSzₘ[a], mps.AR[a])
            end
            left = left * TransferMatrix(mps.AR[i], HS⁺[i], mps.AR[i])
            envs = environments(mps, Hunit)
            value = 0.0
            for a in keys(left)
                # global value
                value += @tensor left[a][3 2; 1] * rightenv(envs, j, mps)[a][1 2; 3]
            end
        end
        overlap[iᵢ+1,j] = value
    end
end

E = [real(overlap[1,i]) for i = 1:nodp]
plt=plot(1:nodp, E)
display(plt)

E = [real(overlap[2,i]) for i = 1:nodp]
plt=plot(1:nodp, E)
display(plt)


break

state = mps
envs = environments(state, H)
left_env = leftenv(envs, i, state)
right_env = rightenv(envs, i, state)

transf = TransferMatrix(mps.AC[i], H[i], mps.AC[i])
transf2 = TransferMatrix(mps.AC[i-1], H[i-1], mps.AC[i-1])

left_env * (transf * right_env)


break
left_env_Z = leftenv(envs_Z, i, mps)

tra_H = TransferMatrix(mps.AC[i], H[i], mps.AR[i])
tra_Sz = TransferMatrix(mps.AC[i], Szₘ, mps.AR[i])

tra = TransferMatrix(mps.AC[i], mps.AR[i])
tra_S⁺ₗ = TransferMatrix(mps.AL[i], S⁺, mps.AL[i])

