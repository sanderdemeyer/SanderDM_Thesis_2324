using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_thirring_hamiltonian_symmetric.jl")

function correlator(state, O₁, O₂, middle, i::Int, N::Int)
    S₁ = (O₁).codom[1]
    S₂ = (O₂).dom[2]

    G = similar(1:N, scalartype(state))
    U = Tensor(ones, S₁)

    @tensor Vₗ[-1 -2; -3] := state.AC[i][3 4; -3] * conj(U[1]) * O₁[1 2; 4 -2] * conj(state.AC[i][3 2; -1])
    @tensor Vᵣ[-1 -2; -3] := state.AC[i][-1 2; 1] * O₂[-2 4; 2 3] * U[3] * conj(state.AC[i][-3 4; 1])

    for j = i-1:-1:1 # j < i ==> factor -i
        if j < i-1
            @tensor Vᵣ[-1 -2; -3] := (-2im) *Vᵣ[1 -2; 4] * (state.AL[j+1])[-1 2; 1] * middle[3; 2] * conj((state.AL[j+1])[-3 3; 4])
        end
        G[j] = 1im*(@tensor Vᵣ[4; 5 7] * state.AL[j][1 2; 4] * O₁[3 6; 2 5] * conj(U[3]) * conj(state.AL[j][1 6; 7]))
    end
    G[i] = @tensor (state.AC[i])[1 2; 8] * O₂[5 4; 2 3] * U[3] * O₁[6 7; 4 5] * conj(U[6]) * conj((state.AC[i])[1 7; 8])
    for j = i+1:N # j > i ==> factor i and conjugate
        if j > i+1
            @tensor Vₗ[-1 -2; -3] := (2im) * Vₗ[1 -2; 4] * (state.AR[j-1])[4 5; -3] * middle[3; 5] * conj((state.AR[j-1])[1 3; -1])
        end
        G[j] = -1im*(@tensor Vₗ[2; 3 5] * state.AR[j][5 6; 7] * O₂[3 4; 6 1] * U[1] * conj(state.AR[j][2 4; 7]))
    end
    return G
end

function final_value_old(left, right)
    final = 0.0
    # println("printing the keys")
    # for key in keys(left)
    #     println(key)
    # end
    # println("rigfht")
    # for key in keys(right)
    #     println(key)
    # end
    # println("done")
    # println("spaces for left")
    # for key in keys(left)
    #     println(key)
    #     println(left[key].codom)
    #     println(left[key].dom)
    # end
    # println("spaces for right")
    # for key in keys(left)
    #     println(key)
    #     println(right[key].codom)
    #     println(right[key].dom)
    # end
    # println(keys(left))
    for (l,r) = [(1,size(left)[1])] #keys(left)
        # for (l,r) = keys(left) #[(1,size(left)[1])] #keys(left)
        #     if (l,r) in keys(right)
        key = 1
        # println(key)
        # println(typeof(left))
        # println(typeof(right))
        # println(left[key].codom)
        # println(left[key].dom)
        # println(right[key].codom)
        # println(right[key].dom)
        @tensor A = left[l][1 2; 3 4] * right[r][4 3; 2 1]
        # @tensor A = left[key][1 2; 3 4] * right[key][4 3; 2 1]
        # println(A)
        # println("hooray")
        final += A
        # final += @tensor left[key][1 2 3 4] * right[key][4 3 2 1]
        # final += @tensor left[key][1 3 4 2] * right[key][2 3 4 1]
        # end
    end
    # println("all done")
    return final
end

function final_value(left, right)
    final = 0.0
    for (l,r) = [(1,size(left)[1])]
        key = 1
        @tensor A = left[l][1 2; 3 4] * right[r][4 3; 2 1]
        final += A
    end
    return final
end

function right_env_below(mps, H, j, envs, O₂)
    U2 = Tensor(ones, (O₂).dom[2])
    right_env = rightenv(envs, j, mps)
    tens_new = []
    for a = 1:size(H[j])[1]
        tensors = []
        for b = 1:size(H[j])[2]
            if (a,b) in keys(H[j])
                @tensor value[-1 -3; -2 -4] := mps.AR[j][-1 2; 1] * H[j][a,b][-2 4; 2 3] * O₂[-3 6; 4 5] * U2[5] * conj(mps.AR[j][-4 6; 7]) * right_env[b][1 3; 7]
                push!(tensors, value)
            end
        end
        push!(tens_new, sum(tensors))
    end
    return PeriodicArray(tens_new)
end

function left_env_below(mps, H, j, envs, O₁)
    U1 = Tensor(ones, (O₁).codom[1])
    left_env = leftenv(envs, j, mps)
    tens_new = []
    for b = 1:size(H[j])[2]
        tensors = []
        for a = 1:size(H[j])[1]
            if (a,b) in keys(H[j])
                @tensor value[-1 -2; -3 -4] := mps.AC[j][1 2; -4] * H[j][a,b][3 4; 2 -3] * O₁[5 6; 4 -2] * conj(U1[5]) * conj(mps.AC[j][7 6; -1]) * left_env[a][7 3; 1]
                push!(tensors, value)
            end
        end
        for tens = tensors
            println("new tensor")
            println(tens.dom)
            println(tens.codom)
        end
        sum_test = sum(tensors)
        push!(tens_new, sum(tensors))
    end
    return PeriodicArray(tens_new)
end


function correlator_tf(state, H, O₁, O₂, middle, i::Int, N::Int)
    S₁ = (O₁).codom[1]
    S₂ = (O₂).dom[2]

    G = similar(1:N, scalartype(state))
    U1 = Tensor(ones, S₁)
    U2 = Tensor(ones, S₂)

    envs = environments(mps, H)
    left_env = leftenv(envs, i, mps)
    right_env = rightenv(envs, i, mps)
    @tensor A_above[-1 -2; -3 -4] := mps.AC[i][-1 1; -4] * O₁[2 -2; 1 -3] * conj(U1[2])
    left = left_env * TransferMatrix(A_above, H[i], mps.AC[i])
    # println(typeof(left))
    # left = transfer_left(left_env, H[i], A_above, mps.AC[i])
    value_onsite = 0.0
    # left_env2 = leftenv(envs, i+1, mps)
    # right_env2 = rightenv(envs, i+1, mps)
    left_env2 = leftenv(envs, i+1, mps)
    right_env2 = rightenv(envs, i+1, mps)
    # for (j,k) in keys(H[i])
    #     value_onsite += @tensor left_env2[j][10 5; 1] * mps.AC[i+1][1 2; 13] * O₁[3 4; 2 7] * conj(U1[3]) * H[i+1][j,k][5 6; 4 12] * O₂[7 9; 6 8] * U2[8] * conj(mps.AC[i+1][10 9; 11]) * right_env2[k][13 12; 11]
    # end
    j = 1
    k = size(H[j])[1]
    # value_onsite = @tensor left_env2[j][10 5; 1] * mps.AC[i+1][1 2; 13] * O₁[3 4; 2 7] * conj(U1[3]) * H[i+1][j,k][5 6; 4 12] * O₂[7 9; 6 8] * U2[8] * conj(mps.AC[i+1][10 9; 11]) * right_env2[k][13 12; 11]
    value_onsite = @tensor left_env2[j][10 5; 1] * mps.AC[i+1][1 2; 13] * O₁[3 4; 2 7] * conj(U1[3]) * H[i+1][j,k][5 6; 4 12] * O₂[7 9; 6 8] * U2[8] * conj(mps.AC[i+1][10 9; 11]) * right_env2[k][13 12; 11]
    G[i] = value_onsite/2

    for j = 1:i-1
        G[j] = 0.0
    end
    for j = i+1:N
        # println("started for j = $(j)")
        # @tensor A_above[-1 -2; -3 -4] := mps.AR[j][-1 1; -4] * O₂[-3 -2; 1 2] * (U2[2])
        # right = TransferMatrix(A_above, H[j], mps.AR[j]) * rightenv(envs, j, mps)
        right = right_env_below(mps, H, j, envs, O₂)
        # println("printing right")
        # println(right)
        # println(keys(right))
        # println(keys(left))
        # println(typeof(left))
        # println(typeof(right))
        # println("Printing keys for j = $(j)")
        # println("fucking hell")

        G[j] = (-1im)*final_value(left, right)
        # println("done with final_value")
        @tensor A_middle[-1 -2; -3] := mps.AR[j][-1 1; -3] * middle[-2; 1]
        # @tensor A_middle_below[-1 -2; -3] := mps.AR[j][-1 1; -3] * conj(middle[1; -2])
        left = left * TransferMatrix(A_middle, H[j], mps.AR[j])
        # left = left * TransferMatrix(mps.AR[j], H[j], A_middle_below)
    end
    return G
end

function correlator_tf(state, H, O₁, O₂, middle, i::Int, N::Int)
    S₁ = (O₁).codom[1]
    S₂ = (O₂).dom[2]

    G = similar(1:N, scalartype(state))
    U1 = Tensor(ones, S₁)
    U2 = Tensor(ones, S₂)

    envs = environments(mps, H)
    left_env = leftenv(envs, i, mps)
    right_env = rightenv(envs, i, mps)
    @tensor A_above[-1 -2; -3 -4] := mps.AC[i][-1 1; -4] * O₁[2 -2; 1 -3] * conj(U1[2])
    left = left_env * TransferMatrix(A_above, H[i], mps.AC[i])
    left_env2 = leftenv(envs, i+1, mps)
    right_env2 = rightenv(envs, i+1, mps)
    j = 1
    k = size(H[j])[1]
    value_onsite = @tensor left_env2[j][10 5; 1] * mps.AC[i+1][1 2; 13] * O₁[3 4; 2 7] * conj(U1[3]) * H[i+1][j,k][5 6; 4 12] * O₂[7 9; 6 8] * U2[8] * conj(mps.AC[i+1][10 9; 11]) * right_env2[k][13 12; 11]
    G[i] = value_onsite/2

    for j = 1:i-1
        G[j] = 0.0
    end
    return G
end

function correlator_tf_april(state, H, O₁, O₂, middle, i::Int, N::Int)
    S₁ = (O₁).codom[1]
    S₂ = (O₂).dom[2]

    G = similar(1:N, scalartype(state))
    U1 = Tensor(ones, S₁)
    U2 = Tensor(ones, S₂)

    envs = environments(mps, H)
    left_env = leftenv(envs, i, mps)
    right_env = rightenv(envs, i, mps)
    @tensor A_above[-1 -2; -3 -4] := mps.AL[i][-1 1; -4] * O₁[2 -2; 1 -3] * conj(U1[2])
    left = left_env * TransferMatrix(A_above, H[i], mps.AL[i])
    value_onsite = 0.0
    # left_env2 = leftenv(envs, i+1, mps)
    # right_env2 = rightenv(envs, i+1, mps)
    left_env2 = leftenv(envs, i+1, mps)
    right_env2 = rightenv(envs, i+1, mps)
    # for (j,k) in keys(H[i])
    #     value_onsite += @tensor left_env2[j][10 5; 1] * mps.AC[i+1][1 2; 13] * O₁[3 4; 2 7] * conj(U1[3]) * H[i+1][j,k][5 6; 4 12] * O₂[7 9; 6 8] * U2[8] * conj(mps.AC[i+1][10 9; 11]) * right_env2[k][13 12; 11]
    # end
    j = 1
    k = size(H[j])[1]
    # value_onsite = @tensor left_env2[j][10 5; 1] * mps.AC[i+1][1 2; 13] * O₁[3 4; 2 7] * conj(U1[3]) * H[i+1][j,k][5 6; 4 12] * O₂[7 9; 6 8] * U2[8] * conj(mps.AC[i+1][10 9; 11]) * right_env2[k][13 12; 11]
    value_onsite = @tensor left_env2[j][10 5; 1] * mps.AC[i+1][1 2; 13] * O₁[3 4; 2 7] * conj(U1[3]) * H[i+1][j,k][5 6; 4 12] * O₂[7 9; 6 8] * U2[8] * conj(mps.AC[i+1][10 9; 11]) * right_env2[k][13 12; 11]
    G[i] = value_onsite/2

    for j = 1:i-1
        G[j] = 0.0
    end
    for j = i+1:N
        println("j = $(j)")
        # right = right_env_below(mps, H, j, envs, O₂)
        # right_env = rightenv(envs, j, mps)
        # tens_new = []
        # for a = 1:size(H[j])[1]
        #     tensors = []
        #     for b = 1:size(H[j])[2]
        #         if (a,b) in keys(H[j])
        #             @tensor value[-1 -3; -2 -4] := mps.AR[j][-1 2; 1] * H[j][a,b][-2 4; 2 3] * O₂[-3 6; 4 5] * U2[5] * conj(mps.AR[j][-4 6; 7]) * right_env[b][1 3; 7]
        #             push!(tensors, value)
        #         end
        #     end
        #     push!(tens_new, sum(tensors))
        # end
        util = MPSKit.fill_data!(similar(st.AL[1], space(envs.lw[H.odim, i + 1], 2)), one)
        U2 = Tensor(ones, (O₂).codom[1])
        U2 = Tensor(ones, (O₂).dom[2])
        final = 0.0
        for index in (H.odim):-1:1
            @tensor apl[-1 -2; -3] := left[index][7 3; 6 1] * mps.AL[j][1 2; -3] * H[j][index, H.odim][3 4; 2 -2] * O₂[6 8; 4 5] * (U2[5]) * conj(mps.AL[j][7 8; -1])
            # @tensor apl[-1 -2; -3 -15] := left[index][7 6; 3 1] * mps.AR[j][1 2; -3] * H[j][index, H.odim][3 4; 2 -2] * O₂[6 8; 4 -15]* conj(mps.AR[j][7 8; -1])
            final += @plansor apl[1 2; 3] * r_LL(mps, j)[3; 1] * conj(util[2])
        end
        G[j] = (-1im)*final
        # G[j] = (-1im)*final_value(left, right)

        # println("done with final_value")
        @tensor A_middle[-1 -2; -3] := mps.AL[j][-1 1; -3] * middle[-2; 1]
        # @tensor A_middle_below[-1 -2; -3] := mps.AR[j][-1 1; -3] * conj(middle[1; -2])
        left = left * TransferMatrix(A_middle, H[j], mps.AL[j])
        # left = left * TransferMatrix(mps.AR[j], H[j], A_middle_below)
    end
    return G
end


function correlator_tf_new(state, H, O₁, O₂, middle, i::Int, N::Int)
    S₁ = (O₁).codom[1]
    S₂ = (O₂).dom[2]

    G = similar(1:N, scalartype(state))
    U1 = Tensor(ones, S₁)
    U2 = Tensor(ones, S₂)

    envs = environments(mps, H)
    left_env = leftenv(envs, i, mps)
    right_env = rightenv(envs, i, mps)
    @tensor A_above[-1 -2; -3 -4] := mps.AC[i][-1 1; -4] * O₁[2 -2; 1 -3] * conj(U1[2])
    left = left_env * TransferMatrix(A_above, H[i], mps.AC[i])
    left_env2 = leftenv(envs, i+1, mps)
    right_env2 = rightenv(envs, i+1, mps)
    j = 1
    k = size(H[j])[1]
    for j = 1:6
        for k = 1:6
            value_onsite = @tensor left_env2[j][10 5; 1] * mps.AC[i+1][1 2; 13] * O₂[7 4; 2 3] * (U2[3]) * H[i+1][j,k][5 6; 4 12] * O₁[8 9; 6 7] * conj(U1[8]) * conj(mps.AC[i+1][10 9; 11]) * right_env2[k][13 12; 11]
            println("for (j,k) = ($(j),$(k)), value = $(value_onsite)")
        end
    end
    G[i] = value_onsite/2

    for j = 1:i-1
        println("zero for $(j)")
        G[j] = 0.0
    end
    for j = i+1:N
        right = right_env_below(mps, H, j, envs, O₂)
        G[j] = (-1im)*final_value(left, right)
        @tensor A_middle[-1 -2; -3] := mps.AR[j][-1 1; -3] * middle[-2; 1]
        left = left * TransferMatrix(A_middle, H[j], mps.AR[j])
    end
    return G
end

function correlator_tf_swapped(state, H, O₁, O₂, middle, i::Int, N::Int)
    S₁ = (O₁).codom[1]
    S₂ = (O₂).dom[2]

    G = similar(1:N, scalartype(state))
    U1 = Tensor(ones, S₁)
    U2 = Tensor(ones, S₂)

    envs = environments(mps, H)
    # left_env = leftenv(envs, i, mps)
    # right_env = rightenv(envs, i, mps)
    # @tensor A_above[-1 -2; -3 -4] := mps.AC[i][-1 1; -4] * O₁[2 -2; 1 -3] * conj(U1[2])
    # # @tensor A_below[-1 -2; -3 -4] := 
    # left = left_env * TransferMatrix(A_above, H[i], mps.AC[i])
    left = left_env_below(mps, H, i, envs, O₁)
    value_onsite = 0.0
    left_env2 = leftenv(envs, i, mps)
    right_env2 = rightenv(envs, i, mps)

    j = 1
    k = size(H[i])[1]
    value_onsite = @tensor left_env2[j][10 5; 1] * mps.AC[i][1 2; 13] * O₁[3 4; 2 7] * conj(U1[3]) * H[i][j,k][5 6; 4 12] * O₂[7 9; 6 8] * U2[8] * conj(mps.AC[i][10 9; 11]) * right_env2[k][13 12; 11]
    value_onsite = @tensor left_env2[j][10 5; 1] * mps.AC[i][1 2; 13] * O₂[7 4; 2 3] * conj(U1[3]) * H[i][j,k][5 6; 4 12] * O₁[8 9; 6 7] * U2[8] * conj(mps.AC[i][10 9; 11]) * right_env2[k][13 12; 11]
    # value_onsite = @tensor left_env2[j][10 5; 1] * mps.AC[i+1][1 2; 13] * O₁[3 9; 6 7] * conj(U1[3]) * H[i+1][j,k][5 6; 4 12] * O₂[7 4; 2 8] * U2[8] * conj(mps.AC[i+1][10 9; 11]) * right_env2[k][13 12; 11]
    G[i] = value_onsite/2

    for j = 1:i-1
        G[j] = 0.0
    end
    for j = i+1:N
        right_env = rightenv(envs, j, mps)
        @tensor A_above[-1 -2; -3 -4] := mps.AC[i][-1 1; -4] * O₂[-3 -2; 1 2] * conj(U2[2])
        right =  TransferMatrix(A_above, H[i], mps.AC[i]) * right_env
        # right = right_env_below(mps, H, j, envs, O₂)
        G[j] = (-1im)*final_value(left, right)
        @tensor A_middle[-1 -2; -3] := mps.AR[j][-1 1; -3] * middle[-2; 1]
        left = left * TransferMatrix(A_middle, H[j], mps.AR[j])
    end
    return G
end


# @tensor A_middle_below[-1 -2; -3] := mps.AR[i][-1 1; -3] * conj(S_z_symm[-2; 1])

#=
m = 0.3
truncation = 2.5
Delta_g = 0.0
v = 0.0

i = 2

@load "SanderDM_Thesis_2324/gs_mps_trunc_$(truncation)_mass_$(m)_v_0.0_Delta_g_$(Delta_g)" mps
@load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
@load "operators_new" S⁺swap S⁻swap S_z_symm


spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
unit_tensor = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], pspace, pspace)
H_unit = @mpoham sum((unit_tensor){i} for i in vertices(InfiniteChain(2)))
H = get_thirring_hamiltonian_symmetric(m, Delta_g, v)

S₁ = (S⁺).codom[1]
S₂ = (S⁻).dom[2]
U1 = Tensor(ones, S₁)
U2 = Tensor(ones, S₂)

envs = environments(mps, H)
left_env = leftenv(envs, i, mps)
right_env = rightenv(envs, i, mps)

# println(left_env[1].codom)
# println(left_env[1].dom)
# println(right_env[1].codom)
# println(right_env[1].dom)

# for (j,k) in keys(H[i])
#     println("($(j),$(k))")
#     println(H[i][j,k].codom)
#     println(H[i][j,k].dom)
# end

# r = right_env_below(mps, H, i+1, envs, S⁻)

G1 = correlator_tf(mps, H, S⁺, S⁻, (2*im)*S_z_symm, 1, 10)
G2 = correlator_tf(mps, H, S⁺, S⁻, (2*im)*S_z_symm, 2, 10)
G3 = correlator_tf(mps, H, S⁻swap, S⁺swap, (2*im)*S_z_symm, 1, 10)
G4 = correlator_tf(mps, H, S⁻swap, S⁺swap, (2*im)*S_z_symm, 2, 10)
G5 = correlator_tf_new(mps, H, S⁺, S⁻, (2*im)*S_z_symm, 1, 10)
G6 = correlator_tf_new(mps, H, S⁺, S⁻, (2*im)*S_z_symm, 2, 10)

break

# G_new = correlator_tf(mps, H_unit, S⁺, S⁻, (2*im)*S_z_symm, 1, 10)
G_new2 = correlator_tf(mps, H_unit, S⁺, S⁻, (2*im)*S_z_symm, 1, 10)
G_old = correlator(mps, S⁺, S⁻, S_z_symm, 1, 10)

#=

break

println(mps.AR[i].codom)
println(mps.AR[i].dom)
println(mps.AR[i+1].codom)
println(mps.AR[i+1].dom)

break

envs = environments(mps, H)
left_env = leftenv(envs, i, mps)
right_env = rightenv(envs, i, mps)

println("left")
for key in keys(left_env)
    println(key)
end
println("right")
for key in keys(right_env)
    println(key)
end
println("H[i]")
for key in keys(H[i])
    println(key)
end





break

G_2 = correlator_tf(mps, H_unit, S⁺, S⁻, (-2*im)*S_z_symm, 2, 10)
G_old = correlator(mps, S⁺, S⁻, S_z_symm, 1, 10)

N = 3
corr = zeros(ComplexF64, 2*N+2, 2*N+2)
for i = 2:2*N+3
    corr_bigger = correlator(mps, S⁺, S⁻, S_z_symm, i-1, 2*N+2)
    corr[i-1,:] = corr_bigger#[2:2*N+1]
end
# H = H_unit



corr_new = zeros(ComplexF64, 2*N+2, 2*N+2)
for i = 2:2*N+3
    corr_bigger = correlator_tf(mps, H, S⁺, S⁻, S_z_symm, i-1, 2*N+2)
    corr_new[i-1,:] = corr_bigger#[2:2*N+1]
end



break

envs = environments(mps, H_unit)
right_env = rightenv(envs, i+1, mps)
left_env = leftenv(envs, i+1, mps)
final_value(left_env, right_env)
@tensor A_below[-1 -4; -2 -3] := mps.AC[i+1][-1 1; -2] * conj(S⁻[-3 1; -4 2]) * conj(U1[2])
t_R = TransferMatrix(mps.AC[i+1], H_unit[i+1], A_below)
right = t_R * right_env

t = TransferMatrix(mps.AR[i], H_unit[i], mps.AR[i])

break


H = get_thirring_hamiltonian_symmetric(m, Delta_g, v)

envs = environments(mps, H)
left_env = leftenv(envs, i, mps)
right_env = rightenv(envs, i+1, mps)

#transfer.jl, lines 112 and beyond

# #mpo transfer, but with A an excitation-tensor
# function transfer_left(v::MPSTensor, O::MPOTensor, A::MPOTensor, Ab::MPSTensor)
#     @plansor t[-1 -2; -3 -4] := v[4 2; 1] * A[1 3; -3 -4] * O[2 5; 3 -2] * conj(Ab[4 5; -1])
# end

@tensor A_above[-1 -2; -3 -4] := mps.AC[i][-1 1; -4] * S⁺[2 -2; 1 -3] * conj(U1[2])
@tensor A_below[-1 -3; -2 -4] := mps.AC[i+1][-1 1; -2] * conj(S⁻[-3 1; -4 2]) * conj(U1[2])
@tensor A_above2[-1 -2; -3 -4] := mps.AC[i+1][-1 1; -4] * S⁻[-3 -2; 1 2] * (U2[2])
# @tensor A_above2[-1 -3; -2 -4] := mps.AC[i+1][-1 1; -4] * S⁻[-3 -2; 1 2] * (U2[2])

t_wo = TransferMatrix(mps.AC[i], mps.AC[i])
t_O = TransferMatrix(mps.AC[i], (-2*im)*S_z_symm, mps.AC[i])
t_L = TransferMatrix(A_above, H[i], mps.AC[i])
# t_R = TransferMatrix(mps.AC[i+1], H[i+1], mps.AC[i+1])
t_R = TransferMatrix(mps.AC[i+1], H[i+1], A_below)
t_R = TransferMatrix(A_above2, H[i+1], mps.AC[i+1])
left = left_env * t_L
right = t_R * right_env

final = 0.0
for key in keys(left)
    global final
    final += @tensor left[key][1 3 4 2] * right[key][2 3 4 1]
end

transfer_left(left_env, H, A_above, mps.AC[i])


for (j,k) in keys(H[i])
    @tensor S_above[-1 -2 -3; -4 -5 -6] := S⁻[-2 1; -4 -6] * H[i][j,k][-1 -3; 1 -5]
    @tensor S_below[-1 -2 -3; -4 -5 -6] := H[i][j,k][-1 1; -4 -5] * S⁻[-2 -3; 1 -6]
    println(norm(S_below-S_above))
end

=#

=#