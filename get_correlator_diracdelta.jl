using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using LaTeXStrings

include("get_thirring_hamiltonian_symmetric.jl")

#=

include("get_thirring_hamiltonian_symmetric.jl")


function _Pk_matrix(k, m, v)
    # return (1.0, 0.0)
    if (m == 0.0 && v == 0.0)
        a_help = -(1im/2)*(1-exp(im*k))
        # λ = -abs(a_help)
        a = conj(a_help)
        b = abs(a_help)
    else 
        λ = (v/2)*sin(k) - sqrt(m^2 + (sin(k/2))^2)
        a = (m+(v/2)*sin(k)) - λ
        b = -(1im/2)*(1-exp(im*k))
    end
    norm = sqrt(abs(a)^2+abs(b)^2)
    return (-b/norm, a/norm)
end

function _c_matrix_plus(k, m, v)
    if (m == 0.0 && v == 0.0)
        a₊ = -1.0
        b₊ = exp(im*k/2)
    else
        λ = sqrt(m^2+sin(k/2)^2)
        a₊ = m - λ
        b₊ = -exp(im*k/2)*sin(k/2)
    end
    c₊¹ = -b₊
    c₊² = a₊
    norm = sqrt(abs(c₊¹)^2 + abs(c₊²)^2)
    return (c₊¹/norm, c₊²/norm)
end

function _c_matrix_min(k, m, v)
    if (m == 0.0 && v == 0.0)
        a₊ = 1.0
        b₊ = exp(im*k/2)
    else
        λ = -sqrt(m^2+sin(k/2)^2)
        a₊ = m - λ
        b₊ = -exp(im*k/2)*sin(k/2)
    end
    c₊¹ = -b₊
    c₊² = a₊
    norm = sqrt(abs(c₊¹)^2 + abs(c₊²)^2)
    return (c₊¹/norm, c₊²/norm)
end

function _c_matrix_separate(k, m, v)
    if (k < 0)
        return _c_matrix_min(k, m, v)
    else
        return _c_matrix_plus(k, m, v)
    end
end

function _c_matrix(k, m, v)
    if (k < 0.0)
        sign = -1.0
    else
        sign = 1.0
    end

    if (m == 0.0 && v == 0.0)
        c₊¹ = -1
        c₊² = exp(-im*k/2)
    else
        λ = sign*sqrt(m^2+sin(k/2)^2)
        a₊ = m - λ
        b₊ = -exp(im*k/2)*sin(k/2)
        c₊¹ = -b₊
        c₊² = a₊
    end
    norm = sqrt(abs(c₊¹)^2 + abs(c₊²)^2)
    return (c₊¹/norm, c₊²/norm)
end


function _integrand_wave_packet_occupation_number(k₀, x₀, m, v, corr::Matrix)
    (c¹, c²) = _c_matrix(k₀, m, v)
    sum = 0
    for m = 0:N-1
        for n = 0:N-1
            factor = exp(im*k₀*(m-n))
            sum += factor * abs(c¹)^2 * corr[1+2*n,1+2*m]
            sum += factor * c² * conj(c¹) * corr[1+2*n+1,1+2*m]
            sum += factor * c¹ * conj(c²) * corr[1+2*n,1+2*m+1]
            sum += factor * abs(c²)^2 * corr[1+2*n+1,1+2*m+1]
        end
    end
    return sum/N
end

function wave_packet_occupation_number(k₀, x₀, m, v, corr::Matrix, corr_energy::Matrix)
    occupation_number = _integrand_wave_packet_occupation_number(k₀, x₀, m, v, corr)
    # energy = _integrand_wave_packet_occupation_number(k₀, x₀, m, v, corr_energy)
    energy = 0
    return (occupation_number, energy)
end


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
S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], Triv_space ⊗ pspace, pspace ⊗ Triv_space)

H = get_thirring_hamiltonian_symmetric(m, delta_g, v)

S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)
unit_tensor = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], pspace, pspace)

println(typeof(S_z_symm))
# println(typeof(hamiltonian))

# @tensor H_with_S_z_symm[-1 -3; -2 -4] := hamiltonian[-1 -3; 1 -4] * S_z_symm[1; -2]

U = Tensor(ones, pspace)

println(U)

N = 50

corr = zeros(ComplexF64, 2*N, 2*N)
corr_energy = zeros(ComplexF64, 2*N, 2*N)

=#

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)

S_xx_S_yy = 0.5*TensorMap([0.0+0.0im 0.0+0.0im 0.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im 1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im 0.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im 0.0+0.0im 0.0+0.0im], pspace ⊗ pspace, pspace ⊗ pspace)

J = [1.0 -1.0]
Sz_plus_12 = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace)
    
Plus_space = U1Space(1 => 1)
S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace ⊗ Plus_space)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace)
S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], Plus_space ⊗ pspace, pspace ⊗ Plus_space)

@tensor operator_threesite[-1 -2 -3; -4 -5 -6] := S⁺[-1; -4 1] * S_z_symm[1 -2; -5 2] * S⁻[2 -3; -6]
@tensor operator_threesite_final[-1 -2 -3; -4 -5 -6] := operator_threesite[-1 -2 -3; -4 -5 -6] - operator_threesite[-3 -2 -1; -6 -5 -4]

S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)

Hopping_term = (-1) * @mpoham sum(S_xx_S_yy{i, i + 1} for i in vertices(InfiniteChain(2)))

H_hop = -S_xx_S_yy

U, S, V = tsvd(H_hop, (3, 1), (2, 4))
# U2, S2, V2 = tsvd(transpose(H_hop,((1, 3), (2, 4))))

@tensor left[-1; -2 -3] := U[-2 -1; 1] * S[1; -3]
@tensor left[-1 -2; -3] = left[-1; -2 -3]

println(typeof(left))
left = add_util_leg_front(left)
println(typeof(left))

@tensor right[-1 -2; -3] := V[-1; -2 -3]
right = add_util_leg_back(right)

println(typeof(U))
println(typeof(S))
println(typeof(V))
println(typeof(left))
println(typeof(right))

println("H = ")
println(H_hop)

U_H_left = Tensor(ones, space(left)[1])
U_H_right = Tensor(ones, space(right)[4])

unit_left = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], space(left)[1] ⊗ pspace, pspace ⊗ space(left)[1])
unit_right = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], conj(space(right)[4]) ⊗ pspace, pspace ⊗ conj(space(right)[4]))


@tensor H_check[-1 -2; -3 -4] := conj(U_H_left[1]) * left[1 -1; -3 2] * right[2 -2 -4 3] * conj(U_H_right[3])
# @tensor H_check[-1 -2; -3 -4] := U_H_left[1] * left[1 -1; -3 2] * right[2 -2 -4 3] * U_H_right[3]


@load "everything_needed" N mps S⁺ S⁻ S_z_symm H

# op_index = 4
i = 1
O₁ = S⁺
O₂ = S⁻
middle_t = S_z_symm
# H_list = fill(unit_left, N)

# H_list[op_index] = left
# H_list[op_index+1] = right

# for i = op_index+2:N
#     H_list[i] = unit_right
# end

# U_H_left = Tensor(ones, space(H_list[i])[1])
# U_H_right = Tensor(ones, space(H_list[i])[4])

U_space₁ = Tensor(ones, space(O₁)[1])
U_space₂ = Tensor(ones, space(O₂)[4])

tensors = [i%2 == 0 ? mps.AR[2] : mps.AR[1] for i in 1:N]
tensors[1] = mps.AC[1]

state = FiniteMPS(tensors)

G = similar(1:N, scalartype(state))

for op_index = 1:N-1

    println("op_index = $(op_index) of N = $(N)")
    H_list = fill(unit_left, N)

    H_list[op_index] = left
    H_list[op_index+1] = right

    U_H_left = Tensor(ones, space(H_list[i])[1])
    U_H_right = Tensor(ones, space(H_list[i])[4])

    for i₂ = op_index+2:N
        H_list[i₂] = unit_right
    end

    @tensor Vₗ[-1 -2; -3 -4] := state.AC[i][1 2; -4] * conj(U_space₁[3]) * O₁[3 4; 2 -3] * H_list[i][5 6; 4 -2] * conj(U_H_left[5]) * conj(state.AC[i][1 6; -1])
    @tensor Vᵣ[-1 -2; -3 -4] := state.AC[i][-1 2; 1] * H_list[i][-2 4; 2 3] * conj(U_H_right[3]) * O₂[-3 6; 4 5] * conj(U_space₂[5]) * conj(state.AC[i][-4 6; 1])

    for j = i-1:-1:1 # j < i ==> factor -i
        # global Vᵣ
        if j < i-1
            @tensor Vᵣ[-1 -2; -3 -4] := (-2im) * Vᵣ[1 4; -3 6] * (state.AL[j+1])[-1 2; 1] * middle_t[3; 2] * H_list[j+1][-2 5; 3 4] * conj((state.AL[j+1])[-4 5; 6])
        end
        U_H_left = Tensor(ones, space(H_list[j])[1])
        G[j] = 1im*(@tensor Vᵣ[4 8; 5 10] * state.AL[j][1 2; 4] * O₁[3 6; 2 5] * conj(U_space₁[3]) * H_list[j][7 9; 6 8] * conj(U_H_left[7]) * conj(state.AL[j][1 9; 10]))
    end
    U_H_left = Tensor(ones, space(H_list[i])[1])
    U_H_right = U_H_right = Tensor(ones, space(H_list[i+1])[4])
    G[i] = @tensor (state.AC[i])[1 2; 10] * O₂[5 4; 2 3] * conj(U_space₂[3]) * H_list[i][6 7; 4 12] * conj(U_H_left[6]) * O₁[8 9; 7 5] * conj(U_space₁[8]) * conj((state.AC[i])[1 9; 15]) * (state.AR[i+1])[10 11; 16] * H_list[i+1][12 14; 11 13] * conj(U_H_right[13]) * conj((state.AR[i+1])[15 14; 16])
    for j = i+1:N # j > i ==> factor i and conjugate
        # global Vₗ
        # global U_H_right
        if j > i+1
            @tensor Vₗ[-1 -2; -3 -4] := (2im) * Vₗ[1 4; -3 1] * (state.AR[j-1])[1 2; -4] * middle_t[3; 2] * H_list[j-1][4 5; 3 -2] * conj((state.AR[j-1])[1 5; -1])
        end
        U_H_right = Tensor(ones, space(H_list[j])[4])
        # @tensor G[-3 -4 -6 -8] := -1im*(Vₗ[9 -6; 7 1] * state.AR[j][1 2; 10] * conj(state.AR[j][9 -4; 10])) * H_list[j][7 -3; 2 -8]
        # @tensor G[-3 -4 -5 -6] := -1im*(Vₗ[9 -5; -6 1] * state.AR[j][1 2; 10] * conj(state.AR[j][9 -4; 10])) * H_list[j][7 -3; 2 -7]
        G[j] = -1im*(@tensor Vₗ[9 7; 4 1] * state.AR[j][1 2; 10] * H_list[j][7 5; 2 3] * conj(U_H_right[3]) * O₂[4 8; 5 6] * conj(U_space₂[6]) * conj(state.AR[j][9 8; 10]))
    end
end



break

# @save "Hopping_term" H_hop

# @load "Hopping_term" H_hop
@load "everything_needed" N mps S⁺ S⁻ S_z_symm H

for i = 1:2*N
    # corr[i,:] = correlator(mps, S⁺, S⁻, S_z_symm, i, 2*N)
    corr_energy[i,:] = correlator(mps, S⁺, S⁻, S_z_symm, left, right, i, 2*N)
    # corr_energy[i,:] = correlator(mps, S⁺, S⁻, H_with_S_z_symm, i, N)
    # change unit_tensor to (-2im*S_z_symm)
end

x₀ = 50
σ = 10/N

k_max = pi
data_points = 100


X = [(2*pi)/N*i - pi for i = 0:N-1]
N̂ = zeros(Float64, data_points)
Ê = zeros(Float64, data_points)

for (index, k₀) in enumerate(X)
    (occ,e) = wave_packet_occupation_number(k₀, x₀, m, v, corr, corr_energy)
    println("occ for k0 = $(k₀) is $(occ)")
    # println("energy for k0 = $(k₀) is $(e)")
    N̂[index] = real(occ)
    Ê[index] = real(e)
end

@save "test_newwww" X N̂ Ê


# plt = plot(N̂, Ê)
plt = plot(N̂, xlabel = "k", ylabel = L"$\left<\hat{N}\right>$")
title!("Occupation number in function of k")
display(plt)
