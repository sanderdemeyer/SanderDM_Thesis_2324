using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using LaTeXStrings

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

function gaussian(k, k₀, σ, x₀)
    # k = mod(k + pi, 2*pi) - pi
    return exp(-im*k*x₀) * exp(-((k-k₀)/(2*σ))^2)/2
    return (abs(k-k₀) < 1e-5)*1
end

function _integrand_wave_packet_occupation_number_dirac_delta(k₀, x₀, σ, m, v, corr::Matrix)
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

function _integrand_wave_packet_occupation_number(k₀, x₀, σ, m, v, corr::Matrix)
    sum = 0
    for iₖ = 0:N-1
        k₁ = (2*pi/N)*iₖ - pi
        for jₖ = 0:N-1
            k₂ = (2*pi/N)*jₖ - pi
            (c¹₁, c²₁) = _c_matrix(k₁, m, v)
            (c¹₂, c²₂) = _c_matrix(k₂, m, v)
            for m = 0:N-1
                for n = 0:N-1
                    # factor = exp(im*k₀*(m-n))
                    factor = exp(im*(-k₁*n+k₂*m)) * gaussian(k₁, k₀, σ, x₀) * conj(gaussian(k₂, k₀, σ, x₀))
                    sum += factor * c¹₁ * conj(c¹₂) * corr[1+2*n,1+2*m]
                    sum += factor * c²₁ * conj(c¹₂) * corr[1+2*n+1,1+2*m]
                    sum += factor * c¹₁ * conj(c²₂) * corr[1+2*n,1+2*m+1]
                    sum += factor * c²₁ * conj(c²₂) * corr[1+2*n+1,1+2*m+1]
                end
            end
        end
    end
    return sum/N
end



function wave_packet_occupation_number(k₀, x₀, σ, m, v, corr::Matrix, corr_energy::Matrix)
    occupation_number = _integrand_wave_packet_occupation_number(k₀, x₀, σ, m, v, corr)
    energy = _integrand_wave_packet_occupation_number(k₀, x₀, σ, m, v, corr_energy)
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
# S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], Triv_space ⊗ pspace, pspace ⊗ Triv_space)

H = get_thirring_hamiltonian_symmetric(m, delta_g, v)

S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)
unit_tensor = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], pspace, pspace)

println(typeof(S_z_symm))
# println(typeof(hamiltonian))

# @tensor H_with_S_z_symm[-1 -3; -2 -4] := hamiltonian[-1 -3; 1 -4] * S_z_symm[1; -2]

U = Tensor(ones, pspace)

println(U)

N = 30



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

unit_operator = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], pspace, pspace)
unit_MPO = @mpoham sum((unit_operator){i} for i in vertices(InfiniteChain(2)))

unit_operator = add_util_leg(unit_operator)


U_H_left = Tensor(ones, space(left)[1])
U_H_right = Tensor(ones, space(right)[4])

unit_left = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], space(left)[1] ⊗ pspace, pspace ⊗ space(left)[1])
unit_right = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], conj(space(right)[4]) ⊗ pspace, pspace ⊗ conj(space(right)[4]))


@tensor H_check[-1 -2; -3 -4] := conj(U_H_left[1]) * left[1 -1; -3 2] * right[2 -2 -4 3] * conj(U_H_right[3])
# @tensor H_check[-1 -2; -3 -4] := U_H_left[1] * left[1 -1; -3 2] * right[2 -2 -4 3] * U_H_right[3]


@load "everything_needed" N mps S⁺ S⁻ S_z_symm H

break

# tensors = [i%2 == 0 ? mps.AR[2] : mps.AR[1] for i in 1:N]
# tensors[1] = mps.AC[1]

# state = FiniteMPS(tensors)

# op_index = 4
i = 1
O₁ = S⁺
O₂ = S⁻
middle_t = S_z_symm

# @load "Hopping_term" H_hop
@load "everything_needed" N mps S⁺ S⁻ S_z_symm H

@load "SanderDM_Thesis_2324/gs_mps_trunc_2.5_mass_0.0_v_0.0_Delta_g_0.0" mps

N = 30

corr = zeros(ComplexF64, 2*N, 2*N)
corr_energy = zeros(ComplexF64, 2*N, 2*N)
corr_energy_check = zeros(ComplexF64, 2*N, 2*N)

# corr_check = zeros(ComplexF64, 2*N, 2*N)
# for i = 2:2*N+1
#     println("i = $(i)")
#     corr_bigger = correlator(mps, S⁺, S⁻, S_z_symm, unit_operator, unit_operator, i, 2*N+2)
#     corr_check[i-1,:] = corr_bigger[2:2*N+1]
# end

println("started with correlation")

corr_energy_check = correlator(mps, S⁺, S⁻, S_z_symm, unit_MPO, 2*N+2)

break


for i = 2:2*N+1
    println("i = $(i)")
    corr_bigger = correlator(mps, S⁺, S⁻, S_z_symm, i, 2*N+2)
    corr_energy_bigger = correlator(mps, S⁺, S⁻, S_z_symm, H, i, 2*N+2)
    corr_energy_check_bigger = correlator(mps, S⁺, S⁻, S_z_symm, unit_MPO, i, 2*N+2)
    corr[i-1,:] = corr_bigger[2:2*N+1]
    corr_energy[i-1,:] = corr_energy_bigger[2:2*N+1]
    corr_energy_check[i-1,:] = corr_energy_check_bigger[2:2*N+1]

    # corr_energy[i,:] = correlator(mps, S⁺, S⁻, H_with_S_z_symm, i, N)
    # change unit_tensor to (-2im*S_z_symm)
end

for i = 1:2*N
    for j = 1:i-1
        corr_energy[i,j] = conj(corr_energy[j,i])
        corr_energy_check[i,j] = conj(corr_energy_check[j,i])
    end
end

corr_energy_check2 = zeros(ComplexF64, 2*N, 2*N)

for i = 1:2*N
    for j = 1:2*N
        if (i == j)
            corr_energy_check2[i,j] = corr_energy_check[i,j]
        else
            corr_energy_check2[i,j] = corr_energy_check[i,j]/2
        end
    end
end

corr_ratio = zeros(ComplexF64, 2*N, 2*N)
for i = 1:2*N
    for j = 1:2*N
        corr_ratio[i,j] = corr[i,j]/corr_energy_check[i,j]
    end
end

break

for i = 1:2*N
    for j = 1:2*N
        if isnan(real(corr_energy[i,j]))
            println("for i = $(i) and j = $(j), corr is $(corr_energy[i,j])")
            corr_energy[i,j] = 0.0
        end
        if isnan(imag(corr_energy[i,j]))
            println("for i = $(i) and j = $(j), corr is $(corr_energy[i,j])")
            corr_energy[i,j] = 0.0
        end
    end
end
# break;

x₀ = div(N,2)
σ = 10/N

k_max = pi
data_points = N


X = [(2*pi)/N*i - pi for i = 0:N-1]
Y = [sin(i/2) for i in X]
N̂ = zeros(Float64, data_points)
Ê = zeros(Float64, data_points)

for (index, k₀) in enumerate(X)
    (occ,e) = wave_packet_occupation_number(k₀, x₀, σ, m, v, corr, corr_energy)
    println("occ for k0 = $(k₀) is $(occ)")
    println("energy for k0 = $(k₀) is $(e)")
    N̂[index] = real(occ)
    Ê[index] = real(e)
end

@save "test_new_N_$(N)" X N̂ Ê


# plt = plot(N̂, Ê)
plt = plot(X, N̂, xlabel = "k", ylabel = L"$\left<\hat{N}\right>$")
title!("Occupation number for N = $(N)")
display(plt)
savefig("Occupation number for N = $(N)")

break;

plt = plot(X, Ê, xlabel = "E", ylabel = L"$\left<\hat{N}\right>$")
title!("Occupation number in function of E")
display(plt)

E_here = zeros(Float64, 25)
X_here = zeros(Float64, 25)
for i = 1:25
    E_here[i] = Ê[i]
    X_here[i] = X[i]
end

plt = plot(Ê, xlabel = "k", ylabel = L"$\left<\hat{E}\right>$")
title!("Occupation number in function of k")
display(plt)

a = 70
b = 31.8
X_accurate = LinRange(-pi, 0, 1000)
Y_accurate = [a - b*sin(k/2) for k in X_accurate]

plt = plot(X_here, E_here, xlabel = "k", ylabel = L"$\left<\hat{E}\right>$", label = "data")
plt2 = plot(X_accurate, Y_accurate)
plot!(plt, X_accurate, Y_accurate, label="fit")
title!("title")
display(plt)


break

##

state = mps
envs = environments(state, H)
left_env = leftenv(envs, i, state)
right_env = rightenv(envs, i, state)

transf = TransferMatrix(mps.AC[i], H[i], mps.AC[i])
transf2 = TransferMatrix(mps.AC[i-1], H[i-1], mps.AC[i-1])

left_env * (transf * right_env)

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