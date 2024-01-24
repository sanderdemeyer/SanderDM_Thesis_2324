using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots

function gaussian_wave_packet(k, σ)
    return exp(-((k%(2*pi))/(2*σ))^2)
end

function _Pk_matrix(k, m, v)
    return (1.0, 0.0)
    if (m == 0.0 && v == 0.0)
        a_help = -(1im/2)*(1-exp(im*k))
        a = conj(a)
        b = abs(a_help)
    else 
        λ = (v/2)*sin(k) - sqrt(m^2 + (sin(k/2))^2)
        a = (m+(v/2)*sin(k)) - λ
        b = -(1im/2)*(1-exp(im*k))            
    end
    norm = sqrt(abs(a)^2+abs(b)^2)
    return (-b/norm, a/norm)
end

function _integrand_wave_packet_occupation_number(k₁, k₂, ω, x₀, σ, m, v, corr::Matrix)
    (c11, c21) = _Pk_matrix(k₁, m, v)
    (c12, c22) = _Pk_matrix(k₂, m, v)
    factor = gaussian_wave_packet(k₁-ω,σ)*gaussian_wave_packet(k₂-ω,σ)
    sum = 0
    for m = 1:div(N,2)
        for n = 1:div(N,2)
            #=
            # sum += exp(-1im*k₁*(2*n+x₀)+1im*k₂*(2*m+x₀)) * conj(c11)*c12 * corr[2*n,2*m]
            sum += exp(-1im*k₁*(n+x₀)+1im*k₂*(m+x₀)) * conj(c11)*c12 * corr[2*n,2*m]
            if 2*n+1 <= N
                # sum += exp(-1im*k₁*(2*n+1+x₀)+1im*k₂*(2*m+x₀)) * conj(c21)*c12 * corr[2*n+1,2*m]
                sum += exp(-1im*k₁*(n+x₀)+1im*k₂*(m+x₀)) * conj(c21)*c12 * corr[2*n+1,2*m]
            end
            if 2*m+1 <= N
                # sum += exp(-1im*k₁*(2*n+x₀)+1im*k₂*(2*m+1+x₀)) * conj(c11)*c22 * corr[2*n,2*m+1]
                sum += exp(-1im*k₁*(n+x₀)+1im*k₂*(m+x₀)) * conj(c11)*c22 * corr[2*n,2*m+1]
            end
            if (2*n+1 <= N) && (2*m+1 <= N)
                # sum += exp(-1im*k₁*(2*n+1+x₀)+1im*k₂*(2*m+1+x₀)) * conj(c21)*c22 * corr[2*n+1,2*m+1]
                sum += exp(-1im*k₁*(n+x₀)+1im*k₂*(m+x₀)) * conj(c21)*c22 * corr[2*n+1,2*m+1]
            end
            =#
            if ((2*n+1 <= N) && (2*m+1 <= N))
                sum += exp(-1im*k₁*(x₀-n)+1im*k₂*(x₀-m)) * conj(c11)*c12 * corr[2*n,2*m]
                sum += exp(-1im*k₁*(x₀-n)+1im*k₂*(x₀-m)) * conj(c21)*c12 * corr[2*n+1,2*m]
                sum += exp(-1im*k₁*(x₀-n)+1im*k₂*(x₀-m)) * conj(c11)*c22 * corr[2*n,2*m+1]
                sum += exp(-1im*k₁*(x₀-n)+1im*k₂*(x₀-m)) * conj(c21)*c22 * corr[2*n+1,2*m+1]
            end
        end
    end
    return factor*sum
end

function wave_packet_occupation_number(ω, x₀, σ, m, v, corr::Matrix)
    dk = 2*pi/N
    occupation_number = 0
    for i₁ = 0:N-1
        for i₂ = 0:N-1
            # println("started for k1 = $(i₁) and k2 = $(i₂)")
            occupation_number += _integrand_wave_packet_occupation_number(dk*i₁, dk*i₂, ω, x₀, σ, m, v, corr)
        end
    end
    return occupation_number
end


@load "SanderDM_Thesis_2324/test_gs_mps" gs_mps


spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)
H = @mpoham sum((S_z_symm){i} for i in vertices(InfiniteChain(2)))
Plus_space = U1Space(1 => 1)
Triv_space = U1Space(0 => 1)
S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Triv_space ⊗ pspace, pspace ⊗ Plus_space)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace ⊗ Triv_space)
S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], Triv_space ⊗ pspace, pspace ⊗ Triv_space)


S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)
unit_tensor = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], pspace, pspace)

U = Tensor(ones, pspace)

println(U)

N = 100

corr = zeros(ComplexF64, N, N)

for i = 1:N
    corr[i,:] = correlator(gs_mps, S⁺, S⁻, S_z_symm, i, N)
    # change unit_tensor to (-2im*S_z_symm)
end

println(typeof(corr))
println(corr)

N = 100
x₀ = 50
σ = 10/N
m = 0.0
v = 0.0

k_max = pi
data_points = 71

X = range(-k_max, k_max, data_points)
Y = zeros(Float64, data_points)
ωs = zeros(Float64, data_points)
ωs2 = zeros(Float64, data_points)

for (index, k₀) in enumerate(X)
    occ = wave_packet_occupation_number(k₀, x₀, σ, m, v, corr)
    println("occ for k0 = $(k₀) is $(occ)")
    Y[index] = real(occ)
    ωs[index] = v*sin(k₀)/2 + sqrt(m^2 + (sin(k₀/2))^2)
    ωs2[index] = v*sin(2*k₀)/2 + sqrt(m^2 + (sin(k₀))^2)
    # occ2 = wave_packet_occupation_number(-5.0, 5, 0.1, 0.4, 0.0, corr)
end
# println("occ positive is $(occ)")
# # println("occ2 negative is $(occ2)")
# println(typeof(occ))
# # println(correlator(gs_mps, S⁺, S⁻, 3, 4:7))

@save "test_newww" X Y ωs ωs2

plt = plot(X, Y)
title!("Plot 1")
display(plt)
plot(ωs, Y)
title!("Plot 2")
display(plt)
plot(ωs2, Y)
title!("Plot 3")
display(plt)
# println(TransferMatrix(gs_mps.AR[1], S⁺, gs_mps.AR[1]))
# println(TransferMatrix(gs_mps.AR[1:2], fill(S_z_symm,2), gs_mps.AR[1:2]))

# println(gs_mps.AR[1])
# println(gs_mps.AR[2])
# println(S_z_symm)


# println(typeof(TransferMatrix(gs_mps.AR[1], S⁺, gs_mps.AR[1])))
# println(typeof(TransferMatrix(gs_mps.AR[1:2])))

# println(typeof(S⁺))
# println(correlator(gs_mps, S⁺, S⁻, 1, 2:5))
# println(correlator(gs_mps, S⁺, S⁻, S_z_symm, 1, 2:5))


# println(TransferMatrix(gs_mps.AC[1]))
