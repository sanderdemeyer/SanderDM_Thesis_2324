using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using LaTeXStrings

include("get_thirring_hamiltonian_symmetric.jl")

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
    k = mod(k + pi, 2*pi) - pi
    return exp(-im*k*x₀) * exp(-((k-k₀)/(2*σ))^2)
    return (abs(k-k₀) < 1e-5)*1
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

function get_occupation_number(mps, N, m, v)
    @load "everything_needed" S⁺ S⁻ S_z_symm
    x₀ = 5
    σ = 10/N
    
    corr = zeros(ComplexF64, 2*N, 2*N)
    corr_energy = zeros(ComplexF64, 2*N, 2*N)
    corr_energy_check = zeros(ComplexF64, 2*N, 2*N)

    for i = 2:2*N+1
        println("i = $(i)")
        corr_bigger = correlator(mps, S⁺, S⁻, S_z_symm, i, 2*N+2)
        # corr_energy_bigger = correlator(mps, S⁺, S⁻, S_z_symm, H, i, 2*N+2)
        # corr_energy_check_bigger = correlator(mps, S⁺, S⁻, S_z_symm, unit_MPO, i, 2*N+2)
        corr[i-1,:] = corr_bigger[2:2*N+1]
        # corr_energy[i-1,:] = corr_energy_bigger[2:2*N+1]
        # corr_energy_check[i-1,:] = corr_energy_check_bigger[2:2*N+1]
    end

    for i = 1:2*N
        for j = 1:i-1
            corr_energy[i,j] = conj(corr_energy[j,i])
            corr_energy_check[i,j] = conj(corr_energy_check[j,i])
        end
    end

    X = [(2*pi)/N*i - pi for i = 0:N-1]
    N̂ = zeros(Float64, N)
    Ê = zeros(Float64, N)

    for (index, k₀) in enumerate(X)
        (occ,e) = wave_packet_occupation_number(k₀, x₀, σ, m, v, corr, corr_energy)
        println("occ for k0 = $(k₀) is $(occ)")
        println("energy for k0 = $(k₀) is $(e)")
        N̂[index] = real(occ)
        Ê[index] = real(e)
    end

    return (X, N̂, Ê)
end