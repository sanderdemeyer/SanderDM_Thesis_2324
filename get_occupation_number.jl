using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using LaTeXStrings
using Random

include("get_thirring_hamiltonian_symmetric.jl")


function _c_matrix(k, m, v, fasefactor)
    if (k < 0.0)
        sign = -1.0
    else
        sign = 1.0
    end

    if (m == 0.0 && v == 0.0)
        c₊¹ = -1
        # c₊² = exp(-im*k)
        c₊² = exp(-im*k/2)
    else
        if (k == 0.0)
            k = 1e-10
        end
        # λ = sign*sqrt(m^2+sin(k)^2)
        λ = sign*sqrt(m^2+sin(k/2)^2)
        c₊¹ = -1
        # c₊² = -exp(-im*k)*(m-λ)/sin(k)
        c₊² = -exp(-im*k/2)*(m-λ)/sin(k/2)
        # c₊² = exp(-im*k/2)
        
    end
    # fasefactor = exp(im*rand()*pi*2)
    print("here")
    norm = sqrt(abs(c₊¹)^2 + abs(c₊²)^2)
    return (c₊¹/norm*fasefactor, c₊²/norm*fasefactor)
end


function _c_matrix(k, m, v; fasefactor = 1.0)
    if (k < 0.0)
        sign = -1.0
    else
        sign = 1.0
    end

    if (m == 0.0 && v == 0.0)
        c₊¹ = -1
        c₊² = exp(-im*k/2)
    else
        if (k == 0.0)
            k = 1e-10
        end
        λ = sign*sqrt(m^2+sin(k/2)^2)
        c₊¹ = -1
        c₊² = -exp(-im*k/2)*(m-λ)/sin(k/2)
        
    end
    norm = sqrt(abs(c₊¹)^2 + abs(c₊²)^2)
    return (c₊¹/norm*fasefactor, c₊²/norm*fasefactor)
end



function gaussian(k, k₀, σ, x₀)
    k = mod(k + pi, 2*pi) - pi
    return exp(-im*k*x₀) * exp(-((k-k₀)/(2*σ))^2)
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

function _integrand_wave_packet_occupation_number_dirac_delta(k₀, x₀, σ, mass, v, corr::Matrix, fasefactor)
    sum = 0
    # fasefactor = exp(im*rand()*pi*2)
    (c¹₁, c²₁) = _c_matrix(k₀, mass, v, 1.0)
    (c¹₂, c²₂) = _c_matrix(k₀, mass, v, 1.0)
    for m = 0:N-1
        for n = 0:N-1
            factor = exp(im*(-k₀*n+k₀*m))
            # factor = exp(im*(m-n)*fasefactor)
            # factor = exp(-im*(-k₀*n+k₀*m))
            # factor = 1.0
            sum += factor * c¹₁ * conj(c¹₂) * corr[1+2*n,1+2*m]
            sum += factor * c²₁ * conj(c¹₂) * corr[1+2*n+1,1+2*m]
            sum += factor * c¹₁ * conj(c²₂) * corr[1+2*n,1+2*m+1]
            sum += factor * c²₁ * conj(c²₂) * corr[1+2*n+1,1+2*m+1]
        end
    end
    return sum/N
end

function get_occupation_number(mps, N₂, m, v; σ = 10/(div(N₂,2)-1), x₀ = div(div(N₂,2)-1,2), fasefactor = 1.0)
    N = div(N₂,2)-1
    println("N = $(N), sigma = $(σ), x_0 = $(x₀)")
    @load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
    corr = zeros(ComplexF64, 2*N, 2*N)
    for i = 2:2*N+1
        corr_bigger = correlator(mps, S⁺, S⁻, S_z_symm, i, 2*N+2)
        corr[i-1,:] = corr_bigger[2:2*N+1]
    end
    
    X = [(2*pi)/N*i - pi for i = 0:N-1]
    N̂ = zeros(Float64, N)
    
    for (index, k₀) in enumerate(X)
        # occ = _integrand_wave_packet_occupation_number(k₀, x₀, σ, m, v, corr)
        occ = _integrand_wave_packet_occupation_number_dirac_delta(k₀, x₀, σ, m, v, corr, fasefactor)
        N̂[index] = real(occ)
    end
    return N̂
end
