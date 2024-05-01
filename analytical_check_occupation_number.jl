using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using LaTeXStrings
using Random

include("get_X_tensors.jl")
include("get_thirring_hamiltonian_symmetric.jl")
include("get_occupation_number_matrices.jl")

function check_rotation(k, m)
    A = [m -(im/2)*(1-exp(im*k)); (im/2)*(1-exp(-im*k)) -m]
    eigen_result = eigen(A)
    # eigenvectors_matrix = eigen_result.vectors
    return adjoint(eigen_result.vectors)
end

function get_rotation(k, m)
    A = [m -(im/2)*(1-exp(im*k)); (im/2)*(1-exp(-im*k)) -m]
    eigen_result = eigen(A)
    # eigenvectors_matrix = eigen_result.vectors
    parameters = inv(transpose(eigen_result.vectors))
    a11 = (parameters[1,1])
    a12 = (parameters[1,2])
    a21 = (parameters[2,1])
    a22 = (parameters[2,2])
    # ax = -sin(k/2)*cos(k/2)
    # ay = sin(k/2)^2
    # az = m
    # a = sqrt(abs(ax)^2+abs(ay)^2+abs(az)^2)
    # @assert abs(a - sqrt(m^2+sin(k/2)^2)) < 1e-8
    # denominator = sqrt(2*a*(az+a))
    # b11 = (im*ay-ax)/denominator
    # b21 = (az+a)/denominator
    # b12 = (az+a)/denominator
    # b22 = (ax+im*ay)/denominator
    # new = [b11 b12; b21 b22]
    # new2 = inv(new)
    # # println(inv(new) - adjoint(new))
    # a11 = new2[1,1]
    # a12 = new2[1,2]
    # a21 = new2[2,1]
    # a22 = new2[2,2]
    return (a11,a12,a21,a22)#, (b11,b12,b21,b22))
end

function get_rotation_vector(mass, X)
    base = [get_rotation(k,mass) for k in X]
    return [[base[k_index][i] for k_index in 1:length(X)] for i = 1:4]
end

N = 20
mass = 1.0
X = [(2*pi)/N*i - pi for i = 0:N-1]

# corr = zeros(ComplexF64, 2*N, 2*N)
# for m = 0:N-1
#     for n = 0:N-1
#         (a11,a12,a21,a22) = get_rotation_vector(mass, X)
#         corr[2*m+2,2*n+2] = sum([exp(-im*k*(m-n))*a21[k_index]*conj(a21[k_index]) for (k_index,k) in enumerate(X)])/N
#         corr[2*m+1,2*n+2] = sum([exp(-im*k*(m-n))*a22[k_index]*conj(a21[k_index]) for (k_index,k) in enumerate(X)])/N
#         corr[2*m+2,2*n+1] = sum([exp(-im*k*(m-n))*a21[k_index]*conj(a22[k_index]) for (k_index,k) in enumerate(X)])/N
#         corr[2*m+1,2*n+1] = sum([exp(-im*k*(m-n))*a22[k_index]*conj(a22[k_index]) for (k_index,k) in enumerate(X)])/N
#     end
# end

corr_ana = zeros(ComplexF64, 2*N, 2*N)
for m = 0:N-1
    for n = 0:N-1
        (a11,a12,a21,a22) = get_rotation_vector(mass, X)
        corr_ana[2*m+2,2*n+2] = sum([exp(-im*k*(m-n))*a12[k_index]*conj(a12[k_index]) for (k_index,k) in enumerate(X)])/N
        corr_ana[2*m+1,2*n+2] = sum([exp(-im*k*(m-n))*a22[k_index]*conj(a12[k_index]) for (k_index,k) in enumerate(X)])/N
        corr_ana[2*m+2,2*n+1] = sum([exp(-im*k*(m-n))*a12[k_index]*conj(a22[k_index]) for (k_index,k) in enumerate(X)])/N
        corr_ana[2*m+1,2*n+1] = sum([exp(-im*k*(m-n))*a22[k_index]*conj(a22[k_index]) for (k_index,k) in enumerate(X)])/N
    end
end

corr_0 = corr_expected[21:22,21:22]
(a11,a12,a21,a22) = get_rotation_vector(mass, X)
println(sum([exp(-im*k*(m-n))*a12[k_index]*conj(a12[k_index]) for (k_index,k) in enumerate(X)])/N)
println(sum([exp(-im*k*(m-n))*a22[k_index]*conj(a12[k_index]) for (k_index,k) in enumerate(X)])/N)
println(sum([exp(-im*k*(m-n))*a12[k_index]*conj(a22[k_index]) for (k_index,k) in enumerate(X)])/N)
println(sum([exp(-im*k*(m-n))*a22[k_index]*conj(a22[k_index]) for (k_index,k) in enumerate(X)])/N)

(V₊,V₋) = V_matrix_pos_neg_energy(X, mass)
# (V₊,V₋) = V_matrix(X, mass)
occ_matrix = adjoint(V₊)*corr*(V₊)

occ_matrix_expected = zeros(Float64, 2*N, 2*N)
for i = 1:N
    if X[i] < 0.0
        occ_matrix_expected[i,i] = 0.0
        occ_matrix_expected[N+i,N+i] = 1.0
    else
        occ_matrix_expected[i,i] = 1.0
        occ_matrix_expected[N+i,N+i] = 0.0
    end
end

V = hcat(V₊, V₋)
corr_expected = V * occ_matrix_expected * adjoint(V)

occ_p = adjoint(V₊) * corr_expected * V₊
occ_m = adjoint(V₋) * corr_expected * V₋

(F, PN) = V_matrix_unpermuted(X, mass)

diag_expected = zeros(ComplexF64, 2*N, 2*N)
for (k_index,k) in enumerate(X)
    diag_expected[2*k_index-1,2*k_index-1] = mass
    diag_expected[2*k_index-1,2*k_index] = (-im/2)*(1-exp(im*k))
    diag_expected[2*k_index,2*k_index-1] = (im/2)*(1-exp(-im*k))
    diag_expected[2*k_index,2*k_index] = -mass
end
corr_energy_expected = F * diag_expected * adjoint(F)

M = get_2D_matrix(1, diag)


# (V₊,V₋) = V_matrix_pos_neg_energy(X, mass)
(V₊,V₋) = V_matrix(X, mass)

positive = true

if positive
    occ_matrix_energy = adjoint(V₊)*(corr_energy_expected)*(V₊)
    occ_matrix = adjoint(V₊)*corr_expected*(V₊)
else
    occ_matrix_energy = adjoint(V₋)*(corr_energy_expected)*(V₋)
    occ_matrix = adjoint(V₋)*corr_expected*(V₋)
end

# occ_matrix_energy = adjoint(PN) * diag_expected * PN


# positive = false
# if positive
#     corr_expected = V₊ * occ_matrix_expected * adjoint(V₊)
# else
#     corr_expected = V₋ * occ_matrix_expected * adjoint(V₋)
# end

occ = zeros(Float64, N)
occ_energy = zeros(Float64, N)
for (i,k₀) in enumerate(X)
    array = gaussian_array(X, k₀, σ, x₀)
    occupation_number = ((array)*occ_matrix*adjoint(array))
    occupation_number_energy = (array)*(occ_matrix_energy)*adjoint(array) / ((array)*occ_matrix*adjoint(array))

    occ[i] = real(occupation_number)
    occ_energy[i] = real(occupation_number_energy)
end

plt = scatter(X, occ, label = "data")
display(plt)

theoretical_energies = [sqrt(mass^2+sin(k/2)^2) for k in X]
plt = scatter(X, occ_energy, label = "data")
scatter!(X, theoretical_energies)
display(plt)

break

function energy(X, mass)
    println("mass = $(mass)")
    N = length(X)
    corr = zeros(ComplexF64, 2*N, 2*N)
    (a11,a12,a21,a22) = get_rotation_vector(mass, X)
    for m = 0:N-1
        for n = 0:N-1
            term_ee = 0.0
            term_oe = 0.0
            term_eo = 0.0
            term_oo = 0.0
            count = 0
            for (k_index,k) in enumerate(X)
                for (l_index,l) in enumerate(X)
                    if k_index != l_index
                        factor = 1.0# * exp(-im*(k/2)*(m-n))
                    else
                        factor = 0.0
                    end
                    if (m == n) && (0 == 1)
                        term1 = factor*exp(-im*k*(m-n))*(
                            (im/2)*(1-exp(-im*k))*a21[k_index]*a22[l_index]*conj(a21[l_index])*conj(a21[k_index])
                            -(im/2)*(1-exp(im*k))*a21[k_index]*conj(a22[l_index])*a21[l_index]*conj(a21[k_index]))
                        term2 = factor*exp(-im*k*(m-n)) * (
                                mass*a21[k_index]*a21[l_index]*conj(a21[l_index])*conj(a21[k_index])
                                -mass*a21[k_index]*a22[l_index]*conj(a22[l_index])*conj(a21[k_index])
                            )
                        term_tot = factor*exp(-im*k*(m-n))*(
                            (im/2)*(1-exp(-im*k))*a21[k_index]*a22[l_index]*conj(a21[l_index])*conj(a21[k_index])
                            -(im/2)*(1-exp(im*k))*a21[k_index]*conj(a22[l_index])*a21[l_index]*conj(a21[k_index]))
                            + factor*exp(-im*k*(m-n)) * (
                                mass*a21[k_index]*a21[l_index]*conj(a21[l_index])*conj(a21[k_index])
                                -mass*a21[k_index]*a22[l_index]*conj(a22[l_index])*conj(a21[k_index])
                            )
        
                        println("mass = $(mass)")
                        println("factor = $(factor)")
                        println("product = $(a21[k_index]*a21[l_index]*conj(a21[l_index])*conj(a21[k_index]))")
                        println("product2 = $(a21[k_index]*a22[l_index]*conj(a22[l_index])*conj(a21[k_index]))")
                        println("term1 = $(term1)")
                        println("term2 = $(term2)")
                        println("term_tot = $(term_tot)")
                    end

                    term_ee += factor*exp(-im*k*(m-n))*(
                        +(im/2)*(1-exp(-im*k))*a21[k_index]*a22[l_index]*conj(a21[l_index])*conj(a21[k_index])
                        -(im/2)*(1-exp(im*k))*a21[k_index]*conj(a22[l_index])*a21[l_index]*conj(a21[k_index]))
                    term_ee += factor*exp(-im*k*(m-n)) * (
                            mass*a21[k_index]*a21[l_index]*conj(a21[l_index])*conj(a21[k_index])
                            -mass*a21[k_index]*a22[l_index]*conj(a22[l_index])*conj(a21[k_index])
                        )
                    term_oe += factor*exp(-im*k*(m-n))*(
                        +(im/2)*(1-exp(-im*k))*a22[k_index]*a22[l_index]*conj(a21[l_index])*conj(a21[k_index])
                        -(im/2)*(1-exp(im*k))*a22[k_index]*conj(a22[l_index])*a21[l_index]*conj(a21[k_index]))
                    term_oe += factor*exp(-im*k*(m-n)) * (
                            mass*a22[k_index]*a21[l_index]*conj(a21[l_index])*conj(a21[k_index])
                            -mass*a22[k_index]*a22[l_index]*conj(a22[l_index])*conj(a21[k_index])
                        )
                    term_eo += factor*exp(-im*k*(m-n))*(
                        +(im/2)*(1-exp(-im*k))*a21[k_index]*a22[l_index]*conj(a21[l_index])*conj(a22[k_index])
                        -(im/2)*(1-exp(im*k))*a21[k_index]*conj(a22[l_index])*a21[l_index]*conj(a22[k_index]))
                    term_eo += factor*exp(-im*k*(m-n)) * (
                            mass*a21[k_index]*a21[l_index]*conj(a21[l_index])*conj(a22[k_index])
                            -mass*a21[k_index]*a22[l_index]*conj(a22[l_index])*conj(a22[k_index])
                        )
                    term_oo += factor*exp(-im*k*(m-n))*(
                        +(im/2)*(1-exp(-im*k))*a22[k_index]*a22[l_index]*conj(a21[l_index])*conj(a22[k_index])
                        -(im/2)*(1-exp(im*k))*a22[k_index]*conj(a22[l_index])*a21[l_index]*conj(a22[k_index]))
                    term_oo += factor*exp(-im*k*(m-n)) * (
                            mass*a22[k_index]*a21[l_index]*conj(a21[l_index])*conj(a22[k_index])
                            -mass*a22[k_index]*a22[l_index]*conj(a22[l_index])*conj(a22[k_index])
                        )
                end
            end
            if (m == n) && (0 == 1)
                println("count = $(count)")
                println("ee = $term_ee")
                println("oe = $term_oe")
                println("eo = $term_eo")
                println("oo = $term_oo")
            end
            corr[2*m+2,2*n+2] = term_ee/(N*(N-1))
            corr[2*m+1,2*n+2] = term_oe/(N*(N-1))
            corr[2*m+2,2*n+1] = term_eo/(N*(N-1))
            corr[2*m+1,2*n+1] = term_oo/(N*(N-1))
        end
    end
    return corr
end

function energy_new(X, mass)
    N = length(X)
    corr = zeros(ComplexF64, 2*N, 2*N)
    (a11,a12,a21,a22) = get_rotation_vector(mass, X)
    for m = 0:N-1
        for n = 0:N-1
            term_ee = 0.0
            term_oe = 0.0
            term_eo = 0.0
            term_oo = 0.0
            count = 0
            for (k_index,k) in enumerate(X)
                for (l_index,l) in enumerate(X)
                    if (k_index == l_index)
                        factor = 0.0
                    else
                        factor = 1.0
                    end
                    term_ee += factor*exp(-im*k*(m-n))*(im/2)*(1-exp(-im*l))*(
                        a12[k_index]*a22[l_index]*conj(a12[l_index])*conj(a12[k_index]))
                    term_ee += factor*exp(-im*k*(m-n)) * (
                            mass*a12[k_index]*a12[l_index]*conj(a12[l_index])*conj(a12[k_index])
                            -mass*a12[k_index]*a22[l_index]*conj(a22[l_index])*conj(a12[k_index])
                        )
                    term_oe += factor*exp(-im*k*(m-n))*(im/2)*(1-exp(-im*l))*(
                        a22[k_index]*a22[l_index]*conj(a12[l_index])*conj(a12[k_index]))
                    term_oe += factor*exp(-im*k*(m-n)) * (
                        mass*a22[k_index]*a12[l_index]*conj(a12[l_index])*conj(a12[k_index])
                        -mass*a22[k_index]*a22[l_index]*conj(a22[l_index])*conj(a12[k_index])
                        )
                    term_eo += factor*exp(-im*k*(m-n))*(im/2)*(1-exp(-im*l))*(
                        a12[k_index]*a22[l_index]*conj(a12[l_index])*conj(a22[k_index]))
                    term_eo += factor*exp(-im*k*(m-n)) * (
                        mass*a12[k_index]*a12[l_index]*conj(a12[l_index])*conj(a22[k_index])
                        -mass*a12[k_index]*a22[l_index]*conj(a22[l_index])*conj(a22[k_index])
                        )
                    term_oo += factor*exp(-im*k*(m-n))*(im/2)*(1-exp(-im*l))*(
                        a22[k_index]*a22[l_index]*conj(a12[l_index])*conj(a22[k_index]))
                    term_oo += factor*exp(-im*k*(m-n)) * (
                        mass*a22[k_index]*a12[l_index]*conj(a12[l_index])*conj(a22[k_index])
                        -mass*a22[k_index]*a22[l_index]*conj(a22[l_index])*conj(a22[k_index])
                        )
                end
            end
            if (m == n) && (0 == 1)
                println("count = $(count)")
                println("ee = $term_ee")
                println("oe = $term_oe")
                println("eo = $term_eo")
                println("oo = $term_oo")
            end
            corr[2*m+2,2*n+2] = term_ee/(N*(N-1))
            corr[2*m+1,2*n+2] = term_oe/(N*(N-1))
            corr[2*m+2,2*n+1] = term_eo/(N*(N-1))
            corr[2*m+1,2*n+1] = term_oo/(N*(N-1))
        end
    end
    return corr
end

corr_energy = energy_new(X, mass)

corr_energy = corr_energy + adjoint(corr_energy)

σ = 0.1
x₀ = div(N,2)

(V₊,V₋) = V_matrix_pos_neg_energy(X, mass)
(F, PN) = V_matrix_unpermuted(X, mass)
# V = F * PN

diag = adjoint(F) * corr_energy * F

diag_expected = zeros(ComplexF64, 2*N, 2*N)
for (k_index,k) in enumerate(X)
    diag_expected[2*k_index-1,2*k_index-1] = mass
    diag_expected[2*k_index-1,2*k_index] = (-im/2)*(1-exp(im*k))
    diag_expected[2*k_index,2*k_index-1] = (im/2)*(1-exp(-im*k))
    diag_expected[2*k_index,2*k_index] = -mass
end
corr_energy_expected = F * diag_expected * adjoint(F)

M = get_2D_matrix(1, diag)

positive = false

if positive
    occ_matrix_energy = adjoint(V₊)*(corr_energy)*(V₊)
    occ_matrix = adjoint(V₊)*corr*(V₊)
else
    occ_matrix_energy = adjoint(V₋)*(corr_energy)*(V₋)
    occ_matrix = adjoint(V₋)*corr*(V₋)
end

occ_matrix_energy = adjoint(PN) * diag_expected * PN

theoretical_energies = [sqrt(mass^2+sin(k/2)^2) for k in X]

occ = zeros(Float64, N)
occ_energy = zeros(Float64, N)
for (i,k₀) in enumerate(X)
    array = gaussian_array(X, k₀, σ, x₀)
    occupation_number = ((array)*occ_matrix*adjoint(array))
    println("i = $(i), k = $(k₀)")
    println(array)
    println(occupation_number)
    # println((array)*(occ_matrix_energy)*adjoint(array) )
    # occupation_number_energy = (array)*(occ_matrix_energy)*adjoint(array) / ((array)*occ_matrix*adjoint(array))
    if (abs(imag(occupation_number)) > 1e-2)
        println("Warning, complex number for occupation number: $(occupation_number)")
    end
    occ[i] = real(occupation_number)
    # occ_energy[i] = real(occupation_number_energy)
end


plt = scatter(X, occ, label = "data")
display(plt)
# plt = scatter(X, (occ.*2).+mass, label = "data")
# plt = scatter(X, occ_energy, label = "data")
# scatter!(X, theoretical_energies, label = "theory")
# display(plt)

(theoretical_energies .-minimum(theoretical_energies))./(occ_energy .-minimum(occ_energy))