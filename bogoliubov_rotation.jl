using LinearAlgebra
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Base
using Plots

include("get_thirring_hamiltonian.jl")

mass = 0.3
v = 0.0
Delta_g = 0.0
truncation = 4.0

@load "SanderDM_Thesis_2324/gs_mps_wo_symmetries_trunc_$(truncation)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)" mps
@load "Dispersion_asymm_Delta_m_$(mass)_delta_g_$(Delta_g)_v_$(v)_irrep_1" gs_energy bounds energies Bs

k_values = LinRange(-bounds, bounds,length(Bs))

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

