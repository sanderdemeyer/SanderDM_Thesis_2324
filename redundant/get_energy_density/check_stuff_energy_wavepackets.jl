using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using LaTeXStrings
using Random

m = 0.3
Delta_g = 0.0
truncation = 2.5

@load "SanderDM_Thesis_2324/gs_mps_trunc_$(truncation)_mass_$(m)_v_0.0_Delta_g_$(Delta_g)" mps

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
Plus_space = U1Space(1 => 1)
Min_space = U1Space(-1 => 1)
trivspace = U1Space(0 => 1)
S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], trivspace ⊗ pspace, pspace ⊗ Plus_space)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace ⊗ trivspace)
S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)
S⁺2 = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Min_space ⊗ pspace, pspace ⊗ trivspace)
S⁻2 = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], trivspace ⊗ pspace, pspace ⊗ Min_space)    

unit4 = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], trivspace ⊗ pspace, pspace ⊗ trivspace)
unit2 = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], pspace, pspace)

H_unit = @mpoham sum((unit2){i} for i in vertices(InfiniteChain(2)))
H = get_thirring_hamiltonian_symmetric(m, Delta_g, v)

corr_bigger = correlator_tf(mps, H_unit, unit4, unit4, unit2, 1, 2*N+2)
corr_bigger = correlator_tf(mps, H_unit, unit4, unit4, unit2, 2, 2*N+2)

corr_bigger = correlator_tf(mps, H_unit, unit4, unit4, unit2, 1, 2*N+2)
corr_bigger = correlator_tf(mps, H, unit4, unit4, unit2, 2, 2*N+2)

corr_bigger = correlator_tf_april(mps, H, unit4, unit4, unit2, 1, 2*N+2)


# from expectation_value:
    #calculate energy density


st = mps
envs = environments(mps, H)
len = length(st)
ens = PeriodicArray(zeros(scalartype(st.AR[1]), len))
for i in 1:len
    util = MPSKit.fill_data!(similar(st.AL[1], space(envs.lw[H.odim, i + 1], 2)), one)
    println(util)
    # for j in (H.odim):-1:1
    for j in 1:(H.odim)
        apl = leftenv(envs, i, st)[j] *
                TransferMatrix(st.AL[i], H[i][j, H.odim], st.AL[i])
        ens[i] += @plansor apl[1 2; 3] * r_LL(st, i)[3; 1] * conj(util[2])
    end
end
println(ens)

st = mps
envs = environments(mps, H);
len = length(st);
ens = PeriodicArray(zeros(scalartype(st.AR[1]), len));
for i in 1:len
    util = MPSKit.fill_data!(similar(st.AL[1], space(envs.lw[H.odim, i + 1], 2)), one);
    println(util)
    # for j in (H.odim):-1:1
    for j in 1:(H.odim)
        left = leftenv(envs,i,st)
        apl = left[j] *
                TransferMatrix(st.AL[i], H[i][j, H.odim], st.AL[i])
        ens[i] += @plansor apl[1 2; 3] * r_LL(st, i)[3; 1] * conj(util[2])
    end
end
println(ens)

i = 2
envs = environments(mps, H);
left1 = leftenv(envs,i,st);
left2 = leftenv(envs,i-2,st) * TransferMatrix(st.AL[i-2], H[i-2], st.AL[i-2]) * TransferMatrix(st.AL[i-1], H[i-1], st.AL[i-1]);