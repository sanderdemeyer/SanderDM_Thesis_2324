using LinearAlgebra
using JLD2
using TensorKit
using KrylovKit
using MPSKitModels, TensorKit, MPSKit
using Base
using Plots

include("get_thirring_hamiltonian.jl")
include("get_thirring_hamiltonian_symmetric.jl")
include("get_X_tensors.jl")

# struct SumTransferMatrix{A,B}
#     tms::Vector

#     function SumTransferMatrix(tms::Vector)
#         # Additional validation or setup can be performed here
#         new(tms::Vector)  # Create and return a new instance of Point
#     end

# end

# function SumTransferMatrix(tms_new::Vector)
#     tms = tms_new
# end


# # TransferMatrix acting as a function
# function (stm::SumTransferMatrix)(vec)
#     results = []
#     for d = stm.tms
#         if d.isflipped
#             push!(results,transfer_left(vec, d.middle, d.above, d.below))
#         else
#             push!(results,transfer_right(vec, d.middle, d.above, d.below))
#         end
#     end
#     return sum(results)
# end;

function one_minus_tm(v, tm; factor = 1.0)
    return v - factor*tm(v)
end

function one_minus_tm_new(v, AL1, AL2, AR1, AR2)
    @tensor extra_term[-1 -2; -3] := AL1[-1 1; 2] * AL2[2 4; 5] * conj(AR1[-3 1; 3]) * conj(AR2[3 4; 6]) * v[5 -2; 6]
    return v - extra_term
    # return v - tm(v)
    # return v - transfer_left(tm, v)
end

function transpose_tail(t::AbstractTensorMap) # make TensorMap{S,1,N₁+N₂-1}
    I1 = TensorKit.codomainind(t)
    I2 = TensorKit.domainind(t)
    return transpose(t, ((I1[1],), (I2..., reverse(Base.tail(I1))...)))
end

N = 500
X = [(2*pi)/N*i - pi for i = 0:N-1]


mass = 0.3
v = 0.0
Delta_g = 0.0
truncation = 3.0

@load "SanderDM_Thesis_2324/gs_mps_trunc_$(truncation)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)" mps
@load "SanderDM_Thesis_2324/Dispersion_Delta_m_$(mass)_delta_g_$(Delta_g)_v_$(v)_trunc_$(truncation)_all_sectors" gs_energy energies Bs anti_energies anti_Bs zero_energies zero_Bs bounds_k
@load "operators_for_occupation_number" S⁺ S⁻ S_z_symm

@tensor Snew[-1 -2; -3 -4] := S⁻[-4 -2; -3 -1]

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
Plus_space = U1Space(1 => 1)
Min_space = U1Space(-1 => 1)
Triv_space = U1Space(0 => 1)
S⁺_old = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Triv_space ⊗ pspace, pspace ⊗ Plus_space)
S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Triv_space ⊗ pspace, pspace ⊗ Min_space')
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace ⊗ Triv_space)


function create_excitation_OLD(mps, a, b, qp_state; momentum = 0.0)
    # @load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
    # @tensor Snew[-1 -2; -3 -4] := S⁻[-4 -2; -3 -1]
    Plus_space = U1Space(1 => 1)
    Min_space = U1Space(-1 => 1)
    Triv_space = U1Space(0 => 1)
    # S⁺_old = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Triv_space ⊗ pspace, pspace ⊗ Plus_space)
    S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Triv_space ⊗ pspace, pspace ⊗ Min_space')
    # S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace ⊗ Triv_space)
    
    AL_new = copy(mps.AL)
    AC_new = copy(mps.AC)
    AR_new = copy(mps.AR)
    for w = 1:length(mps)
        @tensor new_tensor_AL[-1 -2; -3] := mps.AL[w][-1 1; -3] * (-2*im*S_z_symm)[-2; 1]
        AL_new[w] = new_tensor_AL
        @tensor new_tensor_AR[-1 -2; -3] := mps.AR[w][-1 1; -3] * (-2*im*S_z_symm)[-2; 1]
        AR_new[w] = new_tensor_AR
        @tensor new_tensor_AC[-1 -2; -3] := mps.AC[w][-1 1; -3] * (-2*im*S_z_symm)[-2; 1]
        AC_new[w] = new_tensor_AC
    end

    left_gs = InfiniteMPS(AL_new, AR_new, mps.CR, AC_new)
    right_gs = mps

    VRs = [adjoint(leftnull(adjoint(v))) for v in transpose_tail.(right_gs.AR)]

    @tensor should_be_zero[-1; -2] := (VRs[1][-1 1; 2]) * conj(mps.AR[1][-2 2; 1])
    @assert norm(should_be_zero) < 1e-10

    S_space = Tensor(ones, space(S⁺)[1]) # this and the subsequent two lines were Snew instead of S⁺

    @tensor B1[-1 -2; -3 -4] := (a) * mps.AC[1][-1 1; -4] * S⁺[2 -2; 1 -3] * conj(S_space[2])
    @tensor B2[-1 -2; -3 -4] := (b) * mps.AC[2][-1 1; -4] * S⁺[2 -2; 1 -3] * conj(S_space[2])

    t1 = TransferMatrix(mps.AL[1], mps.AR[1])
    t2 = TransferMatrix(mps.AL[2], mps.AR[2])
    # t1 = TensorKit.flip(t1)
    # t2 = TensorKit.flip(t2)
    T12 = t1*t2
    T21 = t2*t1

    @tensor RHS12[-1 -3; -2] := B1[-1 1; -3 2] * conj(mps.AR[1][-2 1; 2]) + exp(im*momentum) * mps.AL[1][-1 1; 2] * B2[2 3; -3 4] * conj(mps.AR[1][-2 1; 5]) * conj(mps.AR[2][5 3; 4])
    @tensor RHS21[-1 -3; -2] := B2[-1 1; -3 2] * conj(mps.AR[2][-2 1; 2]) + mps.AL[2][-1 1; 2] * B1[2 3; -3 4] * conj(mps.AR[2][-2 1; 5]) * conj(mps.AR[1][5 3; 4])
    # @tensor RHS12[-3 -2; -1] := B1[-1 1; -3 2] * conj(mps.AR[1][-2 1; 2]) + mps.AL[1][-1 1; 2] * B2[2 3; -3 4] * conj(mps.AR[1][-2 1; 5]) * conj(mps.AR[2][5 3; 4])
    # @tensor RHS21[-3 -2; -1] := B2[-1 1; -3 2] * conj(mps.AR[2][-2 1; 2]) + mps.AL[2][-1 1; 2] * B1[2 3; -3 4] * conj(mps.AR[2][-2 1; 5]) * conj(mps.AR[1][5 3; 4])

    XL, convhist_L = linsolve(x -> one_minus_tm(x, T21; factor = exp(im*momentum)), RHS21, maxiter=1000, tol = 1e-14)
    XR, convhist_R = linsolve(x -> one_minus_tm(x, T12; factor = exp(im*momentum)), RHS12, maxiter=1000, tol = 1e-14)
    # XL, convhist_L = linsolve(T21, RHS21, maxiter=1000, tol = 1e-14)#, a0 = 1, a1 = -exp(im*momentum))
    # XR, convhist_R = linsolve(T12, RHS12, maxiter=1000, tol = 1e-14)#, a0 = 1, a1 = -exp(im*momentum))

    # XL, convhist_L = linsolve(x -> one_minus_tm_new(x, mps.AL[2], mps.AL[1], mps.AR[2], mps.AR[1]), RHS21, maxiter=1000, tol = 1e-14)
    # XR, convhist_R = linsolve(x -> one_minus_tm_new(x, mps.AL[1], mps.AL[2], mps.AR[1], mps.AR[2]), RHS12, maxiter=1000, tol = 1e-14)

    @tensor Bnew1[-1 -2; -3 -4] := B1[-1 -2; -3 -4] + exp(im*momentum) * mps.AL[1][-1 -2; 1] * XL[1 -3; -4] - XR[-1 -3; 1] * mps.AR[1][1 -2; -4]
    @tensor Bnew2[-1 -2; -3 -4] := B2[-1 -2; -3 -4] + mps.AL[2][-1 -2; 1] * XR[1 -3; -4] - XL[-1 -3; 1] * mps.AR[2][1 -2; -4] 

    @tensor check_B1[-1 -2; -3] := Bnew1[-1 1; -3 2] * conj(mps.AR[1][-2 1; 2])
    @tensor check_B2[-1 -2; -3] := Bnew2[-1 1; -3 2] * conj(mps.AR[2][-2 1; 2])

    @assert norm(check_B1) < 1e-10
    @assert norm(check_B2) < 1e-10

    @tensor X1[-1; -3 -2] := (Bnew1[-1 1; -3 2]) * conj(VRs[1][-2 2; 1])
    @tensor X2[-1; -3 -2] := (Bnew2[-1 1; -3 2]) * conj(VRs[2][-2 2; 1])
    # @tensor X1[-1; -2 -3] := (Bnew1[-1 1; -3 2]) * conj(VRs[1][-2 2; 1])
    # @tensor X2[-1; -2 -3] := (Bnew2[-1 1; -3 2]) * conj(VRs[2][-2 2; 1])
    Xs = [X1, X2]

    @tensor B1_again[-1 -2; -3 -4] := X1[-1; -3 1] * (VRs[1][1 -4; -2])
    @tensor B2_again[-1 -2; -3 -4] := X2[-1; -3 1] * (VRs[2][1 -4; -2])

    @assert norm(B1_again-Bnew1) < 1e-10
    @assert norm(B2_again-Bnew2) < 1e-10

    qp_right = RightGaugedQP(left_gs, right_gs, Xs, VRs, momentum)

    normalize!(qp_right)
    @assert abs(1-norm(qp_right)) < 1e-10

    return sum(dot.(qp_right.Xs, qp_state.Xs))
    return dot(qp_right, qp_state)
end

function create_excitation(mps, a, b, qp_state; momentum = 0.0)
    Plus_space = U1Space(1 => 1)
    Min_space = U1Space(-1 => 1)
    Triv_space = U1Space(0 => 1)
    # S⁺_old = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Triv_space ⊗ pspace, pspace ⊗ Plus_space)
    S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Triv_space ⊗ pspace, pspace ⊗ Min_space')
    # S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace ⊗ Triv_space)
    
    X_tensors = get_X_tensors(mps.AL)

    left_gs = mps
    right_gs = mps

    VRs = [adjoint(leftnull(adjoint(v))) for v in transpose_tail.(right_gs.AR)]

    @tensor should_be_zero[-1; -2] := (VRs[1][-1 1; 2]) * conj(mps.AR[1][-2 2; 1])
    @assert norm(should_be_zero) < 1e-10

    S_space = Tensor(ones, space(S⁺)[1]) # this and the subsequent two lines were Snew instead of S⁺

    # @tensor B1[-1 -2; -3 -4] := (a) * X_tensors[1][-1; 3] * mps.AC[1][3 1; -4] * S⁺[2 -2; 1 -3] * conj(S_space[2])
    # @tensor B2[-1 -2; -3 -4] := (b) * X_tensors[2][-1; 3] * mps.AC[2][3 1; -4] * S⁺[2 -2; 1 -3] * conj(S_space[2])
    @tensor B1[-1 -2; -3 -4] := X_tensors[1][-1; 3] * mps.AC[1][3 1; -4] * S⁺[2 -2; 1 -3] * conj(S_space[2])
    @tensor B2[-1 -2; -3 -4] := X_tensors[2][-1; 3] * mps.AC[2][3 1; -4] * S⁺[2 -2; 1 -3] * conj(S_space[2])

    t1 = TransferMatrix(mps.AL[1], mps.AR[1])
    t2 = TransferMatrix(mps.AL[2], mps.AR[2])
    # t1 = TensorKit.flip(t1)
    # t2 = TensorKit.flip(t2)
    T12 = t1*t2
    T21 = t2*t1

    @tensor RHS12[-1 -3; -2] := B1[-1 1; -3 2] * conj(mps.AR[1][-2 1; 2]) + exp(im*momentum) * mps.AL[1][-1 1; 2] * B2[2 3; -3 4] * conj(mps.AR[1][-2 1; 5]) * conj(mps.AR[2][5 3; 4])
    @tensor RHS21[-1 -3; -2] := B2[-1 1; -3 2] * conj(mps.AR[2][-2 1; 2]) + mps.AL[2][-1 1; 2] * B1[2 3; -3 4] * conj(mps.AR[2][-2 1; 5]) * conj(mps.AR[1][5 3; 4])
    # @tensor RHS12[-3 -2; -1] := B1[-1 1; -3 2] * conj(mps.AR[1][-2 1; 2]) + mps.AL[1][-1 1; 2] * B2[2 3; -3 4] * conj(mps.AR[1][-2 1; 5]) * conj(mps.AR[2][5 3; 4])
    # @tensor RHS21[-3 -2; -1] := B2[-1 1; -3 2] * conj(mps.AR[2][-2 1; 2]) + mps.AL[2][-1 1; 2] * B1[2 3; -3 4] * conj(mps.AR[2][-2 1; 5]) * conj(mps.AR[1][5 3; 4])

    XL, convhist_L = linsolve(x -> one_minus_tm(x, T21; factor = exp(im*momentum)), RHS21, maxiter=1000, tol = 1e-14)
    XR, convhist_R = linsolve(x -> one_minus_tm(x, T12; factor = exp(im*momentum)), RHS12, maxiter=1000, tol = 1e-14)
    # XL, convhist_L = linsolve(T21, RHS21, maxiter=1000, tol = 1e-14)#, a0 = 1, a1 = -exp(im*momentum))
    # XR, convhist_R = linsolve(T12, RHS12, maxiter=1000, tol = 1e-14)#, a0 = 1, a1 = -exp(im*momentum))

    # XL, convhist_L = linsolve(x -> one_minus_tm_new(x, mps.AL[2], mps.AL[1], mps.AR[2], mps.AR[1]), RHS21, maxiter=1000, tol = 1e-14)
    # XR, convhist_R = linsolve(x -> one_minus_tm_new(x, mps.AL[1], mps.AL[2], mps.AR[1], mps.AR[2]), RHS12, maxiter=1000, tol = 1e-14)

    @tensor Bnew1[-1 -2; -3 -4] := B1[-1 -2; -3 -4] + exp(im*momentum) * mps.AL[1][-1 -2; 1] * XL[1 -3; -4] - XR[-1 -3; 1] * mps.AR[1][1 -2; -4]
    @tensor Bnew2[-1 -2; -3 -4] := B2[-1 -2; -3 -4] + mps.AL[2][-1 -2; 1] * XR[1 -3; -4] - XL[-1 -3; 1] * mps.AR[2][1 -2; -4] 

    @tensor check_B1[-1 -2; -3] := Bnew1[-1 1; -3 2] * conj(mps.AR[1][-2 1; 2])
    @tensor check_B2[-1 -2; -3] := Bnew2[-1 1; -3 2] * conj(mps.AR[2][-2 1; 2])

    @assert norm(check_B1) < 1e-10
    @assert norm(check_B2) < 1e-10

    @tensor X1[-1; -3 -2] := (Bnew1[-1 1; -3 2]) * conj(VRs[1][-2 2; 1])
    @tensor X2[-1; -3 -2] := (Bnew2[-1 1; -3 2]) * conj(VRs[2][-2 2; 1])
    # @tensor X1[-1; -2 -3] := (Bnew1[-1 1; -3 2]) * conj(VRs[1][-2 2; 1])
    # @tensor X2[-1; -2 -3] := (Bnew2[-1 1; -3 2]) * conj(VRs[2][-2 2; 1])
    Xs = [X1, X2]

    @tensor B1_again[-1 -2; -3 -4] := X1[-1; -3 1] * (VRs[1][1 -4; -2])
    @tensor B2_again[-1 -2; -3 -4] := X2[-1; -3 1] * (VRs[2][1 -4; -2])

    @assert norm(B1_again-Bnew1) < 1e-10
    @assert norm(B2_again-Bnew2) < 1e-10

    qp_right = RightGaugedQP(left_gs, right_gs, [a*X1, b*X2], VRs, momentum)

    normalize!(qp_right)
    @assert abs(1-norm(qp_right)) < 1e-10

    mass = 0.3
    v = 0.0
    Delta_g = 0.0
    H = get_thirring_hamiltonian_symmetric(mass, Delta_g, v)
    E_orig = expectation_value(qp_state, H)
    E_new = expectation_value(qp_right, H)

    println("E_orig = $(E_orig)")
    println("E_new = $(E_new)")

    return sum(dot.(qp_right.Xs, qp_state.Xs))

    return dot(qp_right, qp_state)
end

function get_excitation(mps, a, b; momentum = 0.0, new = true)
    Min_space = U1Space(-1 => 1)
    Triv_space = U1Space(0 => 1)
    S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Triv_space ⊗ pspace, pspace ⊗ Min_space')
    
    X_tensors = get_X_tensors(mps.AL)

    if new
        left_gs = mps
    else
        AL_new = copy(mps.AL)
        AC_new = copy(mps.AC)
        AR_new = copy(mps.AR)
        for w = 1:length(mps)
            @tensor new_tensor_AL[-1 -2; -3] := mps.AL[w][-1 1; -3] * (-2*im*S_z_symm)[-2; 1]
            AL_new[w] = new_tensor_AL
            @tensor new_tensor_AR[-1 -2; -3] := mps.AR[w][-1 1; -3] * (-2*im*S_z_symm)[-2; 1]
            AR_new[w] = new_tensor_AR
            @tensor new_tensor_AC[-1 -2; -3] := mps.AC[w][-1 1; -3] * (-2*im*S_z_symm)[-2; 1]
            AC_new[w] = new_tensor_AC
        end
        left_gs = InfiniteMPS(AL_new, AR_new, mps.CR, AC_new)
    end    

    right_gs = mps

    VRs = [adjoint(leftnull(adjoint(v))) for v in transpose_tail.(right_gs.AR)]

    @tensor should_be_zero[-1; -2] := (VRs[1][-1 1; 2]) * conj(mps.AR[1][-2 2; 1])
    @assert norm(should_be_zero) < 1e-10

    S_space = Tensor(ones, space(S⁺)[1]) # this and the subsequent two lines were Snew instead of S⁺

    if new
        @tensor B1[-1 -2; -3 -4] := (a) * X_tensors[1][-1; 3] * mps.AC[1][3 1; -4] * S⁺[2 -2; 1 -3] * conj(S_space[2])
        @tensor B2[-1 -2; -3 -4] := (b) * X_tensors[2][-1; 3] * mps.AC[2][3 1; -4] * S⁺[2 -2; 1 -3] * conj(S_space[2])
    else
        @tensor B1[-1 -2; -3 -4] := (a) * mps.AC[1][-1 1; -4] * S⁺[2 -2; 1 -3] * conj(S_space[2])
        @tensor B2[-1 -2; -3 -4] := (b) * mps.AC[2][-1 1; -4] * S⁺[2 -2; 1 -3] * conj(S_space[2])
    end    

    t1 = TransferMatrix(mps.AL[1], mps.AR[1])
    t2 = TransferMatrix(mps.AL[2], mps.AR[2])
    T12 = t1*t2
    T21 = t2*t1

    @tensor RHS12[-1 -3; -2] := B1[-1 1; -3 2] * conj(mps.AR[1][-2 1; 2]) + exp(im*momentum) * mps.AL[1][-1 1; 2] * B2[2 3; -3 4] * conj(mps.AR[1][-2 1; 5]) * conj(mps.AR[2][5 3; 4])
    @tensor RHS21[-1 -3; -2] := B2[-1 1; -3 2] * conj(mps.AR[2][-2 1; 2]) + mps.AL[2][-1 1; 2] * B1[2 3; -3 4] * conj(mps.AR[2][-2 1; 5]) * conj(mps.AR[1][5 3; 4])

    XL, convhist_L = linsolve(x -> one_minus_tm(x, T21; factor = exp(im*momentum)), RHS21, maxiter=1000, tol = 1e-14)
    XR, convhist_R = linsolve(x -> one_minus_tm(x, T12; factor = exp(im*momentum)), RHS12, maxiter=1000, tol = 1e-14)

    @tensor Bnew1[-1 -2; -3 -4] := B1[-1 -2; -3 -4] + exp(im*momentum) * mps.AL[1][-1 -2; 1] * XL[1 -3; -4] - XR[-1 -3; 1] * mps.AR[1][1 -2; -4]
    @tensor Bnew2[-1 -2; -3 -4] := B2[-1 -2; -3 -4] + mps.AL[2][-1 -2; 1] * XR[1 -3; -4] - XL[-1 -3; 1] * mps.AR[2][1 -2; -4] 

    @tensor check_B1[-1 -2; -3] := Bnew1[-1 1; -3 2] * conj(mps.AR[1][-2 1; 2])
    @tensor check_B2[-1 -2; -3] := Bnew2[-1 1; -3 2] * conj(mps.AR[2][-2 1; 2])

    @assert norm(check_B1) < 1e-10
    @assert norm(check_B2) < 1e-10

    @tensor X1[-1; -3 -2] := (Bnew1[-1 1; -3 2]) * conj(VRs[1][-2 2; 1])
    @tensor X2[-1; -3 -2] := (Bnew2[-1 1; -3 2]) * conj(VRs[2][-2 2; 1])
    # @tensor X1[-1; -2 -3] := (Bnew1[-1 1; -3 2]) * conj(VRs[1][-2 2; 1])
    # @tensor X2[-1; -2 -3] := (Bnew2[-1 1; -3 2]) * conj(VRs[2][-2 2; 1])
    Xs = [X1, X2]

    @tensor B1_again[-1 -2; -3 -4] := X1[-1; -3 1] * (VRs[1][1 -4; -2])
    @tensor B2_again[-1 -2; -3 -4] := X2[-1; -3 1] * (VRs[2][1 -4; -2])

    @assert norm(B1_again-Bnew1) < 1e-10
    @assert norm(B2_again-Bnew2) < 1e-10

    qp_right = RightGaugedQP(left_gs, right_gs, Xs, VRs, momentum)

    normalize!(qp_right)
    @assert abs(1-norm(qp_right)) < 1e-10

    return qp_right
end


function get_optimal_rotation(quasiparticle; momentum = 0.0, theta_points = 10, phi_points = 10)
    a_min = -5.0
    b_min = -5.0
    a_max = -5.0
    b_max = -5.0
    overlap_min = 1.0
    overlap_max = 0.0

    overlap10 = abs(create_excitation(mps, 1.0, 0.0, quasiparticle; momentum = momentum))
    overlap01 = abs(create_excitation(mps, 0.0, 1.0, quasiparticle; momentum = momentum))
    println(overlap10)
    println(overlap01)
    overlaps = []
    for theta = LinRange(0.0, pi, theta_points)
        println("theta = $(theta/pi) pi")
        # for thetaa = LinRange(0.0, 2*pi, 50)
        overlaps_theta = []
        for phi = LinRange(0.0, 2*pi, phi_points)
            a = cos(theta)#*exp(im*thetaa)
            b = sin(theta)*exp(im*phi)
            if abs(norm([a b])-1.0) > 1e-10
                println("for theta = $(theta), phi = $(phi), norm = $(abs(norm([a b])-1.0))")
            end
            # overlap = abs(a*overlap10 + b*overlap01)
            overlap = abs(create_excitation(mps, a, b, quasiparticle, momentum = momentum))
            if overlap < overlap_min
                overlap_min = overlap
                a_min = a
                b_min = b
            end
            if overlap > overlap_max
                overlap_max = overlap
                a_max = a
                b_max = b
            end
            push!(overlaps_theta, overlap)
        end
        push!(overlaps, overlaps_theta)
    end
    # for theta = LinRange(0.0, pi, theta_points)
    #     println("theta = $(theta/pi) pi")
    #     # for thetaa = LinRange(0.0, 2*pi, 50)
    #     overlaps_theta = []
    #     for phi = LinRange(0.0, 2*pi, phi_points)
    #         a = cos(theta)#*exp(im*thetaa)
    #         b = sin(theta)*exp(im*phi)
    #         overlap = abs(create_excitation(mps, a, b, quasiparticle; momentum = momentum))
    #         if overlap < overlap_best
    #             overlap_best = overlap
    #             a_best = a
    #             b_best = b
    #         end
    #         push!(overlaps_theta, overlap)
    #     end
    #     push!(overlaps, overlaps_theta)
    # end
    # for r = LinRange(0.0, 3.0, theta_points)
    #     println("r = $(r)")
    #     # for thetaa = LinRange(0.0, 2*pi, 50)
    #     for phi = LinRange(0.0, 2*pi, phi_points)
    #         a = 1.0#*exp(im*thetaa)
    #         b = r*exp(im*phi)
    #         norm = sqrt(abs(a)^2 + abs(b)^2)
    #         a = a / norm
    #         b = b / norm
    #         overlap = abs(create_excitation(mps, a, b, quasiparticle; momentum = momentum))
    #         if overlap < overlap_best
    #             overlap_best = overlap
    #             a_best = a
    #             b_best = b
    #         end
    #         push!(overlaps, overlap)
    #     end
    # end
    println("Minimal overlap occurs for a = $(a_min), b = $(b_min)\n Overlap: $(overlap_min)")
    println("Maximal overlap occurs for a = $(a_max), b = $(b_max)\n Overlap: $(overlap_max)")
    return (a_min, b_min, a_max, b_max, overlap_min, overlap_max, overlaps)
end


function optimize_via_energy(mps, mass, Delta_g, v; momentum = 0.0, theta_points = 10, phi_points = 10, new = true)
    a_min = -5.0
    b_min = -5.0
    energy_min = 1e5

    H = get_thirring_hamiltonian_symmetric(mass, Delta_g, v);

    energies = []
    for theta = LinRange(0.0, pi, theta_points)
        println("theta = $(theta/pi) pi")
        energies_theta = []
        for phi = LinRange(0.0, 2*pi, phi_points)
            a = cos(theta)#*exp(im*thetaa)
            b = sin(theta)*exp(im*phi)
            @assert abs(norm([a b])-1.0) < 1e-10

            qp_state = get_excitation(mps, a, b; momentum = momentum, new = new)
            qp_state_prime = effective_excitation_hamiltonian(H, qp_state)
            energy = norm(qp_state_prime)/norm(qp_state)
            println(energy)
            if energy < energy_min
                energy_min = energy
                a_min = a
                b_min = b
            end
            push!(energies_theta, energy)
        end
        push!(energies, energies_theta)
    end


    println("Minimal energy occurs for a = $(a_min), b = $(b_min)\n Energy: $(energy_min)")
    return (a_min, b_min, energy_min, energies)
end


k_values = LinRange(-bounds_k, bounds_k,length(Bs))

# index = 25
# k = k_values[index]
# qp_anti_state = convert(RightGaugedQP, anti_Bs[index])
# qp_anti_state0 = convert(RightGaugedQP, anti_Bs[18])
# qp_state = convert(RightGaugedQP, anti_Bs[index])
# qp_state0 = convert(RightGaugedQP, anti_Bs[18])

(a_min, b_min, energy_min, energies) = optimize_via_energy(mps, mass, Delta_g, v; momentum = 0.1, theta_points = 50, phi_points = 50, new = true)

break

lambda = sqrt(mass^2+sin(k/2)^2)
a1 = exp(im*k/2)
b1 = -sin(k/2)/(mass-lambda)
norm1 = sqrt(abs(a1)^2+abs(b1)^2)
a2 = exp(im*k/2)
b2 = -sin(k/2)/(mass+lambda)
norm2 = sqrt(abs(a2)^2+abs(b2)^2)

angle_extra = 0.5

a = a2/norm2
b = b2/norm2*exp(im*angle_extra)



println(abs(create_excitation(mps, a, b, qp_state, momentum = k)))
println(abs(create_excitation(mps, 1.0, 1.0, qp_state, momentum = k)))



H = get_thirring_hamiltonian_symmetric(mass, Delta_g, v);
qp_finite = convert(FiniteMPS, qp_state)

qp_state_prime = effective_excitation_hamiltonian(H, qp_state)
break
# a2 = 1.0
# b2 = 0.0
# norm2 = 1.0

# println(abs(create_excitation(mps, a1/norm1, b1/norm1, qp_state, momentum = -k)))
# println(abs(create_excitation(mps, a2/norm2, b2/norm2, qp_state, momentum = -k)))
# println(abs(create_excitation(mps, a1/norm1, b1/norm1, qp_state, momentum = k)))
# println(abs(create_excitation(mps, a2/norm2, b2/norm2, qp_state, momentum = k)))
# println(abs(create_excitation(mps, a1/norm1, b1/norm1, qp_state, momentum = -2*k)))
# println(abs(create_excitation(mps, a2/norm2, b2/norm2, qp_state, momentum = -2*k)))
# println(abs(create_excitation(mps, a1/norm1, b1/norm1, qp_state, momentum = 2*k)))
# println(abs(create_excitation(mps, a2/norm2, b2/norm2, qp_state, momentum = 2*k)))

theta_points = 49
phi_points = 49

println("Started with optimalization")
(a_min, b_min, a_max, b_max, overlap_min, overlap_max, overlaps) = get_optimal_rotation(qp_state; momentum = k, theta_points=theta_points,phi_points=phi_points)

lambda = sqrt(mass^2 + (sin(k/2))^2)
if k == 0.0
    expected = [1.0 0.0]
else
    expected = [-1 -exp(-im*k/2)*(mass-lambda)/sin(k/2)]
    expected = expected/norm(expected)
end

A = [mass -(im/2)*(1-exp(im*k)); (im/2)*(1-exp(-im*k)) -mass]
eigen_result = eigen(A)
eigenvectors_matrix = eigen_result.vectors
V₊ = eigenvectors_matrix[:,1]
V₋ = eigenvectors_matrix[:,2]

bogoliubov_rotation = [a_min b_min]

println(norm(dot(V₊, bogoliubov_rotation)))
println(norm(dot(V₋, bogoliubov_rotation)))
println(norm(dot(expected, bogoliubov_rotation)))


thetas = LinRange(0, pi, theta_points)
phis = LinRange(0, 2*pi, phi_points)

plt = plot(phis, overlaps[1], label = "theta = $(round(thetas[1],digits=3))")
for i = 2:length(overlaps)
    plot!(phis, overlaps[i], label = "theta = $(round(thetas[i],digits=3))")
end
xlabel!("phi")
ylabel!("|overlap|")
display(plt)


break
Bs_old = copy(Bs)

println(typeof(Bs[1].left_gs)) # InfiniteMSP
println((Bs[1].VLs[1])) # 3 legs, adjoint (left x p) <- right
println((Bs[1].VLs[2])) # 3 legs, adjoint (right x p) <- left
println((Bs[1].Xs[1])) # 3 legs, right <- plus' x right
println((Bs[1].Xs[2])) # 3 legs, left  <- plus' x 


k_values = LinRange(-bounds_k, bounds_k,length(Bs))

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
trivspace = U1Space(0 => 1)
middle = (-im) * TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im -1.0+0.0im], pspace, pspace)
# middle = (-im) * TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im -1.0+0.0im], trivspace ⊗ pspace, trivspace ⊗ pspace)
H_middle = @mpoham sum(middle{i} for i in vertices(InfiniteChain(2)))
unit_O = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], pspace, pspace)
unit = @mpoham sum(unit_O{i} for i in vertices(InfiniteChain(2)))

# constructing the new groundstate

@assert length(mps) == 2

AL_new = copy(mps.AL)
for w = 1:length(mps)
    @tensor new_tensor[-1 -2; -3] := mps.AL[w][-1 1; -3] * middle[-2; 1]
    AL_new[w] = new_tensor
end

left_gs = right_gs = InfiniteMPS(AL_new, mps.AR, mps.CR, mps.AC)

# one(sectortype(left_gs))
# calculating the nullspaces


VRs = [adjoint(leftnull(adjoint(v))) for v in transpose_tail.(right_gs.AR)]

for w = 1:length(mps)
    @tensor check[-1;-2] := conj(VRs[w][-1 1; 2]) * mps.AR[w][-2 2; 1]
    println(norm(check))
end

# VR = rightnull(mps.AR[1], )

# constructing the B tensors

# contribution of even and odd sites:
a = 1.0
b = 0.0

S_space = Tensor(ones, space(Snew)[1])

@tensor B1[-1 -2; -3 -4] := (a) * mps.AL[1][-1 1; -4] * Snew[2 -2; 1 -3] * conj(S_space[2])
@tensor B2[-1 -2; -3 -4] := (b) * mps.AL[2][-1 1; -4] * Snew[2 -2; 1 -3] * conj(S_space[2])

B_tensors = PeriodicArray([B1 B2])

# @tensor X1[-1; -3 -2] := conj(B1[-1 1; -3 2]) * (VRs[1][-2 2; 1])
# @tensor X2[-1; -3 -2] := conj(B2[-1 1; -3 2]) * (VRs[2][-2 2; 1])
@tensor X1[-1; -3 -2] := (B1[-1 1 -3; 2]) * conj(VRs[1][-2 2; 1])
@tensor X2[-1; -3 -2] := (B2[-1 1; -3 2]) * conj(VRs[2][-2 2; 1])
Xs = [X1, X2]

Bs_first = copy(Bs[1])

# qp_right = RightGaugedQP(VRs, Xs, left_gs, right_gs)
# decompose B tensors and check orthogonality

momentum = 0.0
# qp_right = RightGaugedQP(left_gs, right_gs, Xs, VRs, momentum)

Bs_new = convert(RightGaugedQP, Bs[1])
Bs2_new = convert(RightGaugedQP, anti_Bs[1])
Bs3_new = convert(RightGaugedQP, zero_Bs[1])

dot(qp_right, Bs_new)
