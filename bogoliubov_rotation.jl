using LinearAlgebra
using JLD2
using TensorKit
using KrylovKit
using MPSKitModels, TensorKit, MPSKit
using Base
using Plots

include("get_thirring_hamiltonian.jl")
include("get_thirring_hamiltonian_symmetric.jl")

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

function one_minus_tm(v, tm)
    return v - tm(v)
end

function transpose_tail(t::AbstractTensorMap) # make TensorMap{S,1,N₁+N₂-1}
    I1 = TensorKit.codomainind(t)
    I2 = TensorKit.domainind(t)
    return transpose(t, ((I1[1],), (I2..., reverse(Base.tail(I1))...)))
end

mass = 0.3
v = 0.0
Delta_g = 0.0
truncation = 3.0

@load "SanderDM_Thesis_2324/gs_mps_trunc_$(truncation)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)" mps
@load "SanderDM_Thesis_2324/Dispersion_Delta_m_$(mass)_delta_g_$(Delta_g)_v_$(v)_trunc_$(truncation)_all_sectors" gs_energy energies Bs anti_energies anti_Bs zero_energies zero_Bs bounds_k
@load "operators_for_occupation_number" S⁺ S⁻ S_z_symm

@tensor Snew[-1 -2; -3 -4] := S⁻[-4 -2; -3 -1]


function create_excitation(mps, a, b, qp_state; momentum = 0.0)
    @load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
    @tensor Snew[-1 -2; -3 -4] := S⁻[-4 -2; -3 -1]

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

    @tensor A[-1; -2] := VRs[1][-1 2; 3] * conj(VRs[1][-2 2; 3])

    @tensor should_be_zero[-1; -2] := (VRs[1][-1 1; 2]) * conj(mps.AR[1][-2 2; 1])
    @assert norm(should_be_zero) < 1e-10

    S_space = Tensor(ones, space(Snew)[1])

    @tensor B1[-1 -2; -3 -4] := (a) * mps.AC[1][-1 1; -4] * Snew[2 -2; 1 -3] * conj(S_space[2])
    @tensor B2[-1 -2; -3 -4] := (b) * mps.AC[2][-1 1; -4] * Snew[2 -2; 1 -3] * conj(S_space[2])

    t1 = TransferMatrix(mps.AL[1], mps.AR[1])
    t2 = TransferMatrix(mps.AL[2], mps.AR[2])
    T12 = t1*t2
    T21 = t2*t1

    @tensor RHS12[-1 -3; -2] := B1[-1 1; -3 2] * conj(mps.AR[1][-2 1; 2]) + mps.AL[1][-1 1; 2] * B2[2 3; -3 4] * conj(mps.AR[1][-2 1; 5]) * conj(mps.AR[2][5 3; 4])
    @tensor RHS21[-1 -3; -2] := B2[-1 1; -3 2] * conj(mps.AR[2][-2 1; 2]) + mps.AL[2][-1 1; 2] * B1[2 3; -3 4] * conj(mps.AR[2][-2 1; 5]) * conj(mps.AR[1][5 3; 4])

    XL, convhist_L = linsolve(x -> one_minus_tm(x, T21), RHS21, maxiter=1000, tol = 1e-14)
    XR, convhist_R = linsolve(x -> one_minus_tm(x, T12), RHS12, maxiter=1000, tol = 1e-14)

    @tensor Bnew1[-1 -2; -3 -4] := B1[-1 -2; -3 -4] + mps.AL[1][-1 -2; 1] * XL[1 -3; -4] - XR[-1 -3; 1] * mps.AR[1][1 -2; -4]
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

    # println("norm of X1 = $(norm(X1))")
    # println("norm of X2 = $(norm(X2))")

    @tensor B1_again[-1 -2; -3 -4] := X1[-1; -3 1] * (VRs[1][1 -4; -2])
    @tensor B2_again[-1 -2; -3 -4] := X2[-1; -3 1] * (VRs[2][1 -4; -2])

    @assert norm(B1_again-Bnew1) < 1e-10
    @assert norm(B2_again-Bnew2) < 1e-10

    # qp_right = RightGaugedQP(left_gs, mps, Xs, VRs, momentum)
    # qp_right = RightGaugedQP(left_gs, mps, Xs, VRs, momentum)
    qp_right = RightGaugedQP(left_gs, right_gs, Xs, VRs, momentum)

    # qp_right = RightGaugedQP(VRs, Xs, left_gs, right_gs)

    normalize!(qp_right)
    @assert abs(1-norm(qp_right)) < 1e-10

    return dot(qp_right, qp_state)
    #=
    return 0
    #check 
    @tensor should_be_zero[-1 -2; -3] := B1[1 2; -3 -2] * conj(mps.AL[1][1 2; -1])

    @tensor B1_again[-1 -2; -3 -4] := X1[-1; -3 1] * (VRs[1][1 -4; -2])

    # gauge the Bs
    w = 1
    @tensor T[-1 -2; -3 -4] := mps.AR[w][-1 1; -3] * conj(mps.AL[w][-2 1; -4])
    @tensor T12[-1 -2; -3 -4] := mps.AL[1][-1 1; 2] * mps.AL[2][2 3; -3] * conj(mps.AL[1][-2 1; 4]) * conj(mps.AL[2][4 3; -4])
    @tensor T21[-1 -2; -3 -4] := mps.AL[2][-1 1; 2] * mps.AL[1][2 3; -3] * conj(mps.AL[2][-2 1; 4]) * conj(mps.AL[1][4 3; -4])

    unit_12 = Tensor(ones, T12.codom ⊗ conj(T12.dom[1]) ⊗ conj(T12.dom[2]))
    unit_21 = Tensor(ones, T21.codom ⊗ conj(T21.dom[1]) ⊗ conj(T21.dom[2]))
    @tensor unit_12_new[-1 -2; -3 -4] := unit_12[-1 -2 -3 -4]
    @tensor unit_21_new[-1 -2; -3 -4] := unit_21[-1 -2 -3 -4]

    stm12 = SumTransferMatrix([T12 unit_21])

    # XL, convhist_L = linsolve(to_invert12, RHS12, maxiter=100)
    XL, convhist_L = linsolve(x -> one_minus_tm(x, T21), RHS21, maxiter=1000, tol = 1e-14)
    XR, convhist_R = linsolve(x -> one_minus_tm(x, T12), RHS12, maxiter=1000, tol = 1e-14)

    to_invert12 = unit_12_new - T12
    to_invert21 = unit_21_new - T21


    inverse12 = pinv(to_invert12)
    inverse21 = pinv(to_invert21)

    inverse12 = pinv(T12)
    inverse21 = pinv(T21)



    # @tensor RHS12[-1 -2; -3] := B1[-1 1; -3 2] * conj(mps.AR[1][-2 1; 2]) + mps.AL[1][-1 1; 2] * B2[2 3; -3 4] * conj(mps.AR[1][-2 1; 5]) * conj(mps.AR[2][5 3; 4])
    # @tensor RHS21[-1 -2; -3] := B2[-1 1; -3 2] * conj(mps.AR[2][-2 1; 2]) + mps.AL[2][-1 1; 2] * B1[2 3; -3 4] * conj(mps.AR[2][-2 1; 5]) * conj(mps.AR[1][5 3; 4])


    try_unit = Tensor(ones, B1.dom[2] ⊗ B1.codom[2] ⊗ B1.dom[2])
    T_unit21 = TransferMatrix(try_unit, try_unit)
    XLstart = Tensor(ones, B1.dom[2] ⊗ B1.dom[2] ⊗ B1.dom[1])

    # XL, convhist_L = linsolve([to_invert21], RHS21, maxiter=100)
    # XR, convhist_R = linsolve(T12, RHS12, maxiter=100)    

    # XL = linsolve(to_invert12, RHS12)

    # @tensor test_whether_inverse12[-1 -2; -3 -4] := to_invert12[-1 -2; 1 2] * inverse12[1 2; -3 -4]
    # @tensor test_whether_inverse12[-1 -2; -3 -4] := inverse12[-1 -2; 1 2] * to_invert12[1 2; -3 -4]
    # @tensor test_whether_inverse12[-1 -2; -3 -4] := to_invert12[-1 -2; 1 2] * conj(to_invert12[2 1; -3 -4])
    # println(norm(test_whether_inverse12))
    # # println(fjdskmfkdsqm)

    # @tensor XR[-1 -3; -2] := inverse12[-1 -2; 1 2] * RHS12[1 -3; 2]
    # @tensor XL[-1 -3; -2] := inverse21[-1 -2; 1 2] * RHS21[1 -3; 2]


    # qp_right = RightGaugedQP(left_gs, right_gs, Xs, VRs, momentum)
    qp_right = RightGaugedQP(left_gs, mps, Xs, VRs, momentum)

    return dot(qp_right, qp_state)
    =#
end

Bs_zero = convert(RightGaugedQP, Bs[18])

function get_optimal_rotation(quasiparticle; momentum = 0.0)
    a_best = -5.0
    b_best = -5.0
    overlap_best = 0.0

    a = 1.0
    for r = LinRange(0.0, 3.0, 20)
        println("r = $(r)")
        for theta = LinRange(0.0, 2*pi, 50)
            b = r*exp(im*theta)
            norm = sqrt(abs(a)^2 + abs(b)^2)
            a = a/norm
            b = b/norm
            overlap = abs(create_excitation(mps, a, b, quasiparticle; momentum = momentum))
            if overlap > overlap_best
                overlap_best = overlap
                a_best = a
                b_best = b
            end
        end
    end
    println("Best overlap occurs for a = $(a_best), b = $(b_best)\n Overlap: $(overlap_best)")
    return (a_best,b_best, overlap_best)
end

index = 25
k = k_values[index]
qp_state = convert(RightGaugedQP, Bs[index])

(a, b, overlap) = get_optimal_rotation(qp_state; momentum = -k*2)

lambda = -sqrt(mass^2 + (sin(k/2))^2)
expected = [-1 -exp(-im*k/2)*(mass-lambda)/sin(k/2)]
expected = expected/norm(expected)

A = [mass -(im/2)*(1-exp(im*k)); (im/2)*(1-exp(-im*k)) -mass]
eigen_result = eigen(A)
eigenvectors_matrix = eigen_result.vectors
V₊ = eigenvectors_matrix[:,1]
V₋ = eigenvectors_matrix[:,2]

bogoliubov_rotation = [a b]

println(norm(dot(V₊, bogoliubov_rotation)))
println(norm(dot(V₋, bogoliubov_rotation)))

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
