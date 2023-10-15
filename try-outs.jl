using TensorOperations
using TensorKit
using MPSKit
using MPSKitModels
using LinearAlgebra: norm

print(zeros(5))

D = 4
j = 2
ctr = 1
state = InfiniteMPS([ℂ^2, ℂ^2],[ℂ^D, ℂ^D])
sz_mpo = TensorMap([1.0 0; 0 -1.0], ℂ^1 * ℂ^2, ℂ^2 * ℂ^1)
sz_mpo = TensorMap([1.0 0; 0 -1.0], ℂ^2, ℂ^2)

println([1 2; 3 4])

S⁺ = TensorMap([0.0 1.0; 0.0 0.0], ℂ^2, ℂ^2)
S⁻ = TensorMap([0.0 0.0; 1.0 0.0], ℂ^2, ℂ^2)

println(S⁺)
println(S⁻)

op = S_xx() + S_yy()
print(op)
print(typeof(op))


@tensor op_new[-1 -2; -3 -4] := (1/2)*(S⁺[-1; -3] * S⁻[-2; -4] + S⁺[-2; -4] * S⁻[-1; -3])
println("jkfjkdqlmfqdsjkflmjfqklsmjf")
println(op)
println(op_new)


#=

Sz = S_z()
@tensor op2[-1 -2; -3 -4] := Sz[-1; -4] * Sz[-2; -3]
@tensor op2[-1 -2; -3 -4] := Sz[-1; -4] * Sz[-3; -2]
@tensor op2[-1 -3; -4 -2] := Sz[-1; -4] * Sz[-2; -3]
@tensor op2[-1 -2; -3 -4] := Sz[-1; -3] * Sz[-4; -2]
@tensor op2[-1 -2; -3 -4] := Sz[-1; -3] * Sz[-2; -4]



@tensor op3[-1 -2; -3 -4] := Sz[-2; -1] * Sz[-3; -4]

println("start comparison")
println("fjdsqklmfjksjfqklmdjkl\n")
println(op2)
println(S_zz())


=#

#=
println("here")
println(length(state))
println("fjdkqlm")
@tensor W = state.AC[1][1 2; 3] * conj(state.AC[1][1 2; 3])
println("fhe")
println(W)
print("done with W")
@tensor V[-1; -2] := sz_mpo[-1; 1] * sz_mpo[1; -2]

print(V)
print(K)


println("done");
#V = TransferMatrix(state.AR[ctr:(j - 1)], [sz_mpo sz_mpo], state.AR[ctr:(j - 1)])



println(exp(1))
println(log(exp(1)))

print(floor(Int, 3.0))

println(abs(1+1im))

println(1 + im*5)

b = zeros(ComplexF64, (1, 20))
for i = 1:5
    global b += i * ones(ComplexF64, (1, 20))
end
b ./= 5
println(b)

println(0%2)

=#

#=
# You can create a random 1 site periodic infinite mps (bond dimension 10) by calling
state = InfiniteMPS([ℂ^2],[ℂ^10]);

state.AL

#We can use a pre-defined hamiltonian from MPSKitModels
hamiltonian = transverse_field_ising();
hamiltonian = heisenberg_XXX();


#And find the groundstate
(groundstate,_) = find_groundstate(state,hamiltonian,VUMPS(maxiter = 3));

println(expectation_value(state, hamiltonian))
println(expectation_value(groundstate, hamiltonian))

spin = 3 // 2
S_x_mat, _, _ = spinmatrices(spin, ComplexF64)
pspace = ComplexSpace(size(S_x_mat, 1))
println(size(S_x_mat))
=#

println("D = ", 60)