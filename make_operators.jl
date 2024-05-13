using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
Plus_space = U1Space(1 => 1)
Min_space = U1Space(-1 => 1)
Triv_space = U1Space(0 => 1)
S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Triv_space ⊗ pspace, pspace ⊗ Plus_space)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace ⊗ Triv_space)
S⁺minspace = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Triv_space ⊗ pspace, pspace ⊗ Min_space')

S⁺swap = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], Min_space ⊗ pspace, pspace ⊗ Triv_space)
S⁻swap = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Triv_space ⊗ pspace, pspace ⊗ Min_space)

S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)

@save "operators_new" S⁺ S⁻ S⁺minspace S⁺swap S⁻swap S_z_symm