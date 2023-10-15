using TensorOperations
using TensorKit
using MPSKit
using MPSKitModels
using LinearAlgebra: norm

function correlation_function(mps, O₁, O₂, max_dist::Int = 200)
    corr_list = zeros(max_dist)

    w = length(mps)

    print(typeof(mps))
    print(typeof(mps.AC[1]))
    
    x = Vector{TensorMap}(undef, w)
    for b = 1:w
        println("here1")
        @tensor tens[-1; -2] := mps.AC[b][1 2; -2] * O₁[3; 2] * conj(mps.AC[b][1 3; -1])
        println("here2")
        x[b] = tens
    end

    for i = 1:max_dist
        println(i)
        x_final = Vector{Complex{Float64}}(undef, w)
        for b = 1:w
            x_final[b] = @tensor x[b][4; 1] * mps.AC[mod1(b+i,w)][1 2; 5] * O₂[3; 2] * conj(mps.AC[mod1(b+i,w)][4 3; 5])
        end
    end
end

D = 3
state = InfiniteMPS([ℂ^2, ℂ^2],[ℂ^D, ℂ^D])
sz_mpo = TensorMap([1.0 0; 0 -1.0], ℂ^1 * ℂ^2, ℂ^2 * ℂ^1)
sz_mpo = TensorMap([1.0 0; 0 -1.0], ℂ^2, ℂ^2)

print(typeof(state))
print(typeof(state.AC[1]))

println("started with corr function")

correlation_function(state, sz_mpo, sz_mpo, 3)

println("done")