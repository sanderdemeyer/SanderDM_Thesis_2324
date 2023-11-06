using TensorOperations
using TensorKit
using MPSKit
using MPSKitModels
using LinearAlgebra: norm
using Statistics


function correlation_function(mps, O₁, O₂, max_dist::Int = 200)
    corr_list = zeros(max_dist)
    w = length(mps)
    x = Vector{TensorMap}(undef, w)
    
    for b = 1:w
        @tensor x[b][-1; -2] := mps.AC[b][1 2; -2] * O₁[3; 2] * conj(mps.AC[b][1 3; -1])
    end

    for i = 1:max_dist
        x_final = Vector{Complex{Float64}}(undef, w)
        for b = 1:w
            x_final[b] = real(@tensor x[b][4; 1] * mps.AR[mod1(b+i,w)][1 2; 5] * O₂[3; 2] * conj(mps.AR[mod1(b+i,w)][4 3; 5]))
        end
        corr_list[i] = Statistics.mean(x_final)
        if i != max_dist
            for b = 1:w
                @tensor x[b][-1; -2] := x[b][1; 2] * mps.AR[mod1(b+i,w)][2 3; -2] * conj(mps.AR[mod1(b+i,w)][1 3; -1])
            end
        end
    end
    return corr_list
end

function correlation_function(mps, O₁, middle, O₂, max_dist::Int = 200)
    corr_list = zeros(max_dist)
    w = length(mps)
    x = Vector{TensorMap}(undef, w)
    for b = 1:w
        @tensor tens[-1; -2] := mps.AC[b][1 2; -2] * O₁[3; 2] * conj(mps.AC[b][1 3; -1])
        x[b] = tens
    end

    for i = 1:max_dist
        x_final = Vector{Complex{Float64}}(undef, w)
        for b = 1:w
            x_final[b] = real(@tensor x[b][4; 1] * mps.AR[mod1(b+i,w)][1 2; 5] * O₂[3; 2] * conj(mps.AR[mod1(b+i,w)][4 3; 5]))
        end
        corr_list[i] = Statistics.mean(x_final)
        if i != max_dist
            for b = 1:w
                @tensor tens[-1; -2] := x[b][1; 2] * mps.AR[mod1(b+i,w)][2 3; -2] * middle[4; 3] * conj(mps.AR[mod1(b+i,w)][1 4; -1])
                x[b] = tens
            end
        end
    end
    return corr_list
end

correlation_length()