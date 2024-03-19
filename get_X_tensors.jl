function get_X_tensors(tensors)
    # tensors is for example mps.AL, mps.AC or mps.AR
    # this assumes the period of the unit cell to be 2.

    w = length(mps)
    spaces = [codomain(tensors[site])[1] for site = 1:w]

    data = [Dict(i => (-1)^i.charge * (1+im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = V1.dims) for site = 1:w]

    data = []
    for site = 1:w
        if (site % 2 == 1)
            push!(data, Dict(i => (-1)^i.charge * (1+im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = spaces[site].dims))
        else
            push!(data, Dict(i => (-1)^((numerator(i.charge)+1)/2) * (1-im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = spaces[site].dims))
        end
    end
    X_tensors = [TensorMap(data[site], spaces[site], spaces[site]) for site = 1:w]
    println("spaces are $(spaces[1])")
    println("spaces are $(spaces[2])")
    println("done")
    return X_tensors
    # data_X1 = Dict(i => (-1)^i.charge * (1+im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = V1.dims)
    # data_X2 = Dict(i => (-1)^((numerator(i.charge)+1)/2) * (1-im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = V2.dims)    
end
