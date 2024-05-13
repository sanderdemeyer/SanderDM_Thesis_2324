function get_X_tensors(tensors)
    # tensors is for example mps.AL, mps.AC or mps.AR
    # this assumes the period of the unit cell to be 2.

    spin = 1//2
    pspace = U1Space(i => 1 for i in (-spin):spin)
    S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)

    w = length(mps)
    @assert w == 2

    spaces = [codomain(tensors[site])[1] for site = 1:w]

    # data = [Dict(i => (-1)^i.charge * (1+im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = V1.dims) for site = 1:w]

    data = []
    for site = 1:w
        if (site % 2 == 1)
            push!(data, Dict(i => (-1)^i.charge * (1+im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = spaces[site].dims))
        else
            push!(data, Dict(i => (-1)^((numerator(i.charge)+1)/2) * (1-im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = spaces[site].dims))
        end
    end
    X_tensors = [TensorMap(data[site], spaces[site], spaces[site]) for site = 1:w]

    @tensor check11[-1 -2; -3] := inv(X_tensors[1])[-1; 1] * tensors[1][1 -2; 2] * X_tensors[2][2; -3]
    @tensor check12[-1 -2; -3] := tensors[1][-1 1; -3] * (2*im)*S_z_symm[-2; 1]
    @tensor check21[-1 -2; -3] := inv(X_tensors[2])[-1; 1] * tensors[2][1 -2; 2] * X_tensors[1][2; -3]
    @tensor check22[-1 -2; -3] := tensors[2][-1 1; -3] * (2*im)*S_z_symm[-2; 1]

    @assert norm(check11-check12) < 1e-10
    @assert norm(check21-check22) < 1e-10

    return X_tensors
    # data_X1 = Dict(i => (-1)^i.charge * (1+im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = V1.dims)
    # data_X2 = Dict(i => (-1)^((numerator(i.charge)+1)/2) * (1-im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = V2.dims)    
end
