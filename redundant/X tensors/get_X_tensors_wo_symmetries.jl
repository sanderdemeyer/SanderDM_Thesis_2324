function get_X_tensors_wo_symmetries(tensors)
    # tensors is for example mps.AL, mps.AC or mps.AR
    # this assumes the period of the unit cell to be 2.

    S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], ℂ^2, ℂ^2)

    w = length(mps)
    @assert w == 2

    spaces = [codomain(tensors[site])[1] for site = 1:w]
    ds = [space.d for space in spaces]
    # data = [Dict(i => (-1)^i.charge * (1+im)/sqrt(2) * Matrix{Float64}(I,j,j) for (i,j) = V1.dims) for site = 1:w]
    amounts = [div(d,2) for d in ds]
    for m = 1:ds[1]
        for n = 1:ds[2]
            data1 = zeros(ComplexF64,ds[1],ds[1]);
            if m+amounts[1] <= ds[1]+1
                for i = 1:ds[1]
                    data1[i,i] = (-1)^((i >= m) && (i < m+amounts[1])) * (1+im)/sqrt(2)
                end
            else
                for i = 1:ds[1]
                    data1[i,i] = (-1)^((i < m+amounts[1]-ds[1]) || (i >= m)) * (1+im)/sqrt(2)
                end
            end
            data2 = zeros(ComplexF64,ds[2],ds[2]);
            if n+amounts[2] <= ds[2]+1
                for i = 1:ds[2]
                    data2[i,i] = (-1)^((i >= n) && (i < n+amounts[2])) * (1+im)/sqrt(2)
                end
            else
                for i = 1:ds[2]
                    data2[i,i] = (-1)^((i < n+amounts[2]-ds[2]) || (i >= n)) * (1+im)/sqrt(2)
                end
            end
            X1 = TensorMap(data1, spaces[1], spaces[1])
            X2 = TensorMap(data2, spaces[2], spaces[2])

            @tensor check11[-1 -2; -3] := inv(X1)[-1; 1] * tensors[1][1 -2; 2] * X2[2; -3]
            @tensor check12[-1 -2; -3] := tensors[1][-1 1; -3] * (2*im)*S_z_symm[-2; 1]
            @tensor check21[-1 -2; -3] := inv(X2)[-1; 1] * tensors[2][1 -2; 2] * X1[2; -3]
            @tensor check22[-1 -2; -3] := tensors[2][-1 1; -3] * (2*im)*S_z_symm[-2; 1]
        
            println("for m = $m and n = $n")
            println(norm(check11))
            println(norm(check21))
            println(norm(check12))
            println(norm(check22))
            println(norm(check11+im*check12))
            println(norm(check21+im*check22))
        end
    end
    X_tensors = [TensorMap(data, spaces[site], spaces[site]) for site = 1:w]

    # data = []
    # for site = 1:w
    #     if (site % 2 == 1)
    #         push!(data, Dict(i => (-1.0+0.0im)^i * (1+im)/sqrt(2) for i = 1:spaces[site].d))
    #     else
    #         push!(data, Dict(i => (-1.0+0.0im)^(i+1//2) * (1-im)/sqrt(2) for i = 1:spaces[site].d))
    #     end
    # end
    # X_tensors = [TensorMap(i -> [Dict(j => [i] for j = 1:spaces[site].d) [i]], spaces[site], spaces[site]) for site = 1:w]

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
