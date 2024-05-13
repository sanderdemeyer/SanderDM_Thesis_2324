function get_entropy(mps::Union{InfiniteMPS,FiniteMPS}, js::AbstractRange{Int})
    @tensor tfm[-1 -2; -3 -4] := mps.AC[js[1]][-1 1; -2] * conj(mps.AC[js[1]][-3 1; -4])
    if length(js) == 1
        return tm
    end
    for j in js[2:end]
        @tensor tfm[-1 -2; -3 -4] := tfm[-1 1; -3 3] * mps.AR[j][1 2; -2] * conj(mps.AR[j][3 2; -4])
    end
    data = convert(Array, tfm)
    data_size = size(data)
    size_of_data = data_size[1]*data_size[2]
    data = reshape(data, size_of_data, size_of_data)
    (eigval, _) = eigen(data)
    eigval = [x for x in real.(eigval) if x > 0]
    return -sum(eigval .* log.(eigval))
end

function get_entropy(mps::Union{Window, WindowMPS}, js::AbstractRange{Int})
    return get_entropy(mps.middle, js)
end