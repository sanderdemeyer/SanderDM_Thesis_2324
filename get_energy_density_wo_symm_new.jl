using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using LaTeXStrings
using Random
using QuadGK

include("get_X_tensors.jl")
include("get_X_tensors_wo_symmetries.jl")
include("get_thirring_hamiltonian.jl")
include("get_occupation_number_matrices.jl")
include("get_thirring_hamiltonian_only_m.jl")
include("get_groundstate_energy.jl")

# code from Daan (unadapted)

function samesiteEcorrelator(state::MPSKit.AbstractMPS, H, O, Xs, js::AbstractRange{Int}, envs = environments(state,H))
    ens = map(js) do j
        @plansor AC[-1 -2; -3] := (Xs[mod1(j,length(Xs))])[-1;1]*state.AC[j][1 2;-3]*O[-2;2]
        left = leftenv(envs,j,state) * TransferMatrix(AC,H[j],AC)
        right = rightenv(envs,j,state)
        curr = 0.
        for k in 1:length(left) #close the expression
            curr += @tensor left[k][1; 3 2] * right[k][2 3;1]
        end
        curr
    end
    return ens
end

function samesiteEcorrelator_old(state::MPSKit.AbstractMPS, H, O, Xs, n::Int, js::AbstractRange{Int}, envs = environments(st,H))

    ens = similar(js, eltype(eltype(state)))

    if n < first(js) #start with n and transfer through to first(js)
        # G = starters[first(js)] # adapted
        G = leftenv(envs, first(js), state)
    elseif n == first(js)
        G = leftenv(envs,n,state)
    else
        G = leftenv(envs,first(js),state)
    end

    prevj = first(js)
    for (jn,j) in enumerate(js)

        #decide what to add
        if prevj < j
            if n < prevj #n has already been done
                G = G * TransferMatrix(state.AR[prevj:j-1],H[prevj:j-1],state.AR[prevj:j-1])
            elseif j < n #n doesn't need to be done, but also not added to G
                G = G * TransferMatrix(state.AL[prevj:j-1],H[prevj:j-1],state.AL[prevj:j-1])
            else #n somewhere in between and needs to be added
                G = G * TransferMatrix(state.AL[prevj:n-1],H[prevj:n-1],state.AL[prevj:n-1])
                if n != j #partial transfer and O2 in the same spot
                    G = PartTransferLeft(state,H,n,G)
                    G = G * TransferMatrix(state.AR[n+1:j-1],H[n+1:j-1],state.AR[n+1:j-1])
                end
            end
        end

        #decide what GR should be
        if n < prevj
            @plansor AR1[-1 -2; -3] := Xs[mod1(j,length(Xs))][-1;1]*state.AR[j][1 2;-3]*O[-2;2]
            @plansor AR2[-1 -2; -3] := Xs[mod1(j,length(Xs))][-1;1]*state.AR[j][1 2;-3]*O[-2;2]
            GR =  TransferMatrix(AR1,H[j],AR2) * rightenv(envs,j,state)
        elseif j < n
            @plansor AL1[-1 -2; -3] := Xs[mod1(j,length(Xs))][-1;1]*state.AL[j][1 2;-3]*O[-2;2]
            @plansor AL2[-1 -2; -3] := Xs[mod1(j,length(Xs))][-1;1]*state.AL[j][1 2;-3]*O[-2;2]
            # GR = TransferMatrix(AL1,H[j],AL2) * closures[j] # adapted
            GR = TransferMatrix(AL1,H[j],AL2) * rightenv(envs, j, state)
        else
            if n==j
                @plansor AC1[-1 -2; -3] := Xs[mod1(j,length(Xs))][-1;1]*state.AC[j][1 2;-3]*O[-2;2]
                @plansor AC2[-1 -2; -3] := Xs[mod1(j,length(Xs))][-1;1]*state.AC[j][1 2;-3]*O[-2;2]
                GR = PartTransferRight(AC1,AC2,H[j],rightenv(envs,j,state),norm(state.AC[end])^2)
            else
                @plansor AR1[-1 -2; -3] := Xs[mod1(j,length(Xs))][-1;1]*state.AR[j][1 2;-3]*O[-2;2]
                @plansor AR2[-1 -2; -3] := Xs[mod1(j,length(Xs))][-1;1]*state.AR[j][1 2;-3]*O[-2;2]
                GR = TransferMatrix(AR1,H[j],AR2) * rightenv(envs,j,state)
            end
        end

        curr = 0.
        for k in 1:length(GR) #close the expression
            curr += @tensor G[k][1; 3 2] * GR[k][2 3;1]
        end
        ens[jn] = curr
        prevj = j

    end
    return ens
end

function Ecorrelator(state::MPSKit.AbstractMPS, H, O, Xs, n::Int, i::Int, js::AbstractRange{Int}, envs = environments(st,H))
    first(js) > i || @error "i should be smaller than j ($i, $(first(js)))"

    ens = similar(js, eltype(eltype(state)))

    # for j in js < n we can calculate closure GR by growing
    smallerjs = filter(<(n),js)
    closureGS = Vector{typeof(rightenv(envs,n,state))}(undef,length(smallerjs))
    if !isempty(smallerjs) #only need this if any js < n
        closureGS[end] = TransferMatrix(state.AL[last(smallerjs)+1:n-1],H[last(smallerjs)+1:n-1],state.AL[last(smallerjs)+1:n-1]) *
            PartTransferRight(state,H,n,rightenv(envs,n,state))
        for jn in length(closureGS)-1:-1:1
            trange = smallerjs[jn]+1:smallerjs[jn+1]
            closureGS[jn] = TransferMatrix(state.AL[trange],H[trange],state.AL[trange]) * closureGS[jn+1]
        end
    end

    if n < i
        G = PartTransferLeft(state,H,n,leftenv(envs,n,state))
        G = G * TransferMatrix(state.AR[n+1:i-1],H[n+1:i-1],state.AR[n+1:i-1])
        @plansor ARx[-1 -2; -3] := Xs[mod1(i,length(Xs))][-1;1]*state.AR[i][1 2;-3]*O[-2;2]
        G = G * TransferMatrix(ARx,H[i],state.AR[i])
    elseif n == i
        @plansor ACx[-1 -2; -3] := Xs[mod1(n,length(Xs))][-1;1]*state.AC[n][1 2;-3]*O[-2;2]
        G = PartTransferLeft(ACx,state.AC[n],H[n],leftenv(envs,n,state), norm(state.AC[end])^2)
    else
        @plansor ALx[-1 -2; -3] := Xs[mod1(i,length(Xs))][-1;1]*state.AL[i][1 2;-3]*O[-2;2]
        G = leftenv(envs,i,state) * TransferMatrix(ALx,H[i],state.AL[i])
    end

    prevj = i+1
    for (jn,j) in enumerate(js)

        #decide what to add
        if prevj < j
            if n < prevj #n has already been done
                G = G * TransferMatrix(state.AR[prevj:j-1],H[prevj:j-1],state.AR[prevj:j-1])
            elseif j < n #n doesn't need to be done, but also not added to G
                G = G * TransferMatrix(state.AL[prevj:j-1],H[prevj:j-1],state.AL[prevj:j-1])
            else #n somewhere in between and needs to be added
                G = G * TransferMatrix(state.AL[prevj:n-1],H[prevj:n-1],state.AL[prevj:n-1])
                if n != j # n needs to be added to G
                    G = PartTransferLeft(state,H,n,G)
                    G = G * TransferMatrix(state.AR[n+1:j-1],H[n+1:j-1],state.AR[n+1:j-1])
                end
            end
        end

        #decide what GR should be
        if n < prevj
            @plansor ARx[-1 -2; -3] := Xs[mod1(j,length(Xs))][-1;1]*state.AR[j][1 2;-3]*O[-2;2]
            GR = TransferMatrix(state.AR[j],H[j],ARx) * rightenv(envs,j,state)
        elseif j < n
            @plansor ALx[-1 -2; -3] := Xs[mod1(j,length(Xs))][-1;1]*state.AL[j][1 2;-3]*O[-2;2]
            GR = TransferMatrix(state.AL[j],H[j],ALx) * closureGS[jn]
        else
            if n==j
                @plansor ACx[-1 -2; -3] := Xs[mod1(j,length(Xs))][-1;1]*state.AC[j][1 2;-3]*O[-2;2]
                GR = PartTransferRight(state.AC[j],ACx,H[j],rightenv(envs,j,state), norm(state.AC[end])^2)
            else
                @plansor ARx[-1 -2; -3] := Xs[mod1(j,length(Xs))][-1;1]*state.AR[j][1 2;-3]*O[-2;2]
                GR = TransferMatrix(state.AR[j],H[j],ARx) * rightenv(envs,j,state)
            end
        end

        curr = 0.
        for k in 1:length(GR) #close the expression
            curr += @tensor G[k][1; 3 2] * GR[k][2 3;1]
        end
        ens[jn] = curr
        prevj = j

    end
    return ens
end

function EcorrelationMatrix(st,L,n,H,O,Xs,envs=environments(st,H))
    ε = zeros(ComplexF64,L,L);
    numberops = samesiteEcorrelator(st,H,O,Xs,1:L,envs)
    println(numberops)
    for i in 1:L
        ε[i,i] = numberops[i]
        ε[i+1:L,i] .= Ecorrelator(st,H,O,Xs,n,i,i+1:L,envs)
    end
    Hermitian(ε,:L)
end


# convertion. Own code

function convert_to_array(tensor::TensorMap)
    data = convert(Array, tensor)
    dims = size(data)
    len = length(dims)
    if len == 3
        return TensorMap(data, ℂ^dims[1] ⊗ ℂ^dims[2], ℂ^dims[3])
    else
        return TensorMap(data, ℂ^dims[1], ℂ^dims[2])
    end
end

function convert_to_array(tensors::Union{PeriodicArray{A}, Vector{A}}) where {A <: TensorMap}
    return [convert_to_array(tensor) for tensor in tensors]
end

function remove_symmetries(mps::InfiniteMPS)
    ALS = PeriodicArray(convert_to_array(mps.AL))
    ARS = PeriodicArray(convert_to_array(mps.AR))
    ACS = PeriodicArray(convert_to_array(mps.AC))
    CRS = PeriodicArray(convert_to_array(mps.CR))
    return InfiniteMPS(ALS, ARS, CRS, ACS)
end

truncation = 2.5
mass = 1.0
Delta_g = 0.0
v = 0.0

N = 20

if mass == 0.0
    @load "N_20_mass_0_occupation_matrices_new" corr corr_energy
elseif mass == 30.0
    @load "N_20_mass_30_occupation_matrices_new" corr corr_energy
elseif mass == 1.0
    @load "N_20_mass_1_occupation_matrices_new" corr corr_energy
else
    @assert 0 == 1 
end

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)    
Plus_space = U1Space(1 => 1)
S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace ⊗ Plus_space)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace)
S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)

Sp_asym = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
Sm_asym = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)


@load "SanderDM_Thesis_2324/gs_mps_trunc_$(truncation)_mass_$(mass)_v_0.0_Delta_g_$(Delta_g)" mps
Xs = get_X_tensors(mps.AL) # geeft hetzelfde indiend AC of AR

H = get_thirring_hamiltonian(mass, Delta_g, v)
unit_tensor = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], ℂ^2, ℂ^2)
H_unit = @mpoham sum((unit_tensor){i} for i in vertices(InfiniteChain(2)))


@tensor check11[-1 -2; -3] := inv(Xs[1])[-1; 1] * mps.AC[1][1 -2; 2] * Xs[2][2; -3]
@tensor check12[-1 -2; -3] := mps.AC[1][-1 1; -3] * (2*im)*S_z_symm[-2; 1]
@tensor check21[-1 -2; -3] := inv(Xs[2])[-1; 1] * mps.AC[2][1 -2; 2] * Xs[1][2; -3]
@tensor check22[-1 -2; -3] := mps.AC[2][-1 1; -3] * (2*im)*S_z_symm[-2; 1]
@assert norm(check11-check12) < 1e-10
@assert norm(check21-check22) < 1e-10

mps = remove_symmetries(mps)
Xs = convert_to_array(Xs)
S_z_symm = convert_to_array(S_z_symm)


@tensor check11[-1 -2; -3] := inv(Xs[1])[-1; 1] * mps.AC[1][1 -2; 2] * Xs[2][2; -3]
@tensor check12[-1 -2; -3] := mps.AC[1][-1 1; -3] * (2*im)*S_z_symm[-2; 1]
@tensor check21[-1 -2; -3] := inv(Xs[2])[-1; 1] * mps.AC[2][1 -2; 2] * Xs[1][2; -3]
@tensor check22[-1 -2; -3] := mps.AC[2][-1 1; -3] * (2*im)*S_z_symm[-2; 1]
@assert norm(check11-check12) < 1e-10
@assert norm(check21-check22) < 1e-10

println("Started with calculating the correlation matrix")

# for i = 1:2*N
#     println(corr[i,i])
# end

same_site = samesiteEcorrelator(mps, H, Sp_asym, Xs, 1:2*N)
same_site_min = samesiteEcorrelator(mps, H, Sm_asym, Xs, 1:2*N)

same_site_one = samesiteEcorrelator(mps, H_unit, Sp_asym, Xs, 1:2*N)
same_site_one_min = samesiteEcorrelator(mps, H_unit, Sm_asym, Xs, 1:2*N)

#=



corr_energy_mps = EcorrelationMatrix(mps,2*N,0,H_unit,Sp_asym,Xs)



# AC1 = TensorMap(convert(Array, mps.AC[1]), ℂ^19 ⊗ ℂ^2, ℂ^17)
# AC2 = TensorMap(convert(Array, mps.AC[2]), ℂ^17 ⊗ ℂ^2, ℂ^19)
# X1 = TensorMap(convert(Array, Xs[1]), ℂ^19, ℂ^19)
# X2 = TensorMap(convert(Array, Xs[2]), ℂ^17, ℂ^17)
# S_z = TensorMap(convert(Array, S_z_symm), ℂ^2, ℂ^2)
# S⁺_asym = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
# S⁻_asym = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)

# @tensor check31[-1 -2; -3] := inv(X1)[-1; 1] * AC1[1 -2; 2] * X2[2; -3]
# @tensor check32[-1 -2; -3] := AC1[-1 1; -3] * (2*im)*S_z[-2; 1]
# @tensor check41[-1 -2; -3] := inv(X2)[-1; 1] * AC2[1 -2; 2] * X1[2; -3]
# @tensor check42[-1 -2; -3] := AC2[-1 1; -3] * (2*im)*S_z[-2; 1]
# @assert norm(check31-check32) < 1e-10
# @assert norm(check41-check42) < 1e-10


X = [(2*pi)/N*i - pi for i = 0:N-1]
σ = 0.1
x₀ = div(N,2)

=#

X = [(2*pi)/N*i - pi for i = 0:N-1]
σ = 0.1
x₀ = div(N,2)

(V₊,V₋) = V_matrix_pos_neg_energy(X, mass)
# (V₊,V₋) = V_matrix(X, mass)
occ_matrix_energy = adjoint(V₊)*(corr_energy)*(V₊)
occ_matrix = adjoint(V₊)*corr*(V₊)
occ_matrix_energy_min = adjoint(V₋)*(corr_energy)*(V₋)
occ_matrix_min = adjoint(V₋)*corr*(V₋)

@assert adjoint(V₊)*(V₊) ≈ I
@assert adjoint(V₋)*(V₋) ≈ I
@assert (V₊)*adjoint(V₊) + (V₋)*adjoint(V₋) ≈ I

# E0 = get_groundstate_energy(mass, Delta_g, v, [10 30], 2.5, 10^(-2.5))[1]/2 # /2 want per site
corr_energy_real = corr_energy + E0*(corr)

occ = zeros(Float64, N)
occ_energy = zeros(Float64, N)
for (i,k₀) in enumerate(X)
    array = gaussian_array(X, k₀, σ, x₀)
    occupation_number = ((array)*occ_matrix*adjoint(array))
    occupation_number_energy = 0.0
    occupation_number_energy = (array)*((occ_matrix_energy)+E0*(-I + 2*occ_matrix))*adjoint(array) / ((array)*(I - occ_matrix)*adjoint(array))
    if (abs(imag(occupation_number)) > 1e-2)
        println("Warning, complex number for occupation number: $(occupation_number)")
    end
    occ[i] = real(occupation_number)
    occ_energy[i] = real(occupation_number_energy)
end

ϵs = zeros(ComplexF64, N, N)
for (i,k) in enumerate(X)
    ϵs[i,i] = sqrt(mass^2 + sin(k/2)^2)
end
ϵs_double = zeros(ComplexF64, 2*N, 2*N)
for (i,k) in enumerate(X)
    ϵs_double[2*i-1,2*i-1] = sqrt(mass^2 + sin(k/2)^2)
    ϵs_double[2*i,2*i] = -sqrt(mass^2 + sin(k/2)^2)
end


@assert V₊ * adjoint(V₊) + V₋ * adjoint(V₋) ≈ I
@assert adjoint(V₊) * V₊ ≈ I

factor = 1

expected_energy = (V₊) * (ϵs * adjoint(V₊))  + factor*E0 * V₊ * adjoint(V₊) # <c_i H c_j^dagger>
expected = V₊ * adjoint(V₊)


# expected_energy_double = adjoint(V₊) * (ϵs_double * (V₊))  + E0 * adjoint(V₊) * (V₊)
# expected_double = adjoint(V₊) * (V₊)

occ_energy = zeros(Float64, N)
for (i,k₀) in enumerate(X)
    array = V₊ * adjoint(gaussian_array(X, k₀, σ, x₀))
    # arraym = V₋ * adjoint(gaussian_array(X, k₀, σ, x₀))
    println(size(array))
    occupation_number = (adjoint(array)*corr*(array))
    occupation_number_energy = adjoint(array)*(expected_energy)*(array)# / (adjoint(array)*(corr)*(array))
    if (abs(imag(occupation_number)) > 1e-2)
        println("Warning, complex number for occupation number: $(occupation_number)")
    end
    occ[i] = real(occupation_number)
    occ_energy[i] = real(occupation_number_energy)
end

plt = plot(X, occ_energy .- (factor)*E0); 
theor = [sqrt(mass^2+sin(k/2)^2) for k in X]
plot!(X, theor, label = "theory")
display(plt);