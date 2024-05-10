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

function energy(k, m)
    if (k < 0.0)
        return -sqrt(m^2+sin(k/2)^2)
    else
        return sqrt(m^2+sin(k/2)^2)
    end
end

function samesiteEcorrelator_new(st, H, O, Xs, n, range, envs)
    G = []
    for i = (range)#.+1
        println(i)
        left_env = leftenv(envs, i, st)
        right_env = rightenv(envs, i, st)
        value_onsite = 0.0
        for (j,k) in keys(H[i]) #[(1,H[i].odim)] #
            # j = 1
            # k = H[i].odim
            term = @tensor left_env[j][16 5; 15] * Xs[mod1(i,length(Xs))][15;1] * st.AC[i][1 2; 13] * O[4; 2 7] * H[i][j,k][5 6; 4 12] * conj(O[6; 9 7]) * conj(Xs[mod1(i,length(Xs))][16;10]) * conj(st.AC[i][10 9; 11]) * right_env[k][13 12; 11]
            println("for (j,k) = ($(j),$(k)), term = $(term)")
            value_onsite += term
        end
        push!(G, value_onsite)
    end
    return G
end

function get_left_mps(mps, middle)
    AL_new = copy(mps.AL)
    AC_new = copy(mps.AC)
    AR_new = copy(mps.AR)
    for w = 1:length(mps)
        @tensor new_tensor_AL[-1 -2; -3] := mps.AL[w][-1 1; -3] * middle[-2; 1]
        AL_new[w] = copy(new_tensor_AL)
        @tensor new_tensor_AR[-1 -2; -3] := mps.AR[w][-1 1; -3] * middle[-2; 1]
        AR_new[w] = copy(new_tensor_AR)
        @tensor new_tensor_AC[-1 -2; -3] := mps.AC[w][-1 1; -3] * middle[-2; 1]
        AC_new[w] = copy(new_tensor_AC)
    end

    return InfiniteMPS(AL_new, AR_new, mps.CR, AC_new)
end

function samesiteEcorrelator_newest(mps, H, O, middle, range)
    G = []
    mps_l = get_left_mps(mps, middle)
    envs = environments(mps,H)
    envs_l = environments(mps_l,H)
    for i = range
        println("i = $i")
        left_env = leftenv(envs_l, i, mps_l)
        right_env = rightenv(envs, i, mps)
        value_onsite = 0.0
        for (j,k) in keys(H[i]) #[(1,H[i].odim)] #
            # j = 1
            # k = H[i].odim
            term = @tensor left_env[j][10 5; 1] * mps.AC[i][1 2; 13] * O[4; 2] * H[i][j,k][5 6; 4 12] * conj(O[6; 9]) * conj(mps.AC[i][10 9; 11]) * right_env[k][13 12; 11]
            println("for (j,k) = ($(j),$(k)), term = $(term)")
            value_onsite += term
        end
        push!(G, value_onsite)
    end
    return G
end

function further_Ecorrelator_newest(mps, H, O, middle, i, L)
    G = zeros(ComplexF64, L)
    mps_l = get_left_mps(mps, middle)
    envs = environments(mps,H)
    envs_l = environments(mps_l,H)

    @tensor ACS[-1 -2; -3] := mps.AC[i][-1 1; -3] * O[-2; 1]
    @tensor ACO[-1 -2; -3] := mps.AC[i][-1 1; -3] * middle[-2; 1]
    left = leftenv(envs_l, i, mps_l) * TransferMatrix(ACS, H[i], ACO)

    for j = i+1:L
        @tensor ARS[-1 -2; -3] := mps.AR[j][-1 1; -3] * O[-2; 1]
        right = TransferMatrix(mps.AR[j], H[j], ARS) * rightenv(envs, j, mps)
        curr = 0.0
        for k in 1:length(right) #close the expression
            term = @tensor left[k][1; 2 3] * right[k][3; 2 1]
            if j < 7
                println("j = $(j), k = $(k), term = $(term)")
            end
            curr += term
        end
        G[j] = curr

        @tensor ARO[-1 -2; -3] := mps.AR[j][-1 1; -3] * middle[-2; 1]
        left = left * TransferMatrix(mps.AR[j], H[j], ARO)
    end
    return G    
end



function samesiteEcorrelator(state::MPSKit.AbstractMPS, H, O, Xs, n::Int, js::AbstractRange{Int}, envs = environments(st,H), starters = starter_noX(state,H,js[1:1],n,envs), closures = closure(state,H,js,n,envs))

    ens = similar(js, eltype(eltype(state)))

    if n < first(js) #start with n and transfer through to first(js)
        G = starters[first(js)]
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
            GR = TransferMatrix(AL1,H[j],AL2) * closures[j]
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
    numberops = samesiteEcorrelator(st,H,O,Xs,n,1:L,envs,starters,closures)
    for i in 1:L
        ε[i,i] = numberops[i]
        ε[i+1:L,i] .= Ecorrelator(st,H,O,Xs,n,i,i+1:L,envs,starters,closures)
    end
    Hermitian(ε,:L)
end

function WavePacket(varmatrixs,normmatrix,m,k0,Δk,x0)
    halfL   = length(varmatrixs);
    _,Vh,ks = Vexact(floor(Int,0.5halfL),m);
    fp,Ew   = generate_fp(size(Vh)[2],m,k0,Δk,x0);
    #@show Ew
    wi      =  Vh*fp;
    Ens     = map(var->wi'*var*wi / (wi'*normmatrix*wi),varmatrixs);
    N       =  real(1-wi'*normmatrix*wi)
    #Ew = real(sum(Ens) - Ens[1]*100) #Ens[1] is stand-in for E0
    return N,Ens,Ew
end

L = 42
mass = 30.0
Delta_g = 0.0
v = 0.0
RAMPING_TIME = 5
dt = 0.01
v_max = 1.5
spatial_sweep = 10
truncation = 2.0
nr_steps = 1500
kappa = 0.6
frequency_of_saving = 5

σ = 0.1
x₀ = div(L,4)
x₀ = 0.5*(x₀-L) # check

@load "SanderDM_Thesis_2324/gs_mps_trunc_$(truncation)_mass_$(mass)_v_0.0_Delta_g_$(Delta_g)" mps
@load "SanderDM_Thesis_2324/gs_mps_wo_symmetries_trunc_$(truncation)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)" mps

if mass == 0.0
    @load "N_20_mass_0_occupation_matrices_new" corr corr_energy
elseif mass == 30.0
    @load "N_20_mass_30_occupation_matrices_new" corr corr_energy
else
    @assert 0 == 1 
end

N = div(L,2)-1
X = [(2*pi)/N*i - pi for i = 0:N-1]
X_new = [(2*pi)/N*i for i = 0:N-1]
Es = [energy(k,mass) for k in X]

S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], ℂ^2, ℂ^2)


H = get_thirring_hamiltonian_only_m(mass)
H_m_0 = get_thirring_hamiltonian_only_m(0.0)
unit_tensor = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], ℂ^2, ℂ^2)
H_unit = @mpoham sum((unit_tensor){i} for i in vertices(InfiniteChain(2)))

envs = environments(mps,H)

middle = (2*im)*S_z_symm
# same_site = samesiteEcorrelator_newest(mps, H, S⁻, middle, 1:2*N)
same_site = samesiteEcorrelator_newest(mps, H, S⁺, -middle, 1:2*N)
same_site_unit = samesiteEcorrelator_newest(mps, H_unit, S⁺, -middle, 1:2*N)

# EcorrelationMatrix(mps,L,0,H,S⁻,Xs,environments(mps,H))

# Gp = further_Ecorrelator_newest(mps, H_unit, S⁺, (-2*im*S_z_symm), 1, 2*N)
# Gm = further_Ecorrelator_newest(mps, H_unit, S⁻, (2*im*S_z_symm), 1, 2*N)

Gm_H = further_Ecorrelator_newest(mps, H_m_0, S⁻, (-2*im*S_z_symm), 1, 2*N)

break
# Starting from symmetries

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

function convert_to_array(tensors::PeriodicArray{A}) where {A <: TensorMap}
    return [convert_to_array(tensor) for tensor in tensors]
end

function remove_symmetries(mps::InfiniteMPS)
    ALS = PeriodicArray(convert_to_array(mps.AL))
    ARS = PeriodicArray(convert_to_array(mps.AR))
    ACS = PeriodicArray(convert_to_array(mps.AC))
    CRS = PeriodicArray(convert_to_array(mps.CR))
    return InfiniteMPS(ALS, ARS, CRS, ACS)
end

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)    
Plus_space = U1Space(1 => 1)
S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace ⊗ Plus_space)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace)
S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)

@load "SanderDM_Thesis_2324/gs_mps_trunc_$(truncation)_mass_$(mass)_v_0.0_Delta_g_$(Delta_g)" mps
Xs = get_X_tensors(mps.AL) # geeft hetzelfde indiend AC of AR



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

break
# Without MPOHam

J = [1.0 -1.0]
Sz_plus_12 = S_z() + 0.5*id(domain(S_z()))

S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)
@tensor operator_threesite[-1 -2 -3; -4 -5 -6] := S⁺[-1; -4] * S_z()[-2; -5] * S⁻[-3; -6]
@tensor operator_threesite_final[-1 -2 -3; -4 -5 -6] := operator_threesite[-1 -2 -3; -4 -5 -6] - operator_threesite[-3 -2 -1; -6 -5 -4]

O_mass = am_tilde_0 * Sz_plus_12
O_hop = (-1) * (S_xx() + S_yy())
@tensor O_int[-1 -2; -3 -4] := (Delta_g) * Sz_plus_12[-1; -3] * Sz_plus_12[-2; -4]




break
#tests

mass_operator = mass*(S_z() + 0.5*id(domain(S_z())))
mass_operator_conj = mass*(0.5*id(domain(S_z())) - S_z())

for w = 1:length(mps)
    test1 = @tensor mps.AC[w][1 2; 4] * mass_operator[3; 2] * conj(mps.AC[w][1 3; 4])
    test2 = @tensor mps.AC[w][1 2; 6] * S⁺[3; 2] * mass_operator[4; 3] * S⁻[5; 4] * conj(mps.AC[w][1 5; 6])
    println(test1)
    println(test2)
    test3 = @tensor mps.AC[w][1 2; 4] * mass_operator_conj[3; 2] * conj(mps.AC[w][1 3; 4])
    test4 = @tensor mps.AC[w][1 2; 6] * S⁺[3; 2] * mass_operator_conj[4; 3] * S⁻[5; 4] * conj(mps.AC[w][1 5; 6])
    println(test3)
    println(test4)
end

break
Ecorr = Ecorrelator(mps, H, S⁺, Xs, 4, 5, 6:10, envs)
Ecorr2 = Ecorrelator(mps, H, S⁺, Xs, 2, 5, 6:10, envs)

unit_tensor = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], pspace, pspace)
H_unit = @mpoham sum((unit_tensor){i} for i in vertices(InfiniteChain(2)))

same_site = samesiteEcorrelator_new(mps, H_unit, S⁻, Xs, 0, 1:2*N, environments(mps,H_unit))
same_site_H = samesiteEcorrelator_new(mps, H, S⁻, Xs, 0, 1:2*N, environments(mps,H))


break

ϵ = EcorrelationMatrix(mps, 2*N, 0, H_unit, S⁺, Xs_inverse, environments(mps,H_unit); new = false)
ϵ = ComplexF64.(copy(ϵ))
for i = 1:2*N
    for j = 1:2*N
        if i != j
            ϵ[i,j] /= -(abs(i-j)+1)
        end
    end
end

ϵ_energy = EcorrelationMatrix(mps, 2*N, 0, H, S⁺, Xs_inverse, environments(mps,H); new = false)
ϵ_energy = ComplexF64.(copy(ϵ_energy))
for i = 1:2*N
    for j = 1:2*N
        if i != j
            ϵ_energy[i,j] /= -(abs(i-j)+1)
        end
    end
end

σ = 0.1
x₀ = div(N,2)

(V₊,V₋) = V_matrix(X, mass)
occ_matrix_energy = adjoint(V₊)*(corr_energy)*(V₊)
occ_matrix = adjoint(V₊)*corr*(V₊)

occ = zeros(Float64, N)
occ_energy = zeros(Float64, N)
for (i,k₀) in enumerate(X)
    array = gaussian_array(X, k₀, σ, x₀)
    occupation_number = ((array)*occ_matrix*adjoint(array))
    occupation_number_energy = (array)*(occ_matrix_energy)*adjoint(array) / ((array)*occ_matrix*adjoint(array))
    if (abs(imag(occupation_number)) > 1e-2)
        println("Warning, complex number for occupation number: $(occupation_number)")
    end
    occ[i] = real(occupation_number)
    occ_energy[i] = real(occupation_number_energy)
end