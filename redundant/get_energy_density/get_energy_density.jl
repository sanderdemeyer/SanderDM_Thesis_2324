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
include("get_thirring_hamiltonian_symmetric.jl")
include("get_occupation_number_matrices.jl")

# transfer, but the upper A is an excited tensor and there is an mpo leg being passed through
function transfer_left(v::MPSKit.MPSTensor, A::MPSKit.MPOTensor, Ab::MPSKit.MPSTensor)
    @plansor t[-1 -2; -3 -4] := v[1 3; 4] * A[4 5; -3 -4] * τ[3 2; 5 -2] * conj(Ab[1 2; -1])
end

function transfer_right(v::MPSKit.MPSTensor, A::MPSKit.MPOTensor, Ab::MPSKit.MPSTensor)
    @plansor t[-1 -2; -3 -4] := A[-1 4; -3 5] * τ[-2 3; 4 2] * conj(Ab[-4 3; 1]) * v[5 2; 1]
end

# transfer, but the lower A is an excited tensor and there is an mpo leg being passed through
function transfer_left(v::MPSKit.MPSTensor, A::MPSKit.MPSTensor, Ab::MPSKit.MPOTensor)
    @plansor t[-1 -2; -3 -4] := v[1 3; 4] * A[4 5; -4] * τ[3 2; 5 -2] * conj(Ab[1 2; -3 -1])
end

function transfer_right(v::MPSKit.MPSTensor, A::MPSKit.MPSTensor, Ab::MPSKit.MPOTensor)
    @plansor t[-1 -2; -3 -4] := A[-1 4; 5] * τ[-2 3; 4 2] * conj(Ab[-4 3; -3 1]) * v[5 2; 1]
end

#mpo transfer, but with A an excitation-tensor
function transfer_left(v::MPSKit.MPSTensor, O::MPSKit.MPOTensor, A::MPSKit.MPOTensor, Ab::MPSKit.MPSTensor)
    @plansor t[-1 -2; -3 -4] := v[4 2; 1] * A[1 3; -3 -4] * O[2 5; 3 -2] * conj(Ab[4 5; -1])
end
function transfer_right(v::MPSKit.MPSTensor, O::MPSKit.MPOTensor, A::MPSKit.MPOTensor, Ab::MPSKit.MPSTensor)
    @plansor t[-1 -2; -3 -4] := A[-1 4; -3 5] * O[-2 2; 4 3] * conj(Ab[-4 2; 1]) * v[5 3; 1]
end

#mpo transfer, but with A and Ab an excitation-tensor
function transfer_left(v::MPSKit.MPSTensor, O::MPSKit.MPOTensor, A::MPSKit.MPOTensor, Ab::MPSKit.MPOTensor)
    @tensor t[-1 -2; -3 -4] := v[4 2; 1] * A[1 3; -3 -4] * O[2 5; 3 -2] * conj(Ab[4 5; -1])
end
function transfer_right(v::MPSKit.MPSTensor, O::MPSKit.MPOTensor, A::MPSKit.MPOTensor, Ab::MPSKit.MPOTensor)
    @tensor t[-1 -2; -3] := A[-1 4; 6 5] * O[-2 2; 4 3] * conj(Ab[-4 2; 6 1]) * v[5 3; 1]
end

#mpo transfer, but with Ab an excitation-tensor
function my_transfer_left(v, O, A, Ab)
    @tensor t[-1 -2; -3 -4 -5 -6] := v[4 -2; 1] * A[1 -6; -4] * conj(Ab[4 -5; -3 -1])
    @plansor t[-1 -2; -3 -4] := v[4 2; 1] * A[1 3; -4] * O[2 5; 3 -2] * conj(Ab[4 5; -3 -1])
end
function my_transfer_right(v::MPSKit.MPSTensor, O::MPSKit.MPOTensor, A::MPSKit.MPSTensor, Ab::MPSKit.MPOTensor)
    @plansor t[-1 -2; -3 -4] := A[-1 4; 5] * O[-2 2; 4 3] * conj(Ab[-4 2; -3 1]) * v[5 3; 1]
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
        # G = PartTransferLeft(state,H,n,leftenv(envs,n,state))
        # G = leftenv(state, H, n) * TransferMatrix(state.AR[n], H[n], state.AR[n])
        # G = G * TransferMatrix(state.AR[n+1:i-1],H[n+1:i-1],state.AR[n+1:i-1])
        @plansor ACx[-1 -2; -4 -3] := Xs[mod1(i,length(Xs))][-1;1]*state.AC[i][1 2;-4]*O[-2;2 -3]
        ACx = permute(ACx, (1,2), (4,3))
        G = leftenv(envs, i, state)
        G = G * TransferMatrix(ACx,H[i],state.AC[i])
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
            if n < prevj #n has already been done. Gebruik dit
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
        if n < prevj # Gebruik dit
            @tensor ARx[-1 -2; -4 -3] := Xs[mod1(j,length(Xs))][-1;1]*state.AR[j][1 2;-4]*O[-2;2 -3]
            ARx = permute(ARx, (1,2),(4,3))
            r_env = rightenv(envs,j,state)
            tf = TransferMatrix(state.AR[j],H[j],ARx)
            GR = tf * r_env
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
            curr += @tensor G[k][4 1; 2 3] * GR[k][3 1; 2 4]
            # curr += @tensor G[k][1 2; 3 4] * GR[k][1 2; 3 4]
        end
        ens[jn] = curr
        prevj = j

    end
    return ens
end

function energy(k, m)
    if (k < 0.0)
        return -sqrt(m^2+sin(k/2)^2)
    else
        return sqrt(m^2+sin(k/2)^2)
    end
end

function samesiteEcorrelator_old(st, H, O, Xs, n, range, envs)
    G = []
    for i = (range)#.+1
        left_env = leftenv(envs, i, st)
        right_env = rightenv(envs, i, st)
        value_onsite = 0.0
        for (j,k) = [(1,H[i].odim)] # in keys(H[i]) #
            # j = 1
            # k = H[i].odim
            value_onsite = @tensor left_env[j][16 5; 15] * Xs[mod1(i,length(Xs))][15;1] * st.AC[i][1 2; 13] * O[4; 2 7] * H[i][j,k][5 6; 4 12] * conj(O[6; 9 7]) * conj(Xs[mod1(i,length(Xs))][16;10]) * conj(st.AC[i][10 9; 11]) * right_env[k][13 12; 11]
        end
        push!(G, value_onsite)
    end
    return G
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
            @tensor AR1[-1 -2; -3 -4] := Xs[mod1(j,length(Xs))][-1;1]*state.AR[j][1 2;-4]*O[-2;2 -3]
            @tensor AR2[-1 -2; -3 -4] := Xs[mod1(j,length(Xs))][-1;1]*state.AR[j][1 2;-4]*O[-2;2 -3]
            GR =  TransferMatrix(AR1,H[j],AR2) * rightenv(envs,j,state)
            GR = my_transfer_left_2e()
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


function EcorrelationMatrix(st,L,n,H,O,Xs,envs=environments(st,H); new = false)
    ε = zeros(ComplexF64,L,L);
    if new
        starters = [leftenv(envs,1,st) for k = 1:L]
        closure = [rightenv(envs,L,st) for k = 1:L]
        numberops = samesiteEcorrelator(st,H,O,Xs,n,1:L,envs, starters, closure)
    else
        # numberops = samesiteEcorrelator_old(st,H,O,Xs,n,1:L,envs)
    end
    for i in 1:L
        ε[i,i] = 0.0 #numberops[i]
        ε[i+1:L,i] .= Ecorrelator(st,H,O,Xs,n,i,i+1:L,envs)#,starters,closures)
    end
    return Hermitian(ε,:L)
end

function WavePacket(varmatrixs,normmatrix,m,k0,Δk,x0) # normmatrix is <c_i^dagger c_j>
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

function get_energy_matrices_right_moving(mps, H, O, N, m, σ, x₀; datapoints = N, Delta_g = 0.0, v = 0.0, new = false)
    Xs = get_X_tensors(mps.AL)
    envs = environments(mps, H)
    corr_energy = transpose(EcorrelationMatrix(mps, 2*N, 0, H, O, Xs, envs; new = new))
    corr_energy_bigger = transpose(EcorrelationMatrix(mps, 2*N+2, 0, H, O, Xs, envs; new = new))
    @assert norm(corr_energy - corr_energy_bigger[1:end-2,1:end-2]) < 1e-10

    # corr_energy = corr_energy_bigger[2:end-1,2:end-1]
    corr_energy = corr_energy_bigger[1:end-2,1:end-2]

    # occupation matrix for H = H_unit: 1/2 op diagonaal
    # fit beter voor lage momenta
    # check convolutie
    # grotere momenta wavepackets die naar links of rechts bewegen.
    # wiki van Pauli matrices: exact diagonalisation die niet singulier is

    for i = 1:2*N
        for j = 1:2*N
            if (i != j)
                corr_energy[i,j] /= (abs(j-i)+1) * (-1)^(i+j)
            end
        end
    end

    @load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
    corr = zeros(ComplexF64, 2*N, 2*N)
    for i = 2:2*N+1
        corr_bigger = correlator(mps, S⁺, S⁻, S_z_symm, i, 2*N+2)
        corr[i-1,:] = corr_bigger[2:2*N+1]
    end


    println(corr_energy./ transpose(corr))

    X = [(2*pi)/N*i - pi for i = 0:N-1]
    X_finer = [(2*pi)/datapoints*i - pi for i = 0:datapoints-1]

    (V₊,V₋) = V_matrix(X, m)
    occ_matrix_energy = adjoint(V₊)*corr_energy*(V₊)
    occ_matrix = adjoint(V₊)*corr*(V₊)
    # occ_matrix_energy = adjoint(V₋)*corr_energy*(V₋)
    # occ_matrix = adjoint(V₋)*corr*(V₋)

    occ = zeros(Float64, datapoints)
    for (i,k₀) in enumerate(X)
        array = gaussian_array(X, k₀, σ, x₀)
        occupation_number = (array)*occ_matrix_energy*adjoint(array)# / ((array)*occ_matrix*adjoint(array))
        if (abs(imag(occupation_number)) > 0.03)
            println("Warning, complex number for occupation number: $(occupation_number)")
        end
        occ[i] = real(occupation_number)
    end

    return (X_finer, occ)
end

L = 42
mass = 00.0
Delta_g = 0.0
v = 0.0
RAMPING_TIME = 5
dt = 0.01
v_max = 1.5
spatial_sweep = 10
truncation = 2.5
nr_steps = 1500
kappa = 0.6
frequency_of_saving = 5

σ = 0.1
x₀ = div(L,4)
x₀ = 0.5*(x₀-L) # check

@load "SanderDM_Thesis_2324/gs_mps_trunc_$(truncation)_mass_$(mass)_v_0.0_Delta_g_$(Delta_g)" mps


N = div(L,2)-1
X = [(2*pi)/N*i - pi for i = 0:N-1]
X_new = [(2*pi)/N*i for i = 0:N-1]
Es = [energy(k,mass) for k in X]
# X = [2*((2*pi)/N*i - pi) for i = 0:N-1]

spin = 1//2
pspace = U1Space(i => 1 for i in (-spin):spin)
Plus_space = U1Space(1 => 1)
Min_space = U1Space(-1 => 1)
trivspace = U1Space(0 => 1)
S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace ⊗ Plus_space)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], pspace, pspace ⊗ Min_space)

function gauss(k, k₀, σ)
    return (1/σ*sqrt(2*pi)) * exp(-(k-k₀)^2/(2*σ^2)) / (2*pi)
end

theoretical_energies = [(-1+2*(k<0.0))*sqrt(mass^2+sin(k)^2) for k in X]
energies_convoluted = [quadgk(x -> (-1+2*(x<0.0))*sqrt(mass^2+sin(x)^2)*gauss(k, x, σ), -pi, k, pi; atol = 1e-10)[1] for k in X]

H = get_thirring_hamiltonian_symmetric(mass, Delta_g, v; new = true)
# spin = 1//2
# pspace = U1Space(i => 1 for i in (-spin):spin)
# S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)
# Hopping_term = (-1) * @mpoham sum(S_xx(){i, i + 1} + S_yy(){i, i + 1} for i in vertices(InfiniteChain(2)))
# nothing_term = @mpoham sum(0.1*S_z_symm{i} for i in vertices(InfiniteChain(2)))
# Hop = Hopping_term + nothing_term
# Hopping_term[1][1,4] = mass*Sz_plus_12

# Sz_plus_12 = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 0.0+0.0im], pspace, pspace)
# Mass_term = (mass) * repeat(MPOHamiltonian([Sz_plus_12]),2)
# for k = 1:Mass_term.odim-1
#     Mass_term[i][k,Mass_term.odim] *= -1
# end
# H = Hopping_term + Mass_term

envs = environments(mps,H)
Xs = get_X_tensors(mps.AL) # geeft hetzelfde indien AC of AR

data = [convert(Array, x) for x in Xs]
ds = [size(dat)[1] for dat in data]
Xs_asymm = [TensorMap(data[i], ℂ^ds[i], ℂ^ds[i]) for i = 1:length(Xs)]


# @load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
# Ecorr = Ecorrelator(mps, H, S⁺, Xs, 4, 5, 6:10, envs)
# Ecorr2 = Ecorrelator(mps, H, S⁺, Xs, 2, 5, 6:10, envs)

unit_tensor = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], pspace, pspace)
H_unit = @mpoham sum((unit_tensor){i} for i in vertices(InfiniteChain(2)))

samesite_unit = samesiteEcorrelator_new(mps, H_unit, S⁺, Xs, 0, 1:2*N, environments(mps,H_unit))
samesite = samesiteEcorrelator_new(mps, H, S⁺, Xs, 0, 1:2*N, environments(mps,H))

@load "N_20_mass_0_occupation_matrices_new" corr corr_energy


break

ϵ = EcorrelationMatrix(mps, N, 0, H_unit, S⁺, Xs, envs; new = false)

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


break

# ϵ_new = EcorrelationMatrix(mps, N, 0, H, S⁺, Xs, envs; new = true)

unit_tensor = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], pspace, pspace)
H_unit = @mpoham sum((unit_tensor){i} for i in vertices(InfiniteChain(2)))

(X_finer, occ) = get_energy_matrices_right_moving(mps, H, S⁺, N, mass, σ, x₀; datapoints = N, Delta_g = 0.0, v = 0.0)

offset = 0
offset = real(mean(expectation_value(mps, H)))

plt = scatter(X_finer, occ, label = "data")
scatter!(X, theoretical_energies, label = "theoretical")
# xlims!(0, maximum(X_finer)*1.1)
# ylims!(-1.2, 1.2)
display(plt)
