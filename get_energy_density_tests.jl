using LinearAlgebra
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using LaTeXStrings
using Random

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

function samesiteEcorrelator(st, H, O, Xs, n, range, envs)
    G = []
    for i = (range).+1
        left_env = leftenv(envs, i, st)
        right_env = rightenv(envs, i, st)
        value_onsite = 0.0
        for (j,k) in keys(H[i]) # = [(1,H[i].odim)] #
            # println("(j,k) = ($(j),$(k))")
            # println(left_env[j].codom)
            # println(left_env[j].dom)
            # println(H[i][j,k].codom)
            # println(H[i][j,k].dom)
            # j = 1
            # k = H[i].odim
            # value_onsite += @tensor left_env[j][16 5; 15] * Xs[mod1(i,length(Xs))][15;1] * st.AC[i][1 2; 13] * O[4; 2 7] * H[i][j,k][5 6; 4 12] * conj(O[6; 9 7]) * conj(Xs[mod1(i,length(Xs))][16;10]) * conj(st.AC[i][10 9; 11]) * right_env[k][13 12; 11]
            # if (((j,k) != (1,1)) && ((j,k) != (H[i].odim,H[i].odim)))
            value_onsite += @tensor left_env[j][16 5; 15] * Xs[mod1(i,length(Xs))][15;1] * st.AC[i][1 2; 13] * O[4; 2 7] * conj(O[6; 9 7]) * H[i][j,k][5 6; 4 12] * conj(Xs[mod1(i,length(Xs))][16;10]) * conj(st.AC[i][10 9; 11]) * right_env[k][13 12; 11]
            # end
        end
        push!(G, value_onsite)
    end
    return G
end


function EcorrelationMatrix(st,L,n,H,O,Xs,envs=environments(st,H))
    ε = zeros(ComplexF64,L,L);
    numberops = samesiteEcorrelator(st,H,O,Xs,n,1:L,envs)
    for i in 1:L
        ε[i,i] = numberops[i]
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

function get_energy_matrices_right_moving(mps, H, O, N, m, σ, x₀; datapoints = N, Delta_g = 0.0, v = 0.0)
    Xs = get_X_tensors(mps.AL)
    envs = environments(mps, H)
    corr_energy = transpose(EcorrelationMatrix(mps, 2*N, 0, H, O, Xs, envs))
    corr_energy_bigger = transpose(EcorrelationMatrix(mps, 2*N+2, 0, H, O, Xs, envs))
    @assert norm(corr_energy - corr_energy_bigger[1:end-2,1:end-2]) < 1e-10

    println("corr_energy is $(corr_energy)")
    for i = 1:2*N
        println("corr_energy[$(i),$(i)] = $(corr_energy[i,i])")
    end

    corr_energy = corr_energy_bigger[2:end-1,2:end-1]
    # corr_energy = corr_energy_bigger[1:end-2,1:end-2]

    @load "operators_for_occupation_number" S⁺ S⁻ S_z_symm
    corr = zeros(ComplexF64, 2*N, 2*N)
    for i = 2:2*N+1
        corr_bigger = correlator(mps, S⁺, S⁻, S_z_symm, i, 2*N+2)
        corr[i-1,:] = corr_bigger[2:2*N+1]
    end

    # corr = corr_energy #./ transpose(corr)

    X = [(2*pi)/N*i - pi for i = 0:N-1]
    X_finer = [(2*pi)/datapoints*i - pi for i = 0:datapoints-1]

    (V₊,V₋) = V_matrix(X, m)
    # occ_matrix_energy = adjoint(V₊)*corr_energy*(V₊)
    # occ_matrix = adjoint(V₊)*corr*(V₊)
    occ_matrix_energy = adjoint(V₋)*corr_energy*(V₋)
    occ_matrix = adjoint(V₋)*corr*(V₋)

    occ = zeros(Float64, datapoints)
    for (i,k₀) in enumerate(X)
        array = gaussian_array(X, k₀, σ, x₀)
        occupation_number = (array)*occ_matrix_energy*adjoint(array) / ((array)*occ_matrix*adjoint(array))
        if (abs(imag(occupation_number)) > 1e-2)
            println("Warning, complex number for occupation number: $(occupation_number)")
        end
        occ[i] = real(occupation_number)
    end

    return (X_finer, occ)
end

L = 100
mass = 0.3
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
x₀ = 1

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


H0 = get_thirring_hamiltonian_symmetric(mass, Delta_g, v; new = true)
envs = environments(mps,H)
Xs = get_X_tensors(mps.AL)

unit_tensor = TensorMap([1.0+0.0im 0.0+0.0im; 0.0+0.0im 1.0+0.0im], pspace, pspace)
H_unit = @mpoham sum((unit_tensor){i} for i in vertices(InfiniteChain(2)))

# Ecorr2 = Ecorrelator(mps, H, S⁺, Xs, 2, 5, 6:10, envs)

H = H0
O = S⁻

#using the new functions
corr_energy = transpose(EcorrelationMatrix(mps, 2*N, 0, H, O, Xs, envs)[:,:])
corr_energy_bigger = transpose(EcorrelationMatrix(mps, 2*N+2, 0, H, O, Xs, envs)[:,:])[2:end-1,2:end-1]
# @assert norm(corr_energy - corr_energy_bigger[1:end-2,1:end-2]) < 1e-10

# println("corr_energy is $(corr_energy)")
# for i = 1:2*N
#     println("corr_energy[$(i),$(i)] = $(corr_energy[i,i])")
# end

# corr_energy = corr_energy_bigger[2:end-1,2:end-1]
# corr_energy = corr_energy_bigger[1:end-2,1:end-2]

corr_energy_divided = copy(corr_energy)
corr_energy_bigger_divided = copy(corr_energy_bigger)
for i = 1:2*N
    for j = 1:2*N
        if i != j
            corr_energy_divided[i,j] /= (abs(j-i)+1) * (-1)^(abs(j-i)+1)
            corr_energy_bigger_divided[i,j] /= (abs(j-i)+1) * (-1)^(abs(j-i)+1)
        end
    end
end

S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], trivspace ⊗ pspace, pspace ⊗ Plus_space)
S⁻ = TensorMap([0.0+0.0im 0.0+0.0im; 1.0+0.0im 0.0+0.0im], Plus_space ⊗ pspace, pspace ⊗ trivspace)
S_z_symm = TensorMap([0.5+0.0im 0.0+0.0im; 0.0+0.0im -0.5+0.0im], pspace, pspace)

corr = zeros(ComplexF64, 2*N, 2*N)
corr_old = zeros(ComplexF64, 2*N, 2*N)
for i = 2:2*N+1
    corr_bigger = correlator(mps, S⁺, S⁻, S_z_symm, i, 2*N+2)
    corr_old[i-1,:] = corr_bigger[2:2*N+1]
end

for i = 1:2*N
    corr_smaller = correlator(mps, S⁺, S⁻, S_z_symm, i, 2*N)
    corr[i,:] = corr_smaller
end

corr_new = copy(corr)
for i = 1:2*N
    for j = 1:2*N
        if i != j
            corr_new[i,j] /= (abs(j-i)+1)
        end
    end
end


break


X = [(2*pi)/N*i - pi for i = 0:N-1]
theoretical_energies = [(-1+2*(k<0.0))*sqrt(mass^2+sin(k/2)^2) for k in X]

(V₊,V₋) = V_matrix(X, mass)
# occ_matrix_energy = adjoint(V₊)*corr_energy*(V₊)
# occ_matrix = adjoint(V₊)*corr*(V₊)
occ_matrix_energy = adjoint(V₋)*corr_energy_bigger_divided*(V₋)
occ_matrix = adjoint(V₋)*corr_old*(V₋)

occ = zeros(Float64, N)
for (i,k₀) in enumerate(X)
    array = gaussian_array(X, k₀, σ, x₀)
    println(array)
    occupation_number = (array)*(occ_matrix_energy)*adjoint(array) # / (((array)*occ_matrix*adjoint(array)))
    if (abs(imag(occupation_number)) > 1e-2)
        println("Warning, complex number for occupation number: $(occupation_number)")
    end
    occ[i] = real(occupation_number)
end

plt = scatter(X_finer, occ, label = "data")
# scatter!(X, theoretical_energies, label = "theoretical")
# xlims!(0, maximum(X_finer)*1.1)
# ylims!(-1.2, 1.2)
display(plt)
