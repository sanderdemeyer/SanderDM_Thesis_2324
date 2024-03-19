include("get_thirring_hamiltonian.jl")
include("get_occupation_number_matrices.jl")

function avged(Es)
    return real.([(Es[2*i+1]+Es[2*i+2])/2 for i = 0:div(length(Es),2)-1])
end


#can be done more efficiently by excluding very small wi
function WavePacketMPS(st::Union{WindowMPS{A},InfiniteMPS{A}},Xs,O,wi,ham,envs=environments(st,ham);xi=0,xf=0) where A <: MPSTensor
# function WavePacketMPS()
    xi = 0
    xf = 0

    N = length(wi)
    #contruct the different B tensors (last one contains AC)
    ki = 1
    kf = N
    Bs = map(1:N) do i
        if i == kf
            # @tensor B[-1 -2; -3] := Xs[xi+i][-1;1] * st.AC[xi+i][1 2; -3] * O[-2;2]
            @tensor B[-1 -2; -3] := st.AC[xi+i][-1 2; -3] * O[-2;2]
        else
            # @tensor B[-1 -2; -3] := Xs[xi+i][-1;1] * st.AL[xi+i][1 2; -3] * O[-2;2]
            @tensor B[-1 -2; -3] := st.AL[xi+i][-1 2; -3] * O[-2;2]
        end
        wi[i]*B
    end



    println(domain(st.AL[xi+ki]))
    println(domain(Bs[ki]))

    #make the bigger mps tensors
    #first the left vector
    Va = domain(st.AL[xi+ki])[1]
    Vb = domain(Bs[ki])[1]
    u = isometry(storagetype(A), Va ⊕ Vb, Va)
    v = leftnull(u)
    @assert domain(v) == ⊗(Vb)
    ntensors = [st.AL[xi+ki] * u' + Bs[ki] * v']
    #now the middle ones
    #@info "Doing excitations over $(N) sites"
    for n in ki+1:kf-1
        A1 = MPSKit._transpose_front(u * MPSKit._transpose_tail(st.AL[xi+n]))
        A2 = MPSKit._transpose_front(v * MPSKit._transpose_tail(st.AL[xi+n]))
        B = MPSKit._transpose_front(u * MPSKit._transpose_tail(Bs[n]))
        Va = domain(A1)[1]
        Vb = domain(A2)[1]
        u = isometry(storagetype(A), Va ⊕ Vb, Va)
        v = leftnull(u)
        @assert domain(v) == ⊗(Vb)
        push!(ntensors,  A1 * u' + B * v' + A2 * v')
    end
    #@assert false
    #finally the right vector
    AN = MPSKit._transpose_front(v * MPSKit._transpose_tail(st.AC[xi+kf]))
    BN = MPSKit._transpose_front(u * MPSKit._transpose_tail(Bs[kf]))
    push!(ntensors,AN+BN)
    #this is very costly apparently
    println("type is $(typeof([st.AL[1:xi+ki-1]...,ntensors...,st.AR[xi+kf+1:xi+xf+N]...]))")
    wpstate =  FiniteMPS([st.AL[1:xi+ki-1]...,ntensors...,st.AR[xi+kf+1:xi+xf+N]...])
    wps_envs = environments(wpstate,ham,nothing,copy(leftenv(envs,1,st)),copy(rightenv(envs,xi+xf+N,st)))
    return wpstate,wps_envs
end

    # Bs2 = []
    # for i = 1:N
    #     if i == kf
    #         # @tensor B[-1 -2; -3] := Xs[xi+i][-1;1] * st.AC[xi+i][1 2; -3] * O[-2;2]
    #         @tensor B[-1 -2; -3] := st.AC[xi+i][-1 2; -3] * O[-2;2]
    #     else
    #         # @tensor B[-1 -2; -3] := Xs[xi+i][-1;1] * st.AL[xi+i][1 2; -3] * O[-2;2]
    #         @tensor B[-1 -2; -3] := st.AL[xi+i][-1 2; -3] * O[-2;2]
    #     end
    #     push!(Bs2,wi[i]*B)
    # end
    # println(norm(Bs2-Bs))

mass = 0.3
Delta_g = 0.0
v = 0.0

N = 60
k₀ = pi/5
σ = 2*(2*pi/N)
x₀ = div(N,2)

X = [(2*pi)/N*i - pi for i = 0:N-1]


@load "SanderDM_Thesis_2324/gs_mps_wo_symmetries_trunc_3.0_mass_0.3_v_0.0_Delta_g_0.0" mps
hamiltonian = get_thirring_hamiltonian(mass, Delta_g, v)

A = (typeof(mps)).parameters[1]
S⁺ = TensorMap([0.0+0.0im 1.0+0.0im; 0.0+0.0im 0.0+0.0im], ℂ^2, ℂ^2)

O = S⁺
Xs = fill(id(domain(S_z())), N)

G = gaussian_array(X, k₀, σ, x₀)
(V₊,V₋) = V_matrix(X, mass)

wi = G * V₊


(wpstate,wps_envs) = WavePacketMPS(mps, Xs, O, wi, hamiltonian)

dt = 2.0
t_end = 5.0
alg = TDVP()
t_span = 0:dt:t_end

E = expectation_value(wpstate,hamiltonian)
plt = plot(1:N, avged(E), label = "before")
display(plt)

(Ψ, envs) = time_evolve!(wpstate, hamiltonian, t_span, alg, wps_envs; verbose=true);

E = expectation_value(Ψ,hamiltonian)
println(E)

plt = plot(1:N, avged(E), label = "after")
display(plt)


plt = plot(1:N, norm.(transpose(G)))
display(plt)