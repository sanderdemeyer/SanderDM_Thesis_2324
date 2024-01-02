using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_thirring_hamiltonian_symmetric_separate.jl")
include("get_thirring_hamiltonian_window.jl")
include("get_groundstate.jl")

function spatial_ramping(i, i_start, i_end)
    if i < i_start
        return 0
    elseif i < i_end
        return (i-i_start)/(i_end-i_start)
    end
    return 1.0
end

function myfinalize(t,Ψ,H,env,tobesaved,O,gl,gs,gr,timefun)
	zs = map(i->expectation_value(Ψ,O,i),0:length(Ψ)+1)
	hs = [gl*timefun(t),(timefun(t).*gs)...,gr*timefun(t)]
	push!(tobesaved,[t,expectation_value(Ψ,H(t),env),zs,hs])
	return Ψ,env
end

N = 50 # Number of sites
i_start = 10
i_end = 40

dt = 0.01
max_time_steps = 600 #3000 #7000
t_end = 10

am_tilde_0 = 0.6
Delta_g = 0.0 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie
v = 0.0

f(t) = 0.1t
f0(t) = 0.0

# (Hopping_term, Mass_term, Interaction_term, Interaction_v_term) = get_thirring_hamiltonian_symmetric_separate(1.0, 1.0, 1.0)
(Hopping_term, Mass_term, Interaction_term, Interaction_v_term, Interaction_v_term_window) = get_thirring_hamiltonian_window(1.0, 1.0, 1.0, N, i_start, i_end)
H_without_v = Hopping_term + am_tilde_0*Mass_term + Delta_g*Interaction_term

truncation = 2.5
# (gs_mps, gs_envs) = get_groundstate(am_tilde_0, Delta_g, v, [20 50], truncation, 8.0)
(gs_mps, gs_envs) = get_groundstate(am_tilde_0, Delta_g, v, [5 10], truncation, 8.0; number_of_loops = 2)

Ψ = WindowMPS(gs_mps,N); # state is a windowMPS

# left Hamiltonian

Ht_left_td = 0*Interaction_v_term
Ht_left = SumOfOperators([H_without_v, Ht_left_td])
H_left = SumOfOperators([H_without_v, Ht_left_td])
# Ht_left = SumOfOperators([H_without_v, TimedOperator(Interaction_v_term,f0)])

# middle Hamiltonian
# H_mid_base = repeat(H_without_v,N)
H_mid_base = H_without_v

J = [1.0 -1.0]

#H_mid_v = @mpoham sum(spatial_ramping(linearize_index(i),i_start,i_end)*Interaction_v_term for i in vertices(FiniteChain(N)))
# H_mid_v = @mpoham sum(J[i]*Interaction_v_term for i in vertices(InfiniteChain(2)))

H_mid_v = Interaction_v_term
# H_mid_v = @mpoham sum(spatial_ramping(i,i_start,i_end)*Interaction_v_term for i in vertices(InfiniteChain(N)))
Ht_mid = SumOfOperators([H_mid_base, TimedOperator(H_mid_v,f)])
H_mid = SumOfOperators([H_mid_base, H_mid_v])

# right Hamiltonian
Ht_right = SumOfOperators([H_without_v, TimedOperator(Interaction_v_term,f)])
H_right = SumOfOperators([H_without_v, Interaction_v_term])


Ht_right = SumOfOperators([H_without_v, 0*Interaction_v_term])
Ht_mid = SumOfOperators([repeat(H_without_v,N), Interaction_v_term_window])
Ht_left = SumOfOperators([H_without_v, Interaction_v_term])

WindowH = Window(Ht_left,Ht_mid,Ht_right);
# WindowH = Window(H_left,H_mid,H_left);
WindowE = environments(Ψ,WindowH);

t = 0.0
tobesaved = []
t_span    = 0:dt:t_end
# alg       = TDVP(finalize=(t,Ψ,H,E)->myfinalize(t,Ψ,H,E,tobesaved,xWindow,g,gs,0.0))
# alg       = TDVP(expalg=Lanczos())
alg       = TDVP()
Ψt = copy(Ψ)

println(t_end/dt)
println(typeof(t_end/dt))

for n = 1:(t_end/dt)
    global t
    global WindowE
    global Ψt
    t += dt
    println("Timestep n = $(n), t = $(t)")

    Ht_right = SumOfOperators([H_without_v, 0*Interaction_v_term])
    Ht_mid = SumOfOperators([repeat(H_without_v,N), f(t)*Interaction_v_term_window])
    Ht_left = SumOfOperators([H_without_v, f(t)*Interaction_v_term])
    
    WindowH = Window(Ht_left,Ht_mid,Ht_right);
    WindowE = environments(Ψ,WindowH);
    
    Ψt,WindowE = timestep!(Ψt,WindowH,t,dt,alg,WindowE;leftevolve=true,rightevolve=false)
    # window_dt,WindowE = time_evolve!(Ψt,WindowH,t_span,alg,WindowE;verbose=true,rightevolve=true)
end