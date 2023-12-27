using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_thirring_hamiltonian_symmetric_separate.jl")
include("get_groundstate.jl")

function spatial_ramping(i, i_start, i_end)
    if i < i_start
        return 0
    elseif i < i_end
        return (i-i_start)/(i_end-i_start)
    end
    return 1.0
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

(Hopping_term, Mass_term, Interaction_term, Interaction_v_term) = get_thirring_hamiltonian_symmetric_separate(1.0, 1.0, 1.0)
H_without_v = Hopping_term + am_tilde_0*Mass_term + Delta_g*Interaction_term

(gs_mps, gs_envs) = get_groundstate(am_tilde_0, Delta_g, v, [20 50], truncation, 8.0)

Ψ = WindowMPS(gs_mps,N); # state is a windowMPS

# left Hamiltonian
Ht_left = H_without_v

# middle Hamiltonian
H_mid_base = repeat(H_left,N)
H_mid_v = @mpoham sum(spatial_ramping(i,i_start,i_end)*Interaction_v_term for i in vertices(InfiniteChain(N)))
Ht_mid = H_mid_base + TimedOperator(H_mid_v,f)

# right Hamiltonian
Ht_right = H_without_v + TimedOperator(Interaction_v_term,f) ;


WindowH = Window(Ht_left,Ht_mid,Ht_right);
WindowE = environments(Ψ,WindowH);


tobesaved = []
t_span    = 0:dt:t_end
alg       = TDVP(finalize=(t,Ψ,H,E)->myfinalize(t,Ψ,H,E,tobesaved,xWindow,g,gs,0.0))
Ψt = copy(Ψ)

window_dt,WindowE = time_evolve!(Ψt,WindowH,t_span,alg,WindowE;verbose=true,rightevolve=true)
