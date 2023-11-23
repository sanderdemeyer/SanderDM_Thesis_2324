using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_thirring_hamiltonian_symmetric_separate.jl")
include("get_groundstate.jl")


MAX_V_VALUE = 2.0
RAMPING_TIME = 100
SIZE = 20

function f(t)
    return min(0.4, 0.2+t/20)
end

# lineaire functie is wss beter.
# entanglement entropie of kleinste waardenin het spectrum (). als log(S) orde D ---> MPS niet meer goed.
# analytische tijdsevolutie? tot op Orde dt^2 juist

dt = 0.001
max_time_steps = 6000

am_tilde_0 = f(0) # best gewoon wat groter.
Delta_g = 0.0 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie
v = 0

# g-->0: continuumlimiet, exact.
# zou kunnen dat v gehernomaliseerd wordt, dus v > 1 kan grondtoestand vinden maar v < veff
#TDVP vs 
#bond dimensie best dynamisch laten groeien. Later altijd grotere D nodig

(Hopping_term, Mass_term, Interaction_term, Interaction_v_term) = get_thirring_hamiltonian_symmetric_separate(1, Delta_g, v)
H_without_mass = Hopping_term + Interaction_term + Interaction_v_term

H_base = Hopping_term + Mass_term + Interaction_term
H_v = Interaction_v_term

(mps, envs) = get_groundstate(am_tilde_0, Delta_g, v, [20 50], 3.0, 8.0)

println("Started with timesteps")


energies = zeros(ComplexF64, max_time_steps)
entropies = zeros(ComplexF64, max_time_steps)
for j = 1:max_time_steps
    t = j*dt
    global mps
    global envs
    #(mps, envs) = timestep(mps, H_base + H_v*f(j), dt, TDVP())
    #(mps, envs) = timestep(mps, H_base + H_v*f(j), dt, TDVP(), envs)
    (mps, envs) = timestep(mps, H_without_mass + Mass_term*f(t), dt, TDVP()) #geen envs meegeven
    #gs_energy = expectation_value(mps, H_base + H_v*f(j))
    gs_energy = expectation_value(mps, H_without_mass + Mass_term*f(t))
    println("Timestep j = $j has energy $(real(gs_energy[1]+gs_energy[2])/2)")
    energies[j] = real(gs_energy[1]+gs_energy[2])/2
    spectrum = entanglement_spectrum(mps)
    entropy = 0
    for index = 1:length(spectrum)
        entropy = entropy -spectrum[index]*log(2, spectrum[index])
    end
    entropies[j] = entropy
    #timestep(Windowmps, window_operator, dt, TDVP(), envs) #leftevolve = True, rightevolve = False
end

# !: in place aanpassen van argument. zonder ! geeft kopie terug.
# timestep! vs timestep. ! past mps aan, zonder ! geeft hij een kopie terug (of omgekeerd)

println("Done with timesteps")

# H(x,t) = H * f(t)
# f(t): ramping function
# H, basic time-independent Window Hamiltoniaan


@save "Thirring_time-evolution_uniform_adiabatic_m_0.3_delta_g_0.0_new_mass_sweep_slow" energies entropies


#=
function timestep!(
    Ψ::WindowMPS,
    H::Window,
    t::Number,
    dt::Number,
    alg::TDVP,
    env::Window=environments(Ψ, H);
    leftevolve=true,
    rightevolve=true,
)

https://github.com/maartenvd/MPSKit.jl/blob/TimeDependent/src/operators/multipliedoperator.jl

TimedOperator(H1, f1) + TimedOperator(H2, f2) + ...
H1 +  TimedOperator(H2, f2) + ...

fi zijn voorgedefinieerde functies

expectation_value(phi, O(t), env)
environments hebben geen time-afhankelijkheid! Want bra(phi) sum(f_i H_i)


TDVP en TDVP2 gelijkaardig als VUMPS en changebonds gebruiken.

=#