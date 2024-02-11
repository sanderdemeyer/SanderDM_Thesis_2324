using LinearAlgebra
using Base
using JLD2
using MPSKit
using MPSKitModels
using TensorKit
using Statistics

include("get_thirring_hamiltonian_symmetric_separate.jl")
include("get_groundstate.jl")


MAX_V_VALUE = 2.0
RAMPING_TIME = 5
SIZE = 20

v_max = 1.1

function f(t)
    #return min(0.4, 0.2+t/10) # Fast
    return min(v_max, t/RAMPING_TIME)
end

function f_mass(t)
    #return min(0.4, 0.2+t/10) # Fast
    return min(0.4, 0.2+t/25) # Slow
end

# lineaire functie is wss beter.
# entanglement entropie of kleinste waardenin het spectrum (). als log(S) orde D ---> MPS niet meer goed.
# analytische tijdsevolutie? tot op Orde dt^2 juist

dt = 0.001
max_time_steps = 6000 #3000 #7000

# am_tilde_0 = f(0) # best gewoon wat groter.
Delta_g = 0.0 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie
v = 0.0
mass_v_sweep = 0.0

# g-->0: continuumlimiet, exact.
# zou kunnen dat v gehernomaliseerd wordt, dus v > 1 kan grondtoestand vinden maar v < veff
#TDVP vs 
#bond dimensie best dynamisch laten groeien. Later altijd grotere D nodig

# (Hopping_term, Mass_term, Interaction_term, Interaction_v_term) = get_thirring_hamiltonian_symmetric_separate(1, Delta_g, v) # For mass sweep
(Hopping_term, Mass_term, Interaction_term, Interaction_v_term) = get_thirring_hamiltonian_symmetric_separate(mass_v_sweep, Delta_g, 1.0) # For v sweep
H_without_mass = Hopping_term + Interaction_term + Interaction_v_term
H_without_v = Hopping_term + Mass_term + Interaction_term

H_base = Hopping_term + Mass_term + Interaction_term
H_v = Interaction_v_term

truncation = 2.5

println("started")
(mps, envs) = get_groundstate(mass_v_sweep, Delta_g, v, [20 50], truncation, 8.0)
#(mps, envs) = get_groundstate(mass_v_sweep, Delta_g, v, [20 50], truncation, 8.0)

tot_bonddim = dims((mps.AL[1]).codom)[1] + dims((mps.AL[1]).dom)[1]
println("Tot bonddim is $tot_bonddim")

@save "gs_mps_trunc_$(truncation)_mass_$(mass_v_sweep)_v_$(v)_Delta_g_$(Delta_g)" mps

assert(false)

fidelity = @tensor mps.AC[1][1 2; 3] * conj(mps.AC[1][1 2; 3])
println(fidelity)
println("Started with timesteps")


energies = zeros(ComplexF64, max_time_steps)
entropies = zeros(ComplexF64, max_time_steps)
fidelities = zeros(ComplexF64, div(max_time_steps,100)+1) #71
true_energies = zeros(ComplexF64, div(max_time_steps,100)+1) #71
true_energies_global = zeros(ComplexF64, div(max_time_steps,100)+1) #71


for j = 1:max_time_steps
    t = j*dt
    global mps
    global envs
    #(mps, envs) = timestep(mps, H_base + H_v*f(j), dt, TDVP()) OLD
    #(mps, envs) = timestep(mps, H_base + H_v*f(j), dt, TDVP(), envs) OLD
    #(mps, envs) = timestep(mps, H_without_mass + Mass_term*f(t), dt, TDVP()) #geen envs meegeven MASS SWEEP
    (mps, envs) = timestep(mps, H_without_v + Interaction_v_term*f(t), dt, TDVP()) #geen envs meegeven V SWEEP
    #gs_energy = expectation_value(mps, H_base + H_v*f(j))
    #gs_energy = expectation_value(mps, H_without_mass + Mass_term*f(t))
    gs_energy = expectation_value(mps, H_without_v + Interaction_v_term*f(t))
    println("Timestep j = $j has energy $(real(gs_energy[1]+gs_energy[2])/2)")
    energies[j] = real(gs_energy[1]+gs_energy[2])/2
    spectrum = entanglement_spectrum(mps)
    entropy = 0
    # for index = 1:length(spectrum)
    #     println(spectrum)
    #     println(spectrum[index])
    #     entropy = entropy -spectrum[index]*log(2, spectrum[index])
    # end
    # entropies[j] = entropy

    if j % 100 == 0
        #(groundstate_mps, groundstate_envs) = get_groundstate(f(t), Delta_g, v, [20 50], 3.0, 8.0, D_start = 0, mps_start = mps)
        #(groundstate_mps, groundstate_envs) = find_groundstate(mps,H_without_mass + Mass_term*f(t),VUMPS(maxiter = 50, tol_galerkin = 1e-12)) # For mass sweep
        (groundstate_mps_local, groundstate_envs_local) = find_groundstate(mps,H_without_v + Interaction_v_term*f(t),VUMPS(maxiter = 50, tol_galerkin = 1e-12)) # For v sweep
        true_gs_energy = expectation_value(groundstate_mps_local, H_without_v + Interaction_v_term*f(t))
        (groundstate_mps, envs) = get_groundstate(mass_v_sweep, Delta_g, f(t), [20 50], truncation, 8.0)
        true_gs_energy_global = expectation_value(groundstate_mps, H_without_v + Interaction_v_term*f(t))

        fidelity = @tensor groundstate_mps_local.AC[1][1 2; 3] * conj(mps.AC[1][1 2; 3])
        fidelities[div(j,100)] = fidelity
        true_energies[div(j,100)] = (true_gs_energy[1] + true_gs_energy[2])/2
        true_energies_global[div(j,100)] = (true_gs_energy_global[1] + true_gs_energy_global[2])/2
    end
    #timestep(Windowmps, window_operator, dt, TDVP(), envs) #leftevolve = True, rightevolve = False
end

# !: in place aanpassen van argument. zonder ! geeft kopie terug.
# timestep! vs timestep. ! past mps aan, zonder ! geeft hij een kopie terug (of omgekeerd)

println("Done with timesteps")

# H(x,t) = H * f(t)
# f(t): ramping function
# H, basic time-independent Window Hamiltoniaan


# @save "Thirring_time-evolution_uniform_adiabatic_m_0.6_delta_g_-0.3_trunc_4.5_new_v_sweep_slower_10000_higher_fidelity" energies entropies fidelities
@save "v_sweep_m_$(mass_v_sweep)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_max_$(v_max)_trunc_$(truncation)" energies entropies fidelities true_energies true_energies_global

println("Tot bonddim is $tot_bonddim")
print("first energy is ")
println(energies[1])
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