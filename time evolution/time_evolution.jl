using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics

include("get_thirring_hamiltonian_symmetric_separate.jl")

#=

am_tilde_0 = 0.3
Delta_g = -0.2
v = 0.0
iterations = [7, 10]
trunc = 2.5;

(mps, envs) = get_groundstate_symmetric(am_tilde_0, Delta_g, 0.0, iterations, trunc, number_of_loops = 1)
H_1 = get_thirring_hamiltonian_symmetric(am_tilde_0, Delta_g, 0.5)

=#

MAX_V_VALUE = 2.0
RAMPING_TIME = 5
SIZE = 20

function f(j)
    return j < RAMPING_TIME ? j*MAX_V_VALUE/RAMPING_TIME : MAX_V_VALUE
end

@load "Base_for_timedependent" mps envs H_1 


am_tilde_0 = 0.3
Delta_g = 0.0
(H_base, H_v) = get_thirring_hamiltonian_symmetric_separate(am_tilde_0, Delta_g, 1.0)

dt = 0.01
max_time_steps = 10
window_size = 7
w
window_state = FiniteMPS([mps.AC[0], mps.AR[1], mps.AR[0], mps.AR[1]])
window_state = FiniteMPS([mps.AC[1], mps.AR[0], mps.AR[1]])
window_state = FiniteMPS([i == 0 ? mps.AC[0] : mps.AR[i%2] for i = 1:window_size])
#window_state = FiniteMPS([mps.AL[0], mps.AC[1], mps.AR[0], mps.AR[1]])
#window_state = FiniteMPS(mps.AL, mps.AR, mps.AC, mps.CR)
windowmps = WindowMPS(mps, window_state, mps)
windowmps = WindowMPS(mps, window_size)
windowmps = WindowMPS(mps, SIZE)

H_middle = repeat(H_base, SIZE)
operator = Window(H_base, H_middle, H_base + 0.5*H_v)

println("Started with timesteps")
# timestep! vs timestep
for j = 1:max_time_steps
    println("Timestep j = $j")
    global mps
    global envs
    #(mps, envs) = timestep(mps, H_base + H_v*f(j*dt), dt, TDVP(), envs)
    (mps, envs) = timestep(mps, H_base + operator, dt, TDVP(), envs)
    timestep(Windowmps, window_operator, dt, TDVP(), envs) #leftevolve = True, rightevolve = False
end

# !: in place aanpassen van argument. zonder ! geeft kopie terug.

println("Done with timesteps")

# H(x,t) = H * f(t)
# f(t): ramping function
# H, basic time-independent Window Hamiltoniaan