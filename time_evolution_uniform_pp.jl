using LinearAlgebra
using Base
using JLD2
using MPSKit
using MPSKitModels
using TensorKit
using Statistics
using Plots

function get_data(dt)
    Delta_g = 0.0
    v = 0.0
    m0 = 0.1
    m_end = 0.4
    D = 50
    truncation = 2.5
    RAMPING_TIME = 5
    t_end = (m_end-m0)*RAMPING_TIME*1.5

    @load "mass_time_evolution_dt_$(dt)_D_$(D)_trunc_$(truncation)_mass_$(m0)_to_$(m_end)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_tend_$(t_end)_v_$(v)" Es Es_time_evolve Es_local fidelities

    Es = [real(Es[i][0]+Es[i][1])/2 for i = 1:length(Es)]
    Es_time_evolve = [real(Es_time_evolve[i][0]+Es_time_evolve[i][1])/2 for i = 1:length(Es_time_evolve)]
    Es_local = [real(Es_local[i][0]+Es_local[i][1])/2 for i = 1:length(Es_local)]
    fidelities = [real(fidelities[i]) for i = 1:length(fidelities)]
    return Es, Es_time_evolve, Es_local, fidelities
end

(Es, Es_time_evolve, Es_local, fidelities) = get_data(0.02)
dt = 0.02
times = [j*dt for j = 1:length(Es_time_evolve)]
println("lengths are $(length(times)) and $(length(Es_time_evolve))")
plt = plot(times, Es_time_evolve, label="dt = 0.02")
for dt = [0.1 0.05 0.02 0.01]
    (Es, Es_time_evolve, Es_local, fidelities) = get_data(dt)
    times = [j*dt for j = 1:length(Es_time_evolve)]
    println("lengths are $(length(times)) and $(length(Es_time_evolve))")
    plot!(times, Es_time_evolve, label = "dt = $(dt)")
    if (dt == 0.01)
        plot!(times, Es_local, label = "exact")
    end
end

display(plt)
