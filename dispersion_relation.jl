using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using JSON

all_data = JSON.parsefile("SanderDM_Thesis_2324/renormalization_data.json")
v_renorms, mass_renorms, c_renorms = all_data

function theoretical_maximum(m, c)
    nodp = 1000
    k_values = LinRange(-pi/2,pi/2,nodp)
    dk = pi/(nodp-1)
    E_values = [sqrt(m^2*c^4+sin(k)^2*c^2) for k in k_values]
    v_values = [(E_values[i+1]-E_values[i])/dk for i = 1:nodp-1]
    return maximum(v_values)
end

am_tilde_0 = 0.3
Delta_g = 0.0
v = 0.0
trunc = 3.0

# v_eff = v_renorms[string(am_tilde_0)][string(Delta_g)] * v
# am_tilde_0_eff = mass_renorms[string(am_tilde_0)][string(Delta_g)] * am_tilde_0
# c_eff = c_renorms[string(am_tilde_0)][string(Delta_g)] * am_tilde_0


# name =  "SanderDM_Thesis_2324/Dispersion_Delta_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)_trunc_$(trunc)_all_sectors"
# f = jldopen(name, "r")
# println(keys(f))


# @load "SanderDM_Thesis_2324/Dispersion_Delta_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)_trunc_$(trunc)_all_sectors" energies Bs anti_energies anti_Bs zero_energies zero_Bs bounds_k
@load "SanderDM_Thesis_2324/Dispersion_Delta_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)_trunc_$(trunc)_all_sectors" gs_energy energies Bs anti_energies anti_Bs zero_energies zero_Bs bounds_k

data_points = 35
k_values = LinRange(-bounds_k, bounds_k,data_points)
dk = 2*bounds_k/(data_points-1)

energies_real = [real(e) for e in energies]
anti_energies_real = [real(e) for e in anti_energies]
zero_energies_real = [real(e) for e in zero_energies]

plotting = true
if plotting
    plt = plot(k_values, energies_real, label="particles", xlabel="k", ylabel="energies", linewidth=2)
    plot!(k_values, anti_energies_real, label="anti-particles", linewidth=2)
    plot!(k_values, zero_energies_real, label="zero-particles", linewidth=2)
    plot!(k_values, [sqrt(am_tilde_0^2+sin(k)^2) for k in k_values], label="theory")
    plot!()
    display(plt)
end

Es = [energies_real, anti_energies_real, zero_energies_real]
for j = 1:3
    max_derivative = 0.0
    max_index = -1
    for i = 1:length(Es[j])-1
        derivative_local = (Es[j][i+1]-Es[j][i])/dk
        if derivative_local > max_derivative
            max_derivative = derivative_local
            max_index = 1
        end
    end
    println("max_der = $(max_derivative) for i = $(max_index)")
end

println([(Es[1][i+1]-Es[1][i])/dk for i = 1:length(Es[1])-1])

println("theoretical max for c = 1 is $(theoretical_maximum(am_tilde_0,1))")
println("theoretical max for renormalized c is $(theoretical_maximum(am_tilde_0,c_renorms[string(am_tilde_0)][string(-0.44999999999999996)]))")
