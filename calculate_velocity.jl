using LinearAlgebra
using Base
using JLD2
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots
using LsqFit
using Polynomials

am_tilde_0 = 0.3
Delta_g = -0.15
v = 0.0
bounds_k = pi/2
trunc = 2.5

k = 1.0

k_values = LinRange(-bounds_k, bounds_k,35)

name = "Dispersion_Delta_m_$(am_tilde_0)_delta_g_$(Delta_g)_v_$(v)_trunc_$(trunc)_all_sectors_newv"

@load name gs_energy energies Bs anti_energies anti_Bs

energies .+= Delta_g
anti_energies .-= Delta_g

order = 16
f = fit(2*k_values, real.(energies)[:], order)

deriv = derivative(f, 1)

println("Velocity at k = $(k) is $(deriv(k))")

scatter(2*k_values, real.(energies)[:], markerstrokewidth=0, label="Data")
plot!(f, extrema(2*k_values)..., label="Interpolation")