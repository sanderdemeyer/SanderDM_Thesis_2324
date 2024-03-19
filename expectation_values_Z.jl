using LinearAlgebra
# using Base
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics


N = 20 # Number of sites
κ = 1.0
dt = 0.1
number_of_timesteps = 150 #3000 #7000
am_tilde_0 = 1.0
Delta_g = 0.0 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie
v = 0.0
v_max = 2.0
RAMPING_TIME = 5
FL = false
truncation = 1.5
frequency_of_saving = 3

# mass_window_time_evolution_variables_v_sweep_N_20_mass_1.0_delta_g_0.0_ramping_5_dt_0.1_nrsteps_150_vmax_2.0_kappa_1.0_trunc_1.5_savefrequency_3.jld2


# @load "mass_test_window_time_evolution_test_v_sweep_N_$(N)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_savefrequency_$(frequency_of_saving)" Es

@load "mass_window_time_evolution_variables_v_sweep_N_20_mass_1.0_delta_g_0.0_ramping_5_dt_0.1_nrsteps_150_vmax_2.0_kappa_1.0_trunc_1.5_savefrequency_3.jld2" Es Exp_Zs
