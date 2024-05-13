include("get_correlator.jl")

# @load "SanderDM_Thesis_2324/gs_mps_trunc_2.5_mass_0.0_v_0.0_Delta_g_0.0" mps
@load "SanderDM_Thesis_2324/window_time_evolution_v_sweep_N_30_mass_0.0_delta_g_0.0_ramping_5_dt_0.01_nrsteps_1500_vmax_1.1_kappa_0.5_trunc_1.5_savefrequency_10" MPSs
N = div(30-2,2)
m = 0.0
v = 0.0
delta_g = 0.0

for i = 1:length(MPSs)
    if (i == 1) || (i == length(MPSs)-1)
        mps = MPSs[i].window
        (X, N̂, Ê) = get_occupation_number(mps, N, m, v)

        plt = plot(X, N̂, xlabel = "k", ylabel = L"$\left<\hat{N}\right>$")
        title!("Occupation number for N = $(N) and i = $(i)")
        display(plt)
    end
end