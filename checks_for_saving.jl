jldopen("window_time_evolution_variables_v_sweep_N_$(N)_mass_$(am_tilde_0)_delta_g_$(Delta_g)_ramping_$(RAMPING_TIME)_dt_$(dt)_nrsteps_$(number_of_timesteps)_vmax_$(v_max)_kappa_$(κ)_trunc_$(truncation)_savefrequency_$(frequency_of_saving).jld2", "r") do file
    t = 15.0
    println(keys(file["MPSs"]))
    # println(file["MPSs/$(t)"])
    println(file["MPSs/3.0"].middle.AC[1] - file["MPSs/15.0"].middle.AC[1])
end

