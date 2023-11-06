function phase_diagram_symmetric(mass, delta_g, v)

    trunc_list = [1.5 2.0 2.5 3.0 3.5 4.0 4.5]

    Energies = Array{Float64, 2}(undef, 1, length(trunc_list)+1)
    Corr_lengths = Array{Float64, 2}(undef, 1, length(trunc_list)+1)
    Bond_dims = Array{Float64, 2}(undef, 1, length(trunc_list)+1)

    println("Truncation is 10^(-1.0)")
    (gs_energy, corr_length, bonddim, groundstate) = get_groundstate_energy_dynamic(mass, delta_g, v, 1.0, 4.5, D_start = 3)
    Energies[1] = gs_energy
    Corr_lengths[1] = corr_length
    Bond_dims[1] = bonddim
    for i = 1:length(trunc_list)
        #=
        global groundstate
        global gs_energy
        global corr_length
        global bonddim
        =#
        trunc = trunc_list[i]
        println("Truncation is 10^(-$trunc)")
        (gs_energy, corr_length, bonddim, groundstate) = get_groundstate_energy_dynamic(mass, delta_g, v, trunc, trunc+3.0, D_start = 0, mps_start = groundstate)
        Energies[i+1] = gs_energy
        Corr_lengths[i+1] = corr_length
        Bond_dims[i+1] = bonddim
    end

    @save "Phase_diagram_correlation_lengths_m_" * "$mass" * "_delta_g_" * "$delta_g" * "_v_" * "$v" Energies Corr_lengths Bond_dims trunc_list
end