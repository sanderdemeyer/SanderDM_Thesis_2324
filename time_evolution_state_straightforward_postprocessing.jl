using LinearAlgebra
# using Base
using KrylovKit
using JLD2
using TensorKit
using MPSKitModels, TensorKit, MPSKit
using Statistics
using Plots

function avged(Es)
    return real.([(Es[2*i+1]+Es[2*i+2])/2 for i = 0:div(length(Es),2)-1])
end

truncation = 1.5
mass = 0.3
Delta_g = -0.15
v = 0.0

N = 50 # 40
dt = 0.2 # 0.8
t_end = 1.0 # 0.8
k = 1.0
sigma = 0.178

E = sqrt(mass^2 + sin(k/2)^2)
expected_speed = sin(k)/(4*E)

# file_name = "SanderDM_Thesis_2324/test_wavepacket_gs_mps_wo_symmetries_trunc_$(truncation)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)_N_$(N)_k_$(k)_sigma_$(sigma)_dt_$(dt)_tend_$(t_end)"
# file_name = "SanderDM_Thesis_2324/test_wavepacket_left_moving_gs_mps_wo_symmetries_trunc_$(truncation)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)_N_$(N)_k_$(k)_sigma_$(sigma)_dt_$(dt)_tend_$(t_end)"
file_name = "SanderDM_Thesis_2324/test_wavepacket_right_moving_gs_mps_wo_symmetries_trunc_$(truncation)_mass_$(mass)_v_$(v)_Delta_g_$(Delta_g)_N_$(N)_k_$(k)_sigma_$(sigma)_dt_$(dt)_tend_$(t_end)"

@load file_name Es

z1 = 7
z2 = 33
plt = plot(z1:z2, avged(Es[1])[z1:z2])
for i = 2:length(Es)
    plot!(z1:z2, avged(Es[i])[z1:z2])
end
display(plt)


peaks = []
for i = 1:length(Es)
    E_wp = avged(Es[i])[2:end-2]
    index_of_peak = findfirst(x->x==maximum(E_wp), E_wp)
    push!(peaks, index_of_peak)
end

times = [t_end*i for i = 1:length(Es)]

plt = scatter(times, peaks)
display(plt)

Es1 = Es[1]
Es2 = Es[2]
Es3 = Es[3]
Es4 = Es[4]
Es5 = Es[5]
Es6 = Es[6]
# Es7 = Es[7]
# Es8 = Es[8]
# Es9 = Es[9]
# Es10 = Es[10]
# Es11 = Es[11]
# Es12 = Es[12]
# Es13 = Es[13]
# Es14 = Es[14]
# Es15 = Es[15]
# Es16 = Es[16]
# Es17 = Es[17]

data = Dict(string(i) => Es[i] for i = 1:6)
@save file_name*"_dict.h5" Es1 Es2 Es3 Es4 Es5 Es6 # Es7 Es8 Es9 Es10 Es11 Es12 Es13 Es14 Es15 Es16 Es17

println("done")