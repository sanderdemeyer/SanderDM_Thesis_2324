import numpy as np
import csv
import matplotlib.pyplot as plt

L = 180
κ = 1.0
truncation = 2.5
D = 14

frequency_of_saving = 3
RAMPING_TIME = 5

dt = 0.05
number_of_timesteps = 2500 #3000 #7000
am_tilde_0 = 0.06
Delta_g = -0.15 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie

v_max = 5.0

name_free = f"Code/bh_time_evolution/bhole_time_evolution_variables_N_{L}_mass_{am_tilde_0}_delta_g_{0.0}_ramping_{RAMPING_TIME}_dt_{dt}_nrsteps_{number_of_timesteps}_vmax_{v_max}_kappa_{κ}_trunc_{truncation}_D_{D}_savefrequency_{frequency_of_saving}.jld2_postprocessing_entropy"
name_int = f"Code/bh_time_evolution/bhole_time_evolution_variables_N_{L}_mass_{am_tilde_0}_delta_g_{Delta_g}_ramping_{RAMPING_TIME}_dt_{dt}_nrsteps_{number_of_timesteps}_vmax_{v_max}_kappa_{κ}_trunc_{truncation}_D_{D}_savefrequency_{frequency_of_saving}.jld2_postprocessing_entropy"

entropies = [[], []]
times = [[], []]
with open(name_free, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    for row in spamreader:
        print(row)
        times[0].append(float(row[0]))
        entropies[0].append(float(row[1]))

with open(name_int, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    for row in spamreader:
        print(row)
        times[1].append(float(row[0]))
        entropies[1].append(float(row[1]))

plt.plot(times[0], entropies[0], label = r'$\Delta(g)=0.0$')
plt.plot(times[1], entropies[1], label = r'$\Delta(g)=-0.15$')
plt.xlabel("Time",fontsize = 15)
plt.ylabel(r"Entropy $S_{\left[j_b-10,j_b\right]}$", fontsize=15)
plt.legend()
plt.show()