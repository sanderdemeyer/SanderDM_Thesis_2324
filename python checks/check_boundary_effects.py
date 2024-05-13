import h5py
import numpy as np
import matplotlib.pyplot as plt

file = "Window_boundary_effects_are_small" 

with h5py.File(file, 'r') as f:
    # Read data
    Es = f[f"avg_energies"][:]
    Szs = f[f"avg_Sz"][:]
    print(Es)
    print(Szs)
    # Es.append(avged([e[0] for e in key1_data]))

N = len(Es)
plt.plot(range(N), Es, label = 'data')
plt.xlabel("Position", fontsize=15)
plt.ylabel("Energy", fontsize=15)
plt.axhline((Es[0]+Es[-1])/2, color = 'black', linestyle = '--', label = "ground state energy")
plt.legend(fontsize = 15) 
plt.show() 

plt.plot(range(N), Szs, label = 'data')
plt.xlabel("Position", fontsize=15)
plt.ylabel("Magnetization", fontsize=15)
plt.axhline(0, color = 'black', linestyle = '--', label = "ground state magnetization")
plt.legend() 
plt.show() 