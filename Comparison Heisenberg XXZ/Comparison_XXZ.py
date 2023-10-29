import h5py
import numpy as np
import matplotlib.pyplot as plt

file_julia_D_30 = "Comparison Heisenberg XXZ/Comparison_XXZ_D_30"
file_julia_D_50 = "Comparison Heisenberg XXZ/Comparison_XXZ_D_50"
file_data = "Comparison Heisenberg XXZ/Data Heisenberg XXZ/Data_Heisenberg_XXZ.txt"
f_D_30 = h5py.File(file_julia_D_30, "r")
f_D_50 = h5py.File(file_julia_D_50, "r")
energies_D_30 = f_D_30["Energies"][:]
energies_D_50 = f_D_50["Energies"][:]


amount = 28
delta_gs = np.linspace(-1, 6, amount)

energies_XXZ_D_30 = [energies_D_30[i] - e/4 for (i,e) in enumerate(delta_gs)]
energies_XXZ_D_50 = [energies_D_50[i] - e/4 for (i,e) in enumerate(delta_gs)]

data_literature = np.loadtxt(file_data, delimiter=",", unpack=False)


print(delta_gs)
print(data_literature[:,0])
print(data_literature[:,1])



plt.scatter(delta_gs, energies_XXZ_D_30, label = 'own data, D = 30')
plt.scatter(delta_gs, energies_XXZ_D_50, label = 'own data, D = 50')
plt.plot(data_literature[:,0], data_literature[:,1], label = 'literature')
plt.legend()
plt.xlabel(r'$\Delta(g)$', fontsize = 15)
plt.ylabel(r'$E_{gs}$', fontsize = 15)
plt.show()


plt.scatter(delta_gs, [(energies_XXZ_D_30[i]-energies_XXZ_D_50[i])/energies_XXZ_D_30[i] for i in range(amount)])
plt.show()