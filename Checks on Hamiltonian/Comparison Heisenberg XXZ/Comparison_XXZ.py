import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

ACCURACY = 1000
RANGES = 1000

def integrand(x, mu):
    return (mu*np.sin(mu))/(2*np.cosh(np.pi*x)*(np.cosh(2*mu*x) - np.cos(mu)))


def analytical(Delta):
    if Delta > 1:
        return -Delta/4
    elif Delta > -1:
        mu = np.arccos(-Delta)
        integral = scipy.integrate.quad(integrand, -RANGES, RANGES, args=mu)[0]
        return (np.cos(mu)/4) - np.sin(mu)/mu * integral
    else:
        lambd = np.arccosh(-Delta)
        summation = 0
        for n in range(1,ACCURACY):
            summation += 1/(1+np.exp(2*lambd*n))
        return np.cosh(lambd)/4 - (np.sinh(lambd)/lambd)*(lambd/2 + 2*lambd*summation)


file_julia_D_30 = "Comparison Heisenberg XXZ/Comparison_XXZ_D_30"
file_julia_D_50 = "Comparison Heisenberg XXZ/Comparison_XXZ_D_50"
file_julia_dyn_trunc_3 = "Comparison Heisenberg XXZ/Comparison_XXZ_dynamic_trunc_3"
file_julia_dyn_trunc_4 = "Comparison Heisenberg XXZ/Comparison_XXZ_dynamic_trunc_4"
file_data = "Comparison Heisenberg XXZ/Data Heisenberg XXZ/Data_Heisenberg_XXZ.txt"
f_D_30 = h5py.File(file_julia_D_30, "r")
f_D_50 = h5py.File(file_julia_D_50, "r")
f_dyn_3 = h5py.File(file_julia_dyn_trunc_3, "r")
f_dyn_4 = h5py.File(file_julia_dyn_trunc_4, "r")
energies_D_30 = f_D_30["Energies"][:]
energies_D_50 = f_D_50["Energies"][:]
energies_dyn_3 = f_dyn_3["Energies"][:]
energies_dyn_4 = f_dyn_4["Energies"][:]


amount = 28
delta_gs = np.linspace(-1, 6, amount)

energies_XXZ_D_30 = [energies_D_30[i] - e/4 for (i,e) in enumerate(delta_gs)]
energies_XXZ_D_50 = [energies_D_50[i] - e/4 for (i,e) in enumerate(delta_gs)]
energies_XXZ_dyn_3 = [energies_dyn_3[i] - e/4 for (i,e) in enumerate(delta_gs)]
energies_XXZ_dyn_4 = [energies_dyn_4[i] - e/4 for (i,e) in enumerate(delta_gs)]

data_literature = np.loadtxt(file_data, delimiter=",", unpack=False)


print(delta_gs)
print(data_literature[:,0])
print(data_literature[:,1])

delta_gs_refined = np.linspace(-1,6,amount*10)

# plt.scatter(delta_gs, energies_XXZ_D_30, label = 'own data, D = 30')
# plt.scatter(delta_gs, energies_XXZ_D_50, label = 'own data, D = 50')
# plt.scatter(delta_gs, energies_XXZ_dyn_3, label = 'own data, dynamic, trunc = 3')
plt.scatter(delta_gs, energies_XXZ_dyn_4, label = 'Own calculations with VUMPS')
plt.plot(delta_gs_refined, [analytical(-e) for e in delta_gs_refined], label = 'analytical solution', c = 'orange')
#plt.plot(data_literature[:,0], data_literature[:,1], label = 'literature')
plt.legend()
plt.xlabel(r'$\Delta(g)$', fontsize = 15)
plt.ylabel(r'$E_{gs}$', fontsize = 15)
plt.show()

ana = [analytical(-e) for e in delta_gs]

plt.scatter(delta_gs, [np.log(abs((energies_XXZ_D_30[i]-ana[i])/ana[i])) for i in range(amount)], label = 'own data, D = 30')
plt.scatter(delta_gs, [np.log(abs((energies_XXZ_D_50[i]-ana[i])/ana[i])) for i in range(amount)], label = 'own data, D = 50')
plt.scatter(delta_gs, [np.log(abs((energies_XXZ_dyn_3[i]-ana[i])/ana[i])) for i in range(amount)], label = 'own data, dynamic, trunc = 3')
plt.scatter(delta_gs, [np.log(abs((energies_XXZ_dyn_4[i]-ana[i])/ana[i])) for i in range(amount)], label = 'own data, dynamic, trunc = 4')
plt.xlabel(r'$\Delta(g)$', fontsize = 15)
plt.ylabel(r'logarithm of the error on $E_{gs}$', fontsize = 15)
plt.legend()
plt.show()

plt.scatter(delta_gs, [abs((energies_XXZ_D_30[i]-ana[i])/ana[i]) for i in range(amount)], label = 'own data, D = 30')
plt.scatter(delta_gs, [abs((energies_XXZ_D_50[i]-ana[i])/ana[i]) for i in range(amount)], label = 'own data, D = 50')
plt.scatter(delta_gs, [abs((energies_XXZ_dyn_3[i]-ana[i])/ana[i]) for i in range(amount)], label = 'own data, dynamic, trunc = 3')
plt.scatter(delta_gs, [abs((energies_XXZ_dyn_4[i]-ana[i])/ana[i]) for i in range(amount)], label = 'own data, dynamic, trunc = 4')
plt.xlabel(r'$\Delta(g)$', fontsize = 15)
plt.ylabel(r'error on $E_{gs}$', fontsize = 15)
plt.legend()
plt.show()