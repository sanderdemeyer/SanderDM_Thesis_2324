import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def gauss(k, k0, sigma):
    return (1/sigma*np.sqrt(2*np.pi)) * np.exp(-(k-k0)**2/(2*sigma**2)) / (2*np.pi)

def fermi_dirac(k, mass, kappa):
    E = np.sqrt(mass**2 + np.sin(k/2)**2)*(1-2*(k<0.0))
    return 1/(1+np.exp(-2*np.pi*E/kappa))

def fermi_dirac_convoluted(X, mass, kappa, sigma):
    Es = []
    for k in X:
        term1 = integrate.quad(lambda x: fermi_dirac(x, mass, kappa)*gauss(k, x, sigma), -np.pi, k)[0]
        term2 = integrate.quad(lambda x: fermi_dirac(x, mass, kappa)*gauss(k, x, sigma), k, np.pi)[0]
        Es.append(term1+term2)
    return Es

L = 180
κ = 0.8
truncation = 2.5
D = 14

N = L//2 - 1

frequency_of_saving = 3
RAMPING_TIME = 5

dt = 0.05
number_of_timesteps = 2500 #3000 #7000
am_tilde_0 = 0.06
Delta_g = -0.15 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie

v_max = 2.0

X = [(2*np.pi)/N*i - np.pi for i in range(N)]

if Delta_g == 0.0:
    name = f"Code/bh_time_evolution/bhole_time_evolution_variables_N_{L}_mass_{am_tilde_0}_delta_g_{Delta_g}_ramping_{RAMPING_TIME}_dt_{dt}_nrsteps_{number_of_timesteps}_vmax_{v_max}_kappa_{κ}_trunc_{truncation}_D_{D}_savefrequency_{frequency_of_saving}.jld2_pp_E_min_kvo2"
    Es = []
    occs = []
    expect = []

    with open(name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in spamreader:
            print(row)
            Es.append(float(row[0]))
            occs.append(float(row[1]))
            expect.append(float(row[2]))

else:
    name = f"Code/bh_time_evolution/bhole_time_evolution_variables_N_{L}_mass_{am_tilde_0}_delta_g_{Delta_g}_ramping_{RAMPING_TIME}_dt_{dt}_nrsteps_{number_of_timesteps}_vmax_{v_max}_kappa_{κ}_trunc_{truncation}_D_{D}_savefrequency_{frequency_of_saving}.jld2_pp_min"
    Es = []
    occs = []
    with open(name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        for row in spamreader:
            print(row)
            occs.append(float(row[0]))
    v = 0.0
    name_dispersion = f"Code/SanderDM_Thesis_2324/Dispersion_Data/Full range/Energies_Dispersion_Delta_L_{L}_m_{am_tilde_0}_delta_g_{Delta_g}_v_{v}_trunc_{truncation}.csv"

    with open(name_dispersion, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            print(row)
            Es.append(float(row[0])+Delta_g)    
    for i in range(len(Es)):
        print(f"i = {i}, E = {Es[i]}")
    Es = [(1-2*(X[i] > 0.0))*Es[i] for i in range(len(Es))]
occs_symmetrized = [occs[i] if (e < 0) else (1-occs[-i]) for (i,e) in enumerate(Es)]

print(X)
print(Es)

renorm_c = 0.839257
renorm_v = 0.995073
renorm_effect = renorm_v/renorm_c

expect = fermi_dirac_convoluted(X, am_tilde_0, κ, 0.178)
expect_zeker_bigger = fermi_dirac_convoluted(X, am_tilde_0, κ*5, 0.178)
expect_renorm = fermi_dirac_convoluted(X, am_tilde_0, κ*renorm_effect, 0.178)

excluded = 3
# plt.scatter(Es, occs, label = 'data', s = 8)
plt.scatter([Es[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], [occs_symmetrized[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], label = 'data', s = 8)
plt.scatter([Es[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], [expect[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], label = 'Fermi-dirac', s = 8)
plt.scatter([Es[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], [expect_renorm[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], label = 'Fermi-dirac - renormalized', s = 8)
# plt.scatter([Es[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], [expect_smaller[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], label = 'expected Fermi-dirac / c', s = 8)
# plt.scatter([Es[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], [expect_zeker_bigger[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], label = 'expected Fermi-dirac - zeker bigger', s = 8)
plt.xlabel("Energy", fontsize = 15)
plt.ylabel(r"Occupation number $\hat{N}$", fontsize = 15)
plt.vlines([am_tilde_0, -am_tilde_0], 0, 1, colors = "black", linestyles='--', label = r"$E = \pm m$")
plt.legend()
plt.show()

plt.scatter([Es[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], [np.log(1/occs_symmetrized[i] - 1) for i in range(excluded,len(Es)-excluded) if (i != L//4)], label = 'data', s = 8)
# plt.scatter(Es, [np.log(1/e - 1) for e in occs_symmetrized], label = 'data - symmetrized', s = 8)
plt.scatter([Es[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], [np.log(1/expect[i] - 1) for i in range(excluded,len(Es)-excluded) if (i != L//4)], label = 'Fermi-dirac', s = 8)
plt.scatter([Es[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], [np.log(1/expect_renorm[i] - 1) for i in range(excluded,len(Es)-excluded) if (i != L//4)], label = 'Fermi-dirac - renormalized', s = 8)
# plt.scatter([Es[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], [np.log(1/expect_smaller[i] - 1) for i in range(excluded,len(Es)-excluded) if (i != L//4)], label = 'expected Fermi-dirac / c', s = 8)
plt.legend()
plt.xlabel("Energy", fontsize = 15)
plt.ylabel(r"$\log{\left(\frac{1}{\hat{N}} - 1\right)}$", fontsize = 15)
plt.show()