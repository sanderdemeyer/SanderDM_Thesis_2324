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

L = 100
κ = 1.0
truncation = 3.5
D = 15

N = L//2 - 1

frequency_of_saving = 3
RAMPING_TIME = 5

dt = 0.2
number_of_timesteps = 0 #3000 #7000
am_tilde_0 = 2.0
Delta_g = 0.0 # voor kleinere g, betere fit op dispertierelatie. Op kleinere regio fitten. Probeer voor delta_g = 0 te kijken of ik exact v of -v kan fitten in de dispertierelatie

v_max = 2.0

X = [(2*np.pi)/N*i - np.pi for i in range(N)]

test = True
if test:
    name = f"bh_time_evolution/bhole_test_time_evolution_variables_N_{L}_mass_{am_tilde_0}_delta_g_{Delta_g}_ramping_{RAMPING_TIME}_dt_{dt}_nrsteps_{number_of_timesteps}_vmax_{v_max}_kappa_{κ}_trunc_{truncation}_D_{D}_savefrequency_{frequency_of_saving}.jld2"
else:
    name = f"bh_time_evolution/bhole_time_evolution_variables_N_{L}_mass_{am_tilde_0}_delta_g_{Delta_g}_ramping_{RAMPING_TIME}_dt_{dt}_nrsteps_{number_of_timesteps}_vmax_{v_max}_kappa_{κ}_trunc_{truncation}_D_{D}_savefrequency_{frequency_of_saving}.jld2"


X = []
Es = []
occs = []
expect = []
entropies = []

with open(name + "_pp_X_E_kvo2", newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    for row in spamreader:
        print(row)
        X.append(float(row[0]))
        Es.append(float(row[1]))
        expect.append(float(row[2]))
with open(name + "_pp_entropies", newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    for row in spamreader:
        print(row)
        entropies.append(float(row[0]))
with open(name + "_pp_occs", newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    for row in spamreader:
        print(row)
        occs.append([float(e) for e in row])
    
print(X)
print(Es)

occs_symmetrized = occs

print(len(Es))
print(len(occs_symmetrized))
print(len(occs_symmetrized[0]))

expect = fermi_dirac_convoluted(X, am_tilde_0/10, κ, 0.2)
excluded = 3
# plt.scatter(Es, occs, label = 'data', s = 8)
plt.scatter([-X[i] for i in range(excluded,len(Es)-excluded)], [occs_symmetrized[0][i] for i in range(excluded,len(Es)-excluded)], label = 'data - t = 0', s = 8)
plt.scatter([-X[i] for i in range(excluded,len(Es)-excluded)], [occs_symmetrized[-1][i] for i in range(excluded,len(Es)-excluded)], label = 'data - t = 42', s = 8)
plt.scatter([-X[i] for i in range(excluded,len(Es)-excluded)], [expect[i] for i in range(excluded,len(Es)-excluded)], label = 'Fermi-dirac', s = 8)
# plt.scatter([Es[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], [expect_smaller[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], label = 'expected Fermi-dirac / c', s = 8)
# plt.scatter([Es[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], [expect_zeker_bigger[i] for i in range(excluded,len(Es)-excluded) if (i != L//4)], label = 'expected Fermi-dirac - zeker bigger', s = 8)
plt.xlabel("momentum", fontsize = 15)
plt.ylabel(r"Occupation number $\hat{N}$", fontsize = 15)
# plt.vlines([am_tilde_0, -am_tilde_0], 0, 1, colors = "black", linestyles='--', label = r"$E = \pm m$")
plt.legend()
plt.show()

times = np.linspace(0, 42, 70)
plt.scatter(times, entropies, s = 8)
plt.xlabel("Time", fontsize = 15)
plt.ylabel(r"Entropy $S_{\left[j_b-10,j_b\right]}$", fontsize = 15)
plt.show()