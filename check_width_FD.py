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


for factor in [0.0001, 0.25, 0.5, 0.75, 1, 1.25]:
    plt.scatter(X, [fermi_dirac(k, factor*am_tilde_0, κ) for k in X], label = f"{factor}")
plt.legend()
plt.show()