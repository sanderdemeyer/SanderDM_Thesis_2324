import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

N = 10000
mass = 0.0

X = [(2*np.pi)/N*i - np.pi for i in range(N)]
Es = [np.sqrt(mass**2 + np.sin(k/2)**2)*(1-2*(k<0.0)) for k in X]


def gauss(k, k0, sigma):
    return (1/sigma*np.sqrt(2*np.pi)) * np.exp(-(k-k0)**2/(2*sigma**2)) / (2*np.pi)

def convolution(X, sigma):
    if sigma == 0.0:
        return [(k < 0.0)*1.0 for k in X]
    Es = []
    for k in X:
        term1 = integrate.quad(lambda x: (x < 0.0)*gauss(k, x, sigma), -np.pi, k)[0]
        term2 = integrate.quad(lambda x: (x < 0.0)*gauss(k, x, sigma), k, np.pi)[0]
        Es.append(term1+term2)
    return Es


delta_ks = [0.05*i for i in range(6)]

for deltak in delta_ks:
    conv = convolution(X, deltak)
    plt.plot(Es[N//3:-N//3], conv[N//3:-N//3], label = fr"$\Delta k = {round(deltak,2)}$")
plt.xlabel("Energy", fontsize = 15)
plt.ylabel(r"Occupation number $\hat{N}$", fontsize = 15)
# plt.vlines([mass, -mass], 0, 1, colors = "black", linestyles='--', label = r"$E = \pm m$")
plt.legend()
plt.show()
