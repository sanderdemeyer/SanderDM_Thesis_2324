import numpy as np
import matplotlib.pyplot as plt

def gaussian_wave_packet(k, sigma):
    return np.exp(-(k/(2*sigma))**2)

sigma = 0.0025
# sigma = 0.1

X = np.linspace(-np.pi,np.pi,1000)
Y = [gaussian_wave_packet(x, sigma) for x in X]

plt.plot(X, Y)
plt.show()

def S(x):
    return 1/(1+np.exp(x))

def t(j, j_b, kappa):
    return 1 - 1/(1+np.exp(2*kappa*(j-j_b)))
    return 1 - S(2*kappa*(j-j_b))

j_b = 25
kappa = 0.5

X = range(50)
Y = [t(j, j_b, kappa) for j in X]

plt.plot(X, [np.log10(y) for y in (Y)])
plt.title(fr'For $j_b$ = {j_b} and $\kappa$ = {kappa}')
plt.show()