import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import r2_score
import csv
from scipy.optimize import curve_fit


def f_taylor(k, m, v, c):
    return m*c**2 + v*k + k**2/(2*m) - (2*v*k**3)/3 - (4*m**2*c**2+3)/(24*m**3*c**2)*k**4


def f_exact(k, m, v, c):
    return v*np.sin(2*k)/2 + np.sqrt(m**2*c**4 + np.sin(k)**2*c**2)

bounds = np.pi/2
k_values = np.linspace(-bounds, bounds, 1000)

m = 0.5
v = 0.3
c = 3

plt.plot(k_values/np.pi, [f_taylor(e, m, v, c) for e in k_values])
plt.plot(k_values/np.pi, [f_exact(e, m, v, c) for e in k_values])
plt.show()