import numpy as np
import matplotlib.pyplot as plt

def spatial_ramping_S(i, ib, iw, kappa):
    value = 1.0 - (1/(1+np.exp(2*kappa*(i-ib)))+1/(1+np.exp(2*kappa*(iw-i))))
    if value < 1e-4:
        return 0.0
    elif value > 1 - 1e-4:
        return 1.0
    return value

N = 160
kappa = 0.5*2.0
ib = 40
iw = 60

R = range(30,50)

lijst_ramping = [spatial_ramping_S(i, ib, iw, kappa) for i in R]

plt.scatter(R, lijst_ramping)
plt.show()