import numpy as np
import matplotlib.pyplot as plt

def spatial_ramping_S(i, i_middle, kappa):
    return 1 - 1/(1+np.exp(2*kappa*(i-i_middle)))

L = 50
i_middle = L//2
kappa = 0.1

X = np.linspace(0, L, 1000)
X_small = np.linspace(i_middle-3, i_middle+3, 1000)
Y = [spatial_ramping_S(i, i_middle, kappa) for i in X]
plt.plot(X, Y)
plt.plot(X_small, [0.5+(kappa/2)*(i-i_middle) for i in X_small])
plt.show()
