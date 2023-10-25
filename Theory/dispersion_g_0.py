import numpy as np
import matplotlib.pyplot as plt

k = np.linspace(-np.pi, np.pi, 1000)
m = 1
Zg_a = 1

energy = -np.sqrt(m**2 + (Zg_a*np.sin(k/2))**2)

plt.plot(k, energy)
plt.show()