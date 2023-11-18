import numpy as np
import matplotlib.pyplot as plt


g_s = np.linspace(-2, 2, 1000)
plt.plot(g_s, [np.cos((np.pi-g)/2) for g in g_s])
plt.show()

m = 0

M = np.array([[0, 0, 0, 0], [0, m, 1, 0], [0, 1, -m, 0], [0, 0, 0, 0]])

a = -0.31830870376832465
print(a**2)
print(-np.sqrt(-a))
print(np.arccos(a)/np.pi)
print(np.arcsin(a)/np.pi)
print(np.arctan(a)/np.pi)