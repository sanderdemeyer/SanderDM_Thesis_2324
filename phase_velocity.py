import numpy as np
import matplotlib.pyplot as plt

def theoretical_maximum(m, c):
    nodp = 1000
    k_values = np.linspace(-np.pi/2,np.pi/2,nodp)
    dk = np.pi/(nodp-1)
    E_values = [np.sqrt((m**2)*(c**4)+(np.sin(k)**2)*(c**2)) for k in k_values]
    v_values = [(E_values[i+1]-E_values[i])/dk for i in range(nodp-1)]
    return max(v_values)

m_values = np.linspace(0,1,100)

plt.plot(m_values, [theoretical_maximum(m,1) for m in m_values])
plt.xlabel('mass', fontsize = 15)
plt.ylabel(r'$v_{p}$', fontsize = 15)
plt.show()