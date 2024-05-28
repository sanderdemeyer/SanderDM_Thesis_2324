import numpy as np
import matplotlib.pyplot as plt

def spatial_ramping_S_bw(i, ib, iw, kappa):
    value = 1.0 - (1/(1+np.exp(2*kappa*(i-ib))) + 1/(1+np.exp(2*kappa*(iw-i))))
    if value < 1e-4:
        return 0.0
    elif value > 1 - 1e-4:
        return 1.0
    return value

def spatial_ramping_S_b(i, ib, kappa):
    value = 1.0 - 1/(1+np.exp(2*kappa*(i-ib)))
    if value < 1e-4:
        return 0.0
    elif value > 1 - 1e-4:
        return 1.0
    return value

N = 100  # Number of sites

kappa = 0.5
ib = 30
iw = 70

dt = 1.0
number_of_timesteps = 20  # 3000 #7000
t_end = dt * number_of_timesteps
frequency_of_saving = 1
RAMPING_TIME = 5

am_tilde_0 = 0.03
Delta_g = 0.0  # For smaller g, better fit on dispersion relation. Try to fit exactly v or -v in the dispersion relation for Delta_g = 0.
v = 0.0
v_max = 1.0

truncation = 1.5

lijst_ramping_bw = [v_max*spatial_ramping_S_bw(i, ib, iw, kappa) for i in range(1, N+1)]
lijst_ramping_b = [v_max*spatial_ramping_S_b(i, 50, kappa) for i in range(1, N+1)]

plt.plot(range(1,N+1), lijst_ramping_bw)
plt.xlabel("Position", fontsize = 15)
plt.ylabel(r"Ramping function $S$", fontsize = 15)
plt.show()


plt.plot(range(1,N+1), lijst_ramping_b)
plt.xlabel("Position", fontsize = 15)
plt.ylabel(r"Ramping function $S$", fontsize = 15)
plt.show()


plt.plot(range(N-1), [lijst_ramping_b[i+1]-lijst_ramping_b[i] for i in range(N-1)])
plt.show()

