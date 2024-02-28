import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def linear(x, a, b):
    return a*x+b

def fermi_dirac(x, kappa):
    return 1/(1+np.exp(2*np.pi*x/kappa))

# kappa = 0.1
# X = np.linspace(-0.15,0.15,1000)
# plt.plot(X, [fermi_dirac(i,kappa) for i in X])
# plt.show()

frequency_of_saving = 5
dt = 0.01

Y = np.array([122.5, 123, 139.3, 144.2, 158.8, 165.4, 177.4, 184.4, 195.5])*frequency_of_saving*dt
X = [47-(34-i) for i in range(len(Y))]
X = [34-i for i in range(len(Y))]

popt, pcov = opt.curve_fit(linear, X, Y)

X_more = np.linspace(min(X), max(X), 1000)
Y_more = popt[0]*X_more+popt[1]

print(f"slope = {popt[0]}")
print(f"starting point = {popt[1]}")

plt.xlabel('Position')
plt.ylabel('Time of first wave arrival')
#(i - N*2//3)*popt[0]
N = 70
plt.plot(X, [(N*2//3 - i)*(0.5) for i in X], label = r'expected for $c = 1$')
plt.plot(X,Y, label = 'data')
plt.plot(X_more, Y_more, label = 'fit')
plt.legend()
plt.show()





print(a)


def spatial_ramping_S(i, i_middle, k):
    return 1 - 1/(1+np.exp(2*k*(i-i_middle)))


N = 30
i_b = 15
kappa = 0.5

lijst_ramping = [spatial_ramping_S(i, i_b, kappa) for i in range(1,N)]

plt.plot(range(1,N), lijst_ramping)
plt.show()

print(a)

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