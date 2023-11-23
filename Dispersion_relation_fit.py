import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import r2_score
import csv
from scipy.optimize import curve_fit

closer = 13
rounding = 3

def fit_linear(x, a, b):
    return a*x+b

def fit_parabolic(x, a, b, c):
    return a*x**2+b*x+c

def function_OLD(x, m_fit, v_fit):
    return (v_fit/2)*np.sin(2*x) + np.sqrt(m_fit**2 + np.sin(x)**2)

def function_sqrt(x, m_fit, v_fit, c_fit):
    return v_fit*x + np.sqrt((m_fit*c_fit**2)**2 + (x*c_fit)**2)

def function(x, m_fit, v_fit, c_fit):
    return v_fit*x + m_fit*c_fit**2 + x**2/(m_fit*c_fit)/2 - x**4/(8*(m_fit*c_fit)**3)
    return v_fit*x + m_fit*c_fit**2*(1 + (x/mc)**2/2 - (x/mc)**4/8)#sqrt(1 + (x/(m*c))**2)
    return v_fit*x + m_fit*c_fit**2*sqrt(1 + (x/(m*c))**2)
    return v_fit*x + np.sqrt((m_fit*c_fit**2)**2 + (x*c_fit)**2)


def fit_function(delta_g, v, mass, plot = False):

    #file = "Dispersion_Delta_m_0.5_delta_g_-0.3_v_0.5"
    file = f"Dispersion_m_{mass} _delta_g_{delta_g} _v_{v}"

    f = h5py.File(file, "r")
    energies = f["energies"][:]
    bounds = f["bounds"]
    amount_data = np.shape(energies)[1]

    bounds = np.pi/12

    energies = [np.real(e[0]) for e in energies[0,:]]


    k = np.linspace(-bounds, bounds,amount_data)
    k_refined = np.linspace(-bounds, bounds, 1000)

    popt, pcov = curve_fit(function, k[closer:-closer], energies[closer:-closer], (mass, v, 1))
    (m_fit, v_fit, c_fit) = popt
    exp = [function(i, m_fit, v_fit, c_fit) for i in k_refined]

    print(f"For Delta_g = {delta_g}")
    print(f'mass gets regularized from {mass} to {m_fit}')
    print(f'v gets regularized from {v} to {v_fit}')
    print(f'c gets regularized from {1} to {c_fit}')

    if plot:
        plt.figure()
        plt.scatter(k, np.array(energies), label = 'quasiparticle')
        plt.plot(k_refined, exp, label = 'fit')
        plt.xlabel('k')
        plt.ylabel('energy')
        plt.legend()
        plt.title(fr"$m = {round(mass,rounding)} \to {round(m_fit,rounding)}$, $c = 1 \to {round(c_fit,rounding)}$ and $v = {round(v,rounding)} \to {round(v_fit,rounding)}$ for $\Delta(g) = {round(delta_g,rounding)}$")
        plt.show()
        string = fr'Fit_m = {mass}_v = {v}_\Delta(g) = {delta_g}.png'
        print(string)
        plt.savefig(fr'Fit_m = {mass}_v = {v}_Delta(g) = {delta_g}.png')

    return (m_fit, v_fit, c_fit)

mass_renorm = []
v_renorm = []
c_renorm = []
for delta_g in [-0.15*i for i in range(1, 5)]:
    local_mass_renorm = []
    local_v_renorm = []
    local_c_renorm = []
    for mass in [0.1*i for i in range(7)]:
        for v in [0.15]:
            (m_fit, v_fit, c_fit) = fit_function(delta_g, v, mass, plot = False)
            local_mass_renorm.append(m_fit)
            local_v_renorm.append(v_fit)
            local_c_renorm.append(c_fit)
    mass_renorm.append(local_mass_renorm)
    v_renorm.append(local_v_renorm)
    c_renorm.append(local_c_renorm)


mass_renorm = np.array(mass_renorm)
v_renorm = np.array(v_renorm)
c_renorm = np.array(c_renorm)
print(mass_renorm)
print(v_renorm)

for m_index in range(1, 7):
    plt.scatter([-0.15*i for i in range(1, 5)], v_renorm[:,m_index]/(-0.15), label = f'mass = {round(m_index*0.1,2)}')
plt.xlabel(r'$\Delta (g)$', fontsize = 15)
plt.ylabel(r'$v_{eff}$', fontsize = 15)
plt.title(r'Relative renormalization from $v = 0.15$', fontsize = 15)
plt.legend()
plt.show()

for m_index in range(1, 7):
    X = [-0.15*i for i in range(1, 5)]
    Y = (mass_renorm[:,m_index])/(m_index*0.1)
    plt.scatter(X, Y, label = f'mass = {round(m_index*0.1,2)}')

    popt, pcov = curve_fit(fit_parabolic, X, Y)
    (a, b, c) = popt
    X_refined = [X[0]+(X[-1]-X[0])*i/1000 for i in range(1000)]
    X_refined = [(X[-1])*i/1000 for i in range(1000)]
    linear_fit_analytical = [fit_parabolic(i, a, b, c) for i in X_refined]
    plt.plot(X_refined, linear_fit_analytical)

plt.scatter([0], [1], s = 25, c = "black")
plt.xlabel(r'$\Delta (g)$', fontsize = 15)
plt.ylabel(r'$mass_{eff}$', fontsize = 15)
plt.title(r'Relative renormalization the mass', fontsize = 15)
plt.legend()
plt.show()

for m_index in range(1, 7):
    plt.scatter([-0.15*i for i in range(1, 5)], c_renorm[:,m_index], label = f'mass = {round(m_index*0.1,2)}')
plt.xlabel(r'$\Delta (g)$', fontsize = 15)
plt.ylabel(r'$c_{eff}$', fontsize = 15)
plt.title(r'Renormalization from $c = 1$', fontsize = 15)
plt.legend()
plt.show()


"""
plt.scatter([-0.15*i for i in range(1, 5)], mass_renorm)
plt.xlabel(r'$\Delta (g)$', fontsize = 15)
plt.ylabel(r'$m_{eff}$', fontsize = 15)
plt.title(r'Renormalization from $m = 0.7$', fontsize = 15)
plt.show()
"""

# v does not seem to have an influence on the renormalization of m => m = m(g). Test m(g, v_sweep) is incoming
# v = v(g, m) or v(g)? g does have an influence