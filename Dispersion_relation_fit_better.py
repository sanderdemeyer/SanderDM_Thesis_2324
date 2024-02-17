import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import r2_score
import csv
from scipy.optimize import curve_fit


def function_fit_base(k0, x, m_fit, v_fit):
    return k0 + v_fit*x + x**2/(2*m_fit) - (2*v_fit*x**3)/3 - (4*m_fit*k0+3)/(24*m_fit**2*k0)*x**4
    # return m_fit*c_fit**2 + v_fit*x + x**2/(2*m_fit) - (2*v_fit*x**3)/3 - (4*m_fit**2*c_fit**2+3)/(24*m_fit**3*c_fit**2)*x**4

def function_4th_order_base(k0, x, a1, a2, a3, a4):
    return k0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4


def fit_function(delta_g, v, mass, plot = False):

    #file = f"Dispersion_Data/{data_folder}/Dispersion_m_{mass} _delta_g_{delta_g} _v_{v}"
    file = f"Dispersion_Data/{data_folder}/Dispersion_closer_m_{mass} _delta_g_{delta_g} _v_{v}"

    f = h5py.File(file, "r")
    energies = f["energies"][:]
    bounds = f["bounds"]
    amount_data = np.shape(energies)[1]
    index_k_0 = (amount_data-1)//2

    #bounds = np.pi/12
    bounds = np.pi/36

    energies = [np.real(e[0]) for e in energies[0,:]]


    k = np.linspace(-bounds, bounds,amount_data)
    k_refined = np.linspace(-bounds, bounds, 1000)

    function_fit = lambda x, m, v : function_fit_base(energies[index_k_0], x, m, v)
    function_4th_order = lambda x, a1, a2, a3, a4 : function_4th_order_base(energies[index_k_0], x, a1, a2, a3, a4)

    popt, pcov = curve_fit(function_fit, k[closer:-closer], energies[closer:-closer], (mass, v))


    (m_fit, v_fit) = popt
    c_fit = np.sqrt(energies[index_k_0]/m_fit)
    m_fit_sigma = pcov[0,0]
    v_fit_sigma = pcov[1,1]
    c_fit_sigma = c_fit/(2*m_fit)*m_fit_sigma

    exp = [function_fit(i, m_fit, v_fit) for i in k_refined]
    exp_plus_sigma = [function_fit(i, m_fit+m_fit_sigma, v_fit+v_fit_sigma) for i in k_refined]
    exp_minus_sigma = [function_fit(i, m_fit-m_fit_sigma, v_fit-v_fit_sigma) for i in k_refined]

    popt_4th_order, pcov4 = curve_fit(function_4th_order, k[closer:-closer], energies[closer:-closer])
    (a1, a2, a3, a4) = popt_4th_order
    exp_4th_order = [function_4th_order(i, a1, a2, a3, a4) for i in k_refined]
    exp_4th_order_plus_sigma = [function_4th_order(i, a1+pcov4[0,0], a2+pcov4[1,1], a3+pcov4[2,2], a4+pcov4[3,3]) for i in k_refined]
    exp_4th_order_minus_sigma = [function_4th_order(i, a1-pcov4[0,0], a2-pcov4[1,1], a3-pcov4[2,2], a4-pcov4[3,3]) for i in k_refined]

    """
    check1 = (-3*a3)/(2*a1)
    check2 = -(a2*(2*a0+3*a2))/(6*a0*a4)
    print(f'for this value, the checks that should be 1 are {(-3*a3)/(2*a1)} and {-(a2*(2*a0+3*a2))/(6*a0*a4)}.')
    print(f'for this value, the low order parameters are {a0}, {a1}, and {a2}.')
    print(f'for this value, the high order parameters are {a3} and {a4}.')
    """

    if plot:
        plt.figure()
        plt.scatter(k, np.array(energies), label = 'quasiparticle')
        plt.plot(k_refined, exp, 'b-', label = '4th order taylor fit')
        # plt.plot(k_refined, exp_plus_sigma, 'b--', label = '4th order taylor fit')
        # plt.plot(k_refined, exp_minus_sigma, 'b--', label = '4th order taylor fit')
        # plt.plot(k_refined, exp_4th_order, 'r-', label = '4th order polynomial fit')
        # plt.plot(k_refined, exp_4th_order_plus_sigma, 'r--', label = '4th order polynomial fit')
        # plt.plot(k_refined, exp_4th_order_minus_sigma, 'r--', label = '4th order polynomial fit')
        plt.xlabel('k', fontsize = 15)
        plt.ylabel('energy', fontsize = 15)
        plt.legend()
        plt.title(fr"$m = {round(mass,rounding)} \to {round(m_fit,rounding)}$, $c = 1 \to {round(c_fit,rounding)}$ and $v = {round(v,rounding)} \to {round(v_fit,rounding)}$ for $\Delta(g) = {round(delta_g,rounding)}$")
        plt.show()
        string = fr'Fit_m = {mass}_v = {v}_\Delta(g) = {delta_g}.png'
        print(string)
        #plt.savefig(fr'Fit_m = {mass}_v = {v}_Delta(g) = {delta_g}.png')

    return (m_fit, v_fit, c_fit, m_fit_sigma, v_fit_sigma, c_fit_sigma)

closer = 20
rounding = 3
schmidt_cut = 3.5
excluded_0 = 1 # do you want to ignore m = 0.0?

mass_renorm = [[], [], []]
v_renorm = [[], [], []]
c_renorm = [[], [], []]
mass_sigma_renorm = [[], [], []]
v_sigma_renorm = [[], [], []]
c_sigma_renorm = [[], [], []]

for (schmidt_number, schmidt_cut) in enumerate([3.5, 4.0, 4.5]):
    #data_folder = f"Data_cut_{schmidt_cut}"
    data_folder = "Data closer"
    for delta_g in [-0.15*i for i in range(1, 5)]:
        local_mass_renorm = []
        local_v_renorm = []
        local_c_renorm = []
        local_mass_sigma_renorm = []
        local_v_sigma_renorm = []
        local_c_sigma_renorm = []
        for mass in [0.1*i for i in range(1, 7)]:
            for v in [0.15]:
                (m_fit, v_fit, c_fit, m_fit_sigma, v_fit_sigma, c_fit_sigma) = fit_function(delta_g, v, mass, plot = True)
                local_mass_renorm.append(m_fit)
                local_v_renorm.append(v_fit)
                local_c_renorm.append(c_fit)
                local_mass_sigma_renorm.append(m_fit_sigma)
                local_v_sigma_renorm.append(v_fit_sigma)
                local_c_sigma_renorm.append(c_fit_sigma)
        mass_renorm[schmidt_number].append(local_mass_renorm)
        v_renorm[schmidt_number].append(local_v_renorm)
        c_renorm[schmidt_number].append(local_c_renorm)
        mass_sigma_renorm[schmidt_number].append(local_mass_sigma_renorm)
        v_sigma_renorm[schmidt_number].append(local_v_sigma_renorm)
        c_sigma_renorm[schmidt_number].append(local_c_sigma_renorm)

"""
mass_renorm = np.array(mass_renorm[0])
v_renorm = np.array(v_renorm[0])
c_renorm = np.array(c_renorm[0])
checks1 = np.array(checks1[0])
checks2 = np.array(checks2[0])
"""
mass_renorm = np.array(mass_renorm)
v_renorm = np.array(v_renorm)
c_renorm = np.array(c_renorm)
mass_sigma_renorm = np.array(mass_sigma_renorm)
v_sigma_renorm = np.array(v_sigma_renorm)
c_sigma_renorm = np.array(c_sigma_renorm)

print(v_renorm)
print(v_sigma_renorm)
print(mass_renorm)
print(mass_sigma_renorm)
print(c_renorm)
print(c_sigma_renorm)



for (schmidt_number, schmidt_cut) in enumerate([3.5, 4.0, 4.5]):
    for m_index in range(1, 7):
        #plt.scatter([-0.15*i for i in range(1, 5)], v_renorm[schmidt_number,:,m_index-excluded_0]/(-0.15), label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}')
        plt.errorbar([-0.15*i for i in range(1, 5)], v_renorm[schmidt_number,:,m_index-excluded_0]/(-0.15), v_sigma_renorm[schmidt_number,:,m_index-excluded_0]/(0.15), label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}', fmt="o")
        #plt.scatter([-0.15*i for i in range(1, 5)], v_renorm[:,m_index-excluded_0]/(-0.15), label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}')
    plt.xlabel(r'$\Delta (g)$', fontsize = 15)
    plt.ylabel(r'$v_{eff}$', fontsize = 15)
    plt.title(fr'Relative renormalization from $v = 0.15$ for schmidt-cut = {schmidt_cut}', fontsize = 15)
    plt.legend()
plt.show()

for (schmidt_number, schmidt_cut) in enumerate([3.5, 4.0, 4.5]):
    for m_index in range(1, 7):
        plt.errorbar([-0.15*i for i in range(1, 5)], mass_renorm[schmidt_number,:,m_index-excluded_0]/(m_index*0.1), mass_sigma_renorm[schmidt_number,:,m_index-excluded_0]/(m_index*0.1), label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}', fmt="o")
        #plt.scatter([-0.15*i for i in range(1, 5)], v_renorm[:,m_index-excluded_0]/(-0.15), label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}')
    plt.xlabel(r'$\Delta (g)$', fontsize = 15)
    plt.ylabel(r'$mass_{eff}$', fontsize = 15)
    plt.title(fr'Relative renormalization the mass for schmidt-cut = {schmidt_cut}', fontsize = 15)
    plt.legend()
plt.show()

for (schmidt_number, schmidt_cut) in enumerate([3.5, 4.0, 4.5]):
    for m_index in range(1, 7):
        plt.errorbar([-0.15*i for i in range(1, 5)], c_renorm[schmidt_number,:,m_index-excluded_0], c_sigma_renorm[schmidt_number,:,m_index-excluded_0], label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}', fmt="o")
        #plt.scatter([-0.15*i for i in range(1, 5)], v_renorm[:,m_index-excluded_0]/(-0.15), label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}')
    plt.xlabel(r'$\Delta (g)$', fontsize = 15)
    plt.ylabel(r'$c_{eff}$', fontsize = 15)
    plt.title(fr'Relative renormalization of c for schmidt-cut = {schmidt_cut}', fontsize = 15)
    plt.legend()
plt.show()

print(intentionele_stop)


for (schmidt_number, schmidt_cut) in enumerate([3.5, 4.0, 4.5]):
    for m_index in range(1, 7):
        plt.scatter([-0.15*i for i in range(1, 5)], checks1[schmidt_number,:,m_index-excluded_0], label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}')
        #plt.scatter([-0.15*i for i in range(1, 5)], checks1[:,m_index-excluded_0], label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}')
    plt.xlabel(r'$\Delta (g)$', fontsize = 15)
    plt.ylabel(r'$check1$', fontsize = 15)
    plt.title(fr'check1 for schmidt-cut = {schmidt_cut}', fontsize = 15)
    plt.legend()
plt.show()

for (schmidt_number, schmidt_cut) in enumerate([3.5, 4.0, 4.5]):
    for m_index in range(1, 7):
        plt.scatter([-0.15*i for i in range(1, 5)], checks2[schmidt_number,:,m_index-excluded_0], label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}')
        #plt.scatter([-0.15*i for i in range(1, 5)], checks2[:,m_index-excluded_0], label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}')
    plt.xlabel(r'$\Delta (g)$', fontsize = 15)
    plt.ylabel(r'$check2$', fontsize = 15)
    plt.title(fr'check2 for schmidt-cut = {schmidt_cut}', fontsize = 15)
    plt.legend()
plt.show()


for m_index in range(3, 7):
    X = [-0.15*i for i in range(1, 5)]
    Y = (mass_renorm[:,m_index-excluded_0])/(m_index*0.1)
    plt.scatter(X, Y, label = f'mass = {round(m_index*0.1,2)}')

    popt, pcov = curve_fit(fit_parabolic, X, Y)
    (a, b, c) = popt
    X_refined = [X[0]+(X[-1]-X[0])*i/1000 for i in range(1000)]
    X_refined = [(X[-1])*i/1000 for i in range(1000)]
    linear_fit_analytical = [fit_parabolic(i, a, b, c) for i in X_refined]
    #plt.plot(X_refined, linear_fit_analytical)

#plt.scatter([0], [1], s = 25, c = "black")
plt.xlabel(r'$\Delta (g)$', fontsize = 15)
plt.ylabel(r'$mass_{eff}$', fontsize = 15)
plt.title(fr'Relative renormalization the mass for schmidt-cut = {schmidt_cut}', fontsize = 15)
plt.legend()
plt.show()

for m_index in range(1, 7):
    plt.scatter([-0.15*i for i in range(1, 5)], c_renorm[:,m_index-excluded_0], label = f'mass = {round(m_index*0.1,2)}')
plt.xlabel(r'$\Delta (g)$', fontsize = 15)
plt.ylabel(r'$c_{eff}$', fontsize = 15)
plt.title(fr'Renormalization from $c = 1$ for schmidt-cut = {schmidt_cut}', fontsize = 15)
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