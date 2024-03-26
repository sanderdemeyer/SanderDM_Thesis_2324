"""
Defines the functions for the fits on the dispertion data.
the dispertion data for k = 0, namely mc^2, is set as a fixed parameter and not a parameter that needs to be fitted

function_fit_base is the fit function based on the taylor series of the exactly known free dispertion relation vk + sqrt((mc^2)^2 + (pc)^2)
function_4th_order_base is a general 4th order fit function

fit_function performs the fit based on function_fit_base
fit_function_factor does the same as the above function, but with a 'factor' denoting how much the data was 'zoomed in'
fit_function_factor_efficient does the same as the above, but more efficiently, which has an influence on the indexing.

Dispersion_relation_fit performs the fitting and iterates over all values of m,v, and delta(g).

"""


import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import r2_score
import csv
from scipy.optimize import curve_fit

rounding = 3
schmidt_cut = 3.5
excluded_0 = 1 # do you want to ignore m = 0.0?


def function_fit_base(k0, x, m_fit, v_fit):
    return k0 + v_fit*x + x**2/(2*m_fit) - (2*v_fit*x**3)/3 - (4*m_fit*k0+3)/(24*m_fit**2*k0)*x**4
    # return m_fit*c_fit**2 + v_fit*x + x**2/(2*m_fit) - (2*v_fit*x**3)/3 - (4*m_fit**2*c_fit**2+3)/(24*m_fit**3*c_fit**2)*x**4

def function_4th_order_base(k0, x, a1, a2, a3, a4):
    return k0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4



def fit_function(delta_g, v, mass, data_folder, closer, plot = False):

    #file = f"Dispersion_Data/{data_folder}/Dispersion_m_{mass} _delta_g_{delta_g} _v_{v}"
    #file = f"Dispersion_Data/{data_folder}/Dispersion_closer_m_{mass} _delta_g_{delta_g} _v_{v}"
    file = f"Dispersion_Data/{data_folder}/Dispersion_pi_over_72_m_{mass} _delta_g_{delta_g} _v_{v}"

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
        plt.plot(k_refined, exp_plus_sigma, 'b--', label = '4th order taylor fit')
        plt.plot(k_refined, exp_minus_sigma, 'b--', label = '4th order taylor fit')
        plt.plot(k_refined, exp_4th_order, 'r-', label = '4th order polynomial fit')
        plt.plot(k_refined, exp_4th_order_plus_sigma, 'r--', label = '4th order polynomial fit')
        plt.plot(k_refined, exp_4th_order_minus_sigma, 'r--', label = '4th order polynomial fit')
        plt.xlabel('k')
        plt.ylabel('energy')
        plt.legend()
        plt.title(fr"$m = {round(mass,rounding)} \to {round(m_fit,rounding)}$, $c = 1 \to {round(c_fit,rounding)}$ and $v = {round(v,rounding)} \to {round(v_fit,rounding)}$ for $\Delta(g) = {round(delta_g,rounding)}$")
        plt.show()
        string = fr'Fit_m = {mass}_v = {v}_\Delta(g) = {delta_g}.png'
        print(string)
        #plt.savefig(fr'Fit_m = {mass}_v = {v}_Delta(g) = {delta_g}.png')

    return (m_fit, v_fit, c_fit, m_fit_sigma, v_fit_sigma, c_fit_sigma)

def fit_function_factor(delta_g, v, mass, data_folder, factor, plot = False):
    ndp = 4 # Number of Data Points aan elke zijde rond k = 0, het totaal aantal datapunten waarmee gefit wordt, is dus 1+2*ndp

    #file = f"Dispersion_Data/{data_folder}/Dispersion_m_{mass} _delta_g_{delta_g} _v_{v}"
    #file = f"Dispersion_Data/{data_folder}/Dispersion_closer_m_{mass} _delta_g_{delta_g} _v_{v}"
    file = f"Dispersion_Data/{data_folder}/Dispersion_pi_over_72_m_{mass} _delta_g_{delta_g} _v_{v}"

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

    interesting_indices = [index_k_0 + factor*(i-ndp) for i in range(1+2*ndp)]
    k_points = [k[index] for index in interesting_indices]
    energy_points = [energies[index] for index in interesting_indices]

    popt, pcov = curve_fit(function_fit, k_points, energy_points, (mass, v))


    (m_fit, v_fit) = popt
    c_fit = np.sqrt(energies[index_k_0]/m_fit)
    m_fit_sigma = pcov[0,0]
    v_fit_sigma = pcov[1,1]
    c_fit_sigma = c_fit/(2*m_fit)*m_fit_sigma

    exp = [function_fit(i, m_fit, v_fit) for i in k_refined]
    exp_plus_sigma = [function_fit(i, m_fit+m_fit_sigma, v_fit+v_fit_sigma) for i in k_refined]
    exp_minus_sigma = [function_fit(i, m_fit-m_fit_sigma, v_fit-v_fit_sigma) for i in k_refined]

    popt_4th_order, pcov4 = curve_fit(function_4th_order, k_points, energy_points)
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
        plt.plot(k_refined, exp_plus_sigma, 'b--', label = '4th order taylor fit')
        plt.plot(k_refined, exp_minus_sigma, 'b--', label = '4th order taylor fit')
        plt.plot(k_refined, exp_4th_order, 'r-', label = '4th order polynomial fit')
        plt.plot(k_refined, exp_4th_order_plus_sigma, 'r--', label = '4th order polynomial fit')
        plt.plot(k_refined, exp_4th_order_minus_sigma, 'r--', label = '4th order polynomial fit')
        plt.xlabel('k')
        plt.ylabel('energy')
        plt.legend()
        plt.title(fr"$m = {round(mass,rounding)} \to {round(m_fit,rounding)}$, $c = 1 \to {round(c_fit,rounding)}$ and $v = {round(v,rounding)} \to {round(v_fit,rounding)}$ for $\Delta(g) = {round(delta_g,rounding)}$")
        plt.show()
        string = fr'Fit_m = {mass}_v = {v}_\Delta(g) = {delta_g}.png'
        print(string)
        #plt.savefig(fr'Fit_m = {mass}_v = {v}_Delta(g) = {delta_g}.png')

    return (m_fit, v_fit, c_fit, m_fit_sigma, v_fit_sigma, c_fit_sigma)

def fit_function_factor_efficient(delta_g, v, mass, data_folder, factor, plot = False):
    ndp = 4 # Number of Data Points aan elke zijde rond k = 0, het totaal aantal datapunten waarmee gefit wordt, is dus 1+2*ndp

    mass = round(mass,2)
    delta_g = round(delta_g,2)
    smallest = np.pi/72/24/8

    if delta_g == -0.0:
        delta_g = 0.0

    factor_max = 48
    k_values = [0]
    indices_to_index = {}
    current_index = 1

    indices_to_index[0] = 0
    for factor_here in range(1,factor_max+1):
        for j in range(1,5):
            if j*factor_here*smallest not in k_values:
                k_values.append(j*factor_here*smallest)
                k_values.append(-j*factor_here*smallest)
                indices_to_index[j*factor_here] = current_index
                indices_to_index[-j*factor_here] = current_index+1
                current_index += 2

    #file = f"Dispersion_Data/{data_folder}/Dispersion_m_{mass} _delta_g_{delta_g} _v_{v}"
    #file = f"Dispersion_Data/{data_folder}/Dispersion_closer_m_{mass} _delta_g_{delta_g} _v_{v}"
    file = f"Dispersion_Data/{data_folder}/Dispersion_pi_over_72_v_0_m_{mass}_delta_g_{delta_g}_v_{v}_trunc_2.5"

    f = h5py.File(file, "r")
    energies = f["energies"][:]
    bounds = f["bounds"]
    amount_data = np.shape(energies)[1]
    index_k_0 = 0 #(amount_data-1)//2


    #bounds = np.pi/12
    bounds = np.pi/72

    energies = [np.real(e[0]) for e in energies[0,:]]

    bounds = (ndp+1)*smallest*factor*1.1
    # k = np.linspace(-bounds, bounds,amount_data)
    k_refined = np.linspace(-bounds, bounds, 1000)

    function_fit = lambda x, m, v : function_fit_base(energies[index_k_0], x, m, v)
    function_4th_order = lambda x, a1, a2, a3, a4 : function_4th_order_base(energies[index_k_0], x, a1, a2, a3, a4)

    k_points = [smallest*j*factor for j in range(-ndp, ndp+1)]
    #energy_points = [indices_to_index[j*factor] for j in range(-ndp, ndp+1)]
    energy_points = [energies[indices_to_index[j*factor]] for j in range(-ndp, ndp+1)]

    popt, pcov = curve_fit(function_fit, k_points, energy_points, (mass, v))

    (m_fit, v_fit) = popt
    c_fit = np.sqrt(energies[index_k_0]/m_fit)
    m_fit_sigma = pcov[0,0]
    v_fit_sigma = pcov[1,1]
    c_fit_sigma = c_fit/(2*m_fit)*m_fit_sigma

    exp = [function_fit(i, m_fit, v_fit) for i in k_refined]
    exp_plus_sigma = [function_fit(i, m_fit+m_fit_sigma, v_fit+v_fit_sigma) for i in k_refined]
    exp_minus_sigma = [function_fit(i, m_fit-m_fit_sigma, v_fit-v_fit_sigma) for i in k_refined]

    popt_4th_order, pcov4 = curve_fit(function_4th_order, k_points, energy_points)
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
        plt.scatter(k_points, energy_points, label = 'quasiparticle')
        # plt.scatter(k, np.array(energies), label = 'quasiparticle')
        plt.plot(k_refined, exp, 'b-', label = '4th order taylor fit')
        plt.plot(k_refined, exp_plus_sigma, 'b--', label = '4th order taylor fit')
        plt.plot(k_refined, exp_minus_sigma, 'b--', label = '4th order taylor fit')
        plt.plot(k_refined, exp_4th_order, 'r-', label = '4th order polynomial fit')
        plt.plot(k_refined, exp_4th_order_plus_sigma, 'r--', label = '4th order polynomial fit')
        plt.plot(k_refined, exp_4th_order_minus_sigma, 'r--', label = '4th order polynomial fit')
        plt.xlabel('k')
        plt.ylabel('energy')
        plt.legend()
        plt.title(fr"$m = {round(mass,rounding)} \to {round(m_fit,rounding)}$, $c = 1 \to {round(c_fit,rounding)}$ and $v = {round(v,rounding)} \to {round(v_fit,rounding)}$ for $\Delta(g) = {round(delta_g,rounding)}$")
        plt.show()
        string = fr'Fit_m = {mass}_v = {v}_\Delta(g) = {delta_g}.png'
        print(string)
        #plt.savefig(fr'Fit_m = {mass}_v = {v}_Delta(g) = {delta_g}.png')

    return (m_fit, v_fit, c_fit, m_fit_sigma, v_fit_sigma, c_fit_sigma)


def Dispersion_relation_fit(closer, way_of_fitting = "factor", plot = False):
    data_folder = "Data closer"
    data_folder = "Data pi_over_72"
    data_folder = "Data_pi_over_72_v_0"

    if way_of_fitting == "factor":
        function_to_fit = fit_function_factor
    elif way_of_fitting == "closer":
        function_to_fit = fit_function
    elif way_of_fitting == "factor_efficient":
        function_to_fit = fit_function_factor_efficient
    else:
        print("Insert valid way_of_fitting")
        assert 0 == 1

    # mass_renorm = [[], [], []]
    # v_renorm = [[], [], []]
    # c_renorm = [[], [], []]
    # mass_sigma_renorm = [[], [], []]
    # v_sigma_renorm = [[], [], []]
    # c_sigma_renorm = [[], [], []]

    delta_g_number = 5
    mass_number = 7
    v_number = 1
    schmidt_cut_number = 3
    mass_renorm = np.zeros((schmidt_cut_number, delta_g_number-1, mass_number))
    v_renorm = np.zeros((schmidt_cut_number, delta_g_number-1, mass_number))
    c_renorm = np.zeros((schmidt_cut_number, delta_g_number-1, mass_number))
    mass_sigma_renorm = np.zeros((schmidt_cut_number, delta_g_number-1, mass_number))
    v_sigma_renorm = np.zeros((schmidt_cut_number, delta_g_number-1, mass_number))
    c_sigma_renorm = np.zeros((schmidt_cut_number, delta_g_number-1, mass_number))

    for (schmidt_number, schmidt_cut) in enumerate([3.5, 4.0, 4.5]):
        #data_folder = f"Data_cut_{schmidt_cut}"
        for (delta_g_index,delta_g) in enumerate([-0.15*i for i in range(1, delta_g_number)]): # 4 should be 5 sometimes
            for (mass_index,mass) in enumerate([0.1*i for i in range(1, mass_number)]):
                v = 0.0
                (m_fit, v_fit, c_fit, m_fit_sigma, v_fit_sigma, c_fit_sigma) = function_to_fit(delta_g, v, mass, data_folder, closer, plot = plot)
                mass_renorm[schmidt_number,delta_g_index,mass_index] = m_fit
                v_renorm[schmidt_number,delta_g_index,mass_index] = v_fit
                c_renorm[schmidt_number,delta_g_index,mass_index] = c_fit
                mass_sigma_renorm[schmidt_number,delta_g_index,mass_index] = m_fit_sigma
                v_sigma_renorm[schmidt_number,delta_g_index,mass_index] = v_fit_sigma
                c_sigma_renorm[schmidt_number,delta_g_index,mass_index] = c_fit_sigma

    # for (schmidt_number, schmidt_cut) in enumerate([3.5, 4.0, 4.5]):
    #     #data_folder = f"Data_cut_{schmidt_cut}"
    #     for delta_g in [-0.15*i for i in range(1, delta_g_number)]: # 4 should be 5 sometimes
    #         local_mass_renorm = []
    #         local_v_renorm = []
    #         local_c_renorm = []
    #         local_mass_sigma_renorm = []
    #         local_v_sigma_renorm = []
    #         local_c_sigma_renorm = []
    #         for mass in [0.1*i for i in range(1, mass_number)]:
    #             for v in [0.15]:
    #                 (m_fit, v_fit, c_fit, m_fit_sigma, v_fit_sigma, c_fit_sigma) = function_to_fit(delta_g, v, mass, data_folder, closer, plot = False)
    #                 local_mass_renorm.append(m_fit)
    #                 local_v_renorm.append(v_fit)
    #                 local_c_renorm.append(c_fit)
    #                 local_mass_sigma_renorm.append(m_fit_sigma)
    #                 local_v_sigma_renorm.append(v_fit_sigma)
    #                 local_c_sigma_renorm.append(c_fit_sigma)
    #         mass_renorm[schmidt_number].append(local_mass_renorm)
    #         v_renorm[schmidt_number].append(local_v_renorm)
    #         c_renorm[schmidt_number].append(local_c_renorm)
    #         mass_sigma_renorm[schmidt_number].append(local_mass_sigma_renorm)
    #         v_sigma_renorm[schmidt_number].append(local_v_sigma_renorm)
    #         c_sigma_renorm[schmidt_number].append(local_c_sigma_renorm)

    # mass_renorm = np.array(mass_renorm)
    # v_renorm = np.array(v_renorm)
    # c_renorm = np.array(c_renorm)
    # mass_sigma_renorm = np.array(mass_sigma_renorm)
    # v_sigma_renorm = np.array(v_sigma_renorm)
    # c_sigma_renorm = np.array(c_sigma_renorm)

    return (mass_renorm, v_renorm, c_renorm, mass_sigma_renorm, v_sigma_renorm, c_sigma_renorm)