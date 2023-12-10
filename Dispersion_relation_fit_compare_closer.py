import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import r2_score
import csv
from scipy.optimize import curve_fit

import Dispersion_relation_fit_function as drff


def fit_sqrt(x, extrap, a):
    return extrap + a*np.sqrt(x)

def fit_exp(x, extrap, b, c):
    return extrap + b*np.exp(c*x)

rounding = 3
schmidt_cut = 3.5
excluded_0 = 1 # do you want to ignore m = 0.0?

mass_renorms = []
v_renorms = []
c_renorms = []
mass_sigma_renorms = []
v_sigma_renorms = []
c_sigma_renorms = []

way_of_fitting = "factor_efficient"

if way_of_fitting == "closer":
    max_index = 20
elif way_of_fitting == "factor":
    max_index = 6
elif way_of_fitting == "factor_efficient":
    max_index = 48
else:
    print("Insert valid way_of_fitting")
    assert 0 == 1

for (ind, closer) in enumerate([1 + i for i in range(max_index)]):
    #(mass_renorm, v_renorm, c_renorm, mass_sigma_renorm, v_sigma_renorm, c_sigma_renorm) = drff.Dispersion_relation_fit(closer, way_of_fitting="closer")
    (mass_renorm, v_renorm, c_renorm, mass_sigma_renorm, v_sigma_renorm, c_sigma_renorm) = drff.Dispersion_relation_fit(closer, way_of_fitting=way_of_fitting)
    mass_renorms.append(mass_renorm)
    v_renorms.append(v_renorm)
    c_renorms.append(c_renorm)
    mass_sigma_renorms.append(mass_sigma_renorm)
    v_sigma_renorms.append(v_sigma_renorm)
    c_sigma_renorms.append(c_sigma_renorm)

mass_renorms = np.array(mass_renorms)
v_renorms = np.array(v_renorms)
c_renorms = np.array(c_renorms)
mass_sigma_renorms = np.array(mass_sigma_renorms)
v_sigma_renorms = np.array(v_sigma_renorms)
c_sigma_renorms = np.array(c_sigma_renorms)

function_to_fit = fit_exp

show = ["c"]
error_bars = False

delta_g_range = 4

if "v" in show:
    for m_index in range(1,7):
        for (i,delta_g) in enumerate([-0.15*i for i in range(1, delta_g_range)]):
            """
            popt, pcov = curve_fit(function_to_fit, [1 + i for i in range(max_index)], v_renorms[:,0,i,m_index-excluded_0], (m_index*0.1, 0, 1))
            factor_list = np.linspace(0, max_index, 1000)
            fitted_sqrt = [function_to_fit(j, popt[0], popt[1], popt[2]) for j in factor_list]
            plt.plot(factor_list, fitted_sqrt)   
            """
            if error_bars:
                plt.errorbar([1 + i for i in range(max_index)], v_renorms[:,0,i,m_index-excluded_0], v_sigma_renorms[:,0,i,m_index-excluded_0], label = fr'mass = {round(m_index*0.1,2)}, $\Delta(g)$ = {delta_g}', fmt="o")
            else:
                plt.scatter([1 + i for i in range(max_index)], v_renorms[:,0,i,m_index-excluded_0], label = fr'mass = {round(m_index*0.1,2)}, $\Delta(g)$ = {delta_g}')
            plt.xlabel(r'number of data points', fontsize = 15)
            plt.ylabel(r'$v_{eff}$', fontsize = 15)
            plt.title(fr'Relative renormalization from $v = 0.15$ for schmidt-cut = {schmidt_cut}', fontsize = 15)
            plt.legend()
            plt.show()

if "mass" in show:
    for m_index in range(1,7):
        for (i,delta_g) in enumerate([-0.15*i for i in range(1, delta_g_range)]):
            if error_bars:
                plt.errorbar([1 + i for i in range(max_index)], mass_renorms[:,0,i,m_index-excluded_0], mass_sigma_renorms[:,0,i,m_index-excluded_0], label = fr'mass = {round(m_index*0.1,2)}, $\Delta(g)$ = {delta_g}', fmt="o")
            else:
                plt.scatter([1 + i for i in range(max_index)], mass_renorms[:,0,i,m_index-excluded_0], label = fr'mass = {round(m_index*0.1,2)}, $\Delta(g)$ = {delta_g}')
            plt.xlabel(r'number of data points', fontsize = 15)
            plt.ylabel(r'$mass_{eff}$', fontsize = 15)
            plt.title(fr'Relative renormalization of the mass for schmidt-cut = {schmidt_cut}', fontsize = 15)
            plt.legend()
            plt.show()

if "c" in show:
    for m_index in range(1,7):
        for (i,delta_g) in enumerate([-0.15*i for i in range(1, delta_g_range)]):
            if error_bars:
                plt.errorbar([1 + i for i in range(max_index)], c_renorms[:,0,i,m_index-excluded_0], c_sigma_renorms[:,0,i,m_index-excluded_0], label = fr'mass = {round(m_index*0.1,2)}, $\Delta(g)$ = {delta_g}', fmt="o")
            else:
                plt.scatter([1 + i for i in range(max_index)], c_renorms[:,0,i,m_index-excluded_0], label = fr'mass = {round(m_index*0.1,2)}, $\Delta(g)$ = {delta_g}')
            plt.xlabel(r'number of data points', fontsize = 15)
            plt.ylabel(r'$c_{eff}$', fontsize = 15)
            plt.title(fr'Relative renormalization of c for schmidt-cut = {schmidt_cut}', fontsize = 15)
            plt.legend()
            plt.show()



for (ind, closer) in enumerate([1 + i for i in range(20)]):
    for m_index in range(1, 7):
        print(f'index is {ind}')
        #plt.scatter([-0.15*i for i in range(1, 5)], v_renorm[schmidt_number,:,m_index-excluded_0]/(-0.15), label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}')
        #plt.scatter([-0.15*i for i in range(1, 5)], v_renorms[ind,0,:,m_index-excluded_0]/(-0.15))
        plt.errorbar([-0.15*i for i in range(1, 5)], v_renorms[ind,0,:,m_index-excluded_0]/(-0.15), v_sigma_renorms[ind,0,:,m_index-excluded_0]/(0.15), label = f'mass = {round(m_index*0.1,2)}. data points = {51-2*closer}', fmt="o")
        #plt.scatter([-0.15*i for i in range(1, 5)], v_renorm[:,m_index-excluded_0]/(-0.15), label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}')
    plt.xlabel(r'$\Delta (g)$', fontsize = 15)
    plt.ylabel(r'$v_{eff}$', fontsize = 15)
    plt.title(fr'Relative renormalization from $v = 0.15$ for schmidt-cut = {schmidt_cut}', fontsize = 15)
    plt.legend()
plt.show()


for (ind, closer) in enumerate([1 + i for i in range(20)]):
    for m_index in range(1, 7):
        #plt.scatter([-0.15*i for i in range(1, 5)], v_renorm[schmidt_number,:,m_index-excluded_0]/(-0.15), label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}')
        #plt.scatter([-0.15*i for i in range(1, 5)], v_renorms[ind,0,:,m_index-excluded_0]/(-0.15))
        plt.errorbar([-0.15*i for i in range(1, 5)], mass_renorms[ind,0,:,m_index-excluded_0]/(0.1*m_index), mass_sigma_renorms[ind,0,:,m_index-excluded_0]/(0.1*m_index), label = f'mass = {round(m_index*0.1,2)}. data points = {51-2*closer}', fmt="o")
        #plt.scatter([-0.15*i for i in range(1, 5)], v_renorm[:,m_index-excluded_0]/(-0.15), label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}')
    plt.xlabel(r'$\Delta (g)$', fontsize = 15)
    plt.ylabel(r'$mass_{eff}$', fontsize = 15)
    plt.title(fr'Relative renormalization of the mass for schmidt-cut = {schmidt_cut}', fontsize = 15)
    plt.legend()
plt.show()

for (ind, closer) in enumerate([1 + i for i in range(20)]):
    for m_index in range(1, 7):
        #plt.scatter([-0.15*i for i in range(1, 5)], v_renorm[schmidt_number,:,m_index-excluded_0]/(-0.15), label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}')
        #plt.scatter([-0.15*i for i in range(1, 5)], v_renorms[ind,0,:,m_index-excluded_0]/(-0.15))
        plt.errorbar([-0.15*i for i in range(1, 5)], c_renorms[ind,0,:,m_index-excluded_0], v_sigma_renorms[ind,0,:,m_index-excluded_0]/(0.15), label = f'mass = {round(m_index*0.1,2)}. data points = {51-2*closer}', fmt="o")
        #plt.scatter([-0.15*i for i in range(1, 5)], v_renorm[:,m_index-excluded_0]/(-0.15), label = f'mass = {round(m_index*0.1,2)}. schmidt = {schmidt_cut}')
    plt.xlabel(r'$\Delta (g)$', fontsize = 15)
    plt.ylabel(r'$c_{eff}$', fontsize = 15)
    plt.title(fr'Relative renormalization of c for schmidt-cut = {schmidt_cut}', fontsize = 15)
    plt.legend()
plt.show()
