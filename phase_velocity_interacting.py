import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

am_tilde_0 = 0.3
Delta_g = -0.15
v = 0.0
trunc = 4.0
bounds_k = np.pi/2

file = f"SanderDM_Thesis_2324/data/Derivative_info_Dispersion_Delta_m_{am_tilde_0}_delta_g_{Delta_g}_v_{v}_trunc_{trunc}_all_sectors_newv_more.csv"

energies = []

with open(file, newline='') as csvfile:
    # Create a CSV reader object
    csvreader = csv.reader(csvfile)
    
    # Iterate over each row in the CSV file
    for row in csvreader:
        # Each row is a list of strings representing the values in that row
        # Print the row
        energies.append(float(row[0]))

k_values = np.linspace(-bounds_k, bounds_k,len(energies))

def polynomial(x, *params):
    summation = 0
    for (i,param) in enumerate(params):
        summation += param * x**i
    return summation

def polynomial_derivative(x, *params):
    summation = 0
    for (i,param) in enumerate(params):
        summation += param * i * x**(i-1)
    return summation

order = 16
p0 = [1] * order
popt, pcov = curve_fit(polynomial, k_values, energies, p0 = p0)#, method = 'trf')

k_values_finer = np.linspace(-bounds_k,bounds_k, 1000)

split = True
if split:
    plt.plot(2*k_values, [e - Delta_g for e in energies], label = 'U(1) sector +1')
    plt.plot(2*k_values, [e + Delta_g for e in energies], label = 'U(1) sector -1')
else:
    plt.plot(2*k_values_finer, [polynomial(x, *popt) for x in k_values_finer], label = 'polynomial fit', c = 'red')
    plt.scatter(2*k_values, energies, label = 'data', s = 3)

plt.xlabel(r"momentum $k$", fontsize = 15)
plt.ylabel("energy", fontsize = 15)
plt.legend(fontsize=12)
plt.show()

k = 0.5

def get_data(k, popt, pcov, order):
    vel = polynomial_derivative(k, *popt)

    param_plus = [popt[i]+pcov[i,i] for i in range(order)]
    param_min = [popt[i]-pcov[i,i] for i in range(order)]

    vel_plus = polynomial_derivative(k, *param_plus) 
    vel_min = polynomial_derivative(k, *param_min) 

    print(f"For k = {k}: derivative is {vel/2}, max is {vel_plus/2}, min is {vel_min/2}")

    print(f"For k = {k}: std is {(vel_plus-vel)/2} or {(vel-vel_min)/2}")

get_data(0.5, popt, pcov, order)
get_data(-0.75, popt, pcov, order)
