import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import csv
from scipy.optimize import curve_fit 
m = 0.3
N = 50

N_values = [50, 100, 200, 400]
delta_gs = [0.0, -0.15, -0.3, -0.45]

def linear(x, a, b, c):
    return a*x**2 + b*x + c

maxima_total = []
for delta_g in delta_gs:
    maxima = []
    for N in N_values:
        file_path = f"SanderDM_Thesis_2324/Bogoliubov/Bogliuobov_parameters_m_{m}_N_{N}_Delta_g_{delta_g}"
        # with open(file_path, "r") as file:
        #     # Read lines from the file and strip newline characters
        #     lines = [line.strip("\t") for line in file.readlines()]
        #     print(lines)

        phis = []
        thetas = []
        with open(file_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')

            for row in spamreader:
                phis.append(float(row[0]))        
                thetas.append(float(row[1]))
        maxima.append(max(thetas))
        X = [(2*np.pi)/N*i - np.pi for i in range(N)]
        plt.plot(X,thetas, label = f"N = {N}")
    plt.xlabel("Momentum", fontsize = 15)
    plt.ylabel(fr"Bogoliubov parameter $\theta$", fontsize = 15)
    plt.legend(fontsize=12)
    # plt.savefig(f"SanderDM_Thesis_2324/Bogoliubov/Figure_Bogliuobov_parameters_theta_m_{m}_N_{N}_Delta_g_{delta_g}.png")
    # plt.show()
    maxima_total.append(maxima)

plt.show()
print(maxima_total)
maxima_total = np.array(maxima_total)

(popt, pcov) = curve_fit(linear, delta_gs, [maxima_total[g][-1] for g in range(len(delta_gs))])
print(popt)
print(pcov)
X_values = np.linspace(min(delta_gs), 0, 1000)
Y_values = [linear(x, popt[0],popt[1], popt[2]) for x in X_values]
for n in range(len(N_values)):
    print(maxima_total[n][:])
    plt.scatter(delta_gs, [maxima_total[g][n] for g in range(len(delta_gs))], label = f"N = {N_values[n]}")
plt.plot(X_values, Y_values, c = 'black', label = 'quadratic fit')
plt.xlabel(r"$\Delta(g)$", fontsize = 15)
plt.ylabel(r"Maximal value of $\theta$", fontsize = 15)
plt.legend()
plt.show()

print(a)

for delta_g in delta_gs:
    for N in N_values:
        file_path = f"SanderDM_Thesis_2324/Bogoliubov/Bogliuobov_parameters_m_{m}_N_{N}_Delta_g_{delta_g}"
        # with open(file_path, "r") as file:
        #     # Read lines from the file and strip newline characters
        #     lines = [line.strip("\t") for line in file.readlines()]
        #     print(lines)

        phis = []
        thetas = []
        with open(file_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')

            for row in spamreader:
                phis.append(float(row[0]))        
                thetas.append(float(row[1]))
        X = [(2*np.pi)/N*i - np.pi for i in range(N)]
        plt.plot(X,phis, label = f"N = {N}")
    plt.xlabel("Momentum", fontsize = 15)
    plt.ylabel(fr"Bogoliubov parameter $\phi$", fontsize = 15)
    plt.legend(fontsize=12)
    plt.savefig(f"SanderDM_Thesis_2324/Bogoliubov/Figure_Bogliuobov_parameters_phis_m_{m}_N_{N}_Delta_g_{delta_g}.png")
    plt.show()
