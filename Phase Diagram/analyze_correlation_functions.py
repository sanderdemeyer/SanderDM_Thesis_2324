import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.metrics import r2_score
import csv

def fit_f(x, B, A, eta):
    return B*(x**eta)*(A**x)

def fit_exp(x, C, xi):
    return C*np.exp(x*(-xi))

def fit_PL(x, C, eta):
    return C*(x**eta)

def fit_linear(x, a, b):
    return a*x+b

f = h5py.File("SanderDM_Thesis_2324/Thirring_correlation_functions_new", "r")
f = h5py.File("SanderDM_Thesis_2324/Thirring_2023_10_18_E_and_corr", "r")
corr_functions = f["corr_functions"][:]
energies = f["energies"][:]
print(np.shape(corr_functions))

R2_exp_values = np.zeros((10, 5))
R2_PL_values = np.zeros((10, 5))
Energies_xy = np.zeros((10, 5))

for i in range(10):
    for j in range(5):
        corr = corr_functions[:,i,j]
        corr_clean = [corr[2*i][0] for i in range(50)]

        Energies_xy[i,j] = (energies[:,i,j][0][0] + energies[:,i,j][1][0])/2

        """
        popt, pcov = scipy.optimize.curve_fit(fit_exp, range(50), corr_clean)
        y_pred_exp = fit_exp(range(50), *popt)
        R2_exp = r2_score(corr_clean, y_pred_exp)
        popt, pcov = scipy.optimize.curve_fit(fit_PL, range(50), corr_clean)
        y_pred_PL = fit_PL(range(50), *popt)
        R2_PL = r2_score(corr_clean, y_pred_PL)
        print(f"for i = {i} and j = {j}, The R^2 values for exp = {R2_exp} and PL = {R2_PL}")
        print(f"for i = {i} and j = {j}, the optimized parameters are")
        print(popt)
        plt.plot(range(50), [np.log(abs(e)) for e in corr_clean], label = 'data')
        plt.plot(range(50), [np.log(abs(e)) for e in y_pred_exp], label = 'exponential')
        plt.plot(range(50), [np.log(abs(e)) for e in y_pred_PL], label = 'power law')
        plt.legend()
        plt.show()
        """

        popt, pcov = scipy.optimize.curve_fit(fit_linear, range(50), [np.log(abs(e)) for e in corr_clean])
        y_pred_exp = fit_linear(range(50), *popt)
        R2_exp = r2_score([np.log(abs(e)) for e in corr_clean], y_pred_exp)
        R2_exp_values[i,j] = R2_exp

        popt, pcov = scipy.optimize.curve_fit(fit_linear, np.array([np.log(e) for e in range(1,50)]), [np.log(abs(e)) for e in corr_clean[1:]])
        y_pred_exp = fit_linear(np.array([np.log(e) for e in range(1,50)]), *popt)
        R2_PL = r2_score([np.log(abs(e)) for e in corr_clean[1:]], y_pred_exp)
        R2_PL_values[i,j] = R2_PL

        print(f"for i = {i} and j = {j}, The R^2 value for exp = {R2_exp}")
        print(f"for i = {i} and j = {j}, the optimized parameters are")
        print(popt)
        # plt.plot(range(50), [np.log(abs(e)) for e in corr_clean], label = 'data')
        # plt.plot(range(49), y_pred_exp, label = 'exponential')
        # plt.show()

masses = [0, 0.1, 0.241, 0.405, 0.5]
delta_gs = [0.5, 0.375, 0.25, 0.125, 0, -0.2, -0.4, -0.6, -0.8, -1]

rows = []
with open("SanderDM_Thesis_2324/Data_Banuls/data_m_0.txt", 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        rows.append([float(e) for e in row])
rows = np.array(rows)

for j in range(5):
    plt.scatter([np.pi - 2*np.arccos(e) for e in delta_gs], Energies_xy[:,j], label = f'Own data for {masses[j]}')
plt.plot(rows[:,0], rows[:,1], label = 'Data Banuls-2019 for m = 0')
plt.xlabel('g', fontsize = 15)
plt.ylabel(r'$E_{gs}$', fontsize = 15)
plt.legend()
plt.show()

X_values = [0, 0.2, 0.4, 0.6, 0.8, 1]
Y_values = [0, 0.1, 0.2, 0.3, 0.4]
X_values = delta_gs
Y_values = masses
x = [X_values for i in range(5)]
y = [[Y_values[i] for j in range(10)] for i in range(5)]

x_flat = []
y_flat = []
R_exp_flat = []
R_PL_flat = []
E_flat = []

for j in range(5):
    x_flat.extend(X_values)
    y_flat.extend([Y_values[j] for kl in range(10)])
    R_exp_flat.extend(R2_exp_values[:,j])
    R_PL_flat.extend(R2_PL_values[:,j])
    E_flat.extend(Energies_xy[:,j])


print(np.shape(x_flat))
print(np.shape(y_flat))
# Create a heatmap plot
#plt.imshow(R_values, cmap='viridis')  # 'viridis' is just an example colormap; choose one you prefer
plt.scatter(np.array(x_flat), y_flat, c=R_exp_flat, cmap='viridis')  # 'viridis' is just an example colormap; choose one you prefer

plt.colorbar()  # Add a colorbar to the plot for reference


# Add labels and title
plt.xlabel(r'$\Delta$ (g)')
plt.ylabel(r'$am_0$')
plt.title('R2 value for exponential fit')

# Show the plot
plt.show()

