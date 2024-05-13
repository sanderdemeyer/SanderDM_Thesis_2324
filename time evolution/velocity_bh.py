import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

delta_g = -0.15
name = f"bhole_time_evolution_variables_N_180_mass_0.06_delta_g_{delta_g}_ramping_5_dt_0.05_nrsteps_2500_vmax_5.0_kappa_1.0_trunc_2.5_D_14_savefrequency_3.jld2_Es_full"

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(name)

# Display the DataFrame
Es = np.array(df)

(nodp, L) = np.shape(Es)
print(nodp)
print(L)

times = [0.05*3*i for i in range(nodp)]
z1 = 41# 35 # 40
z2 = 46# 53 # 49
plt.plot(times, [Es[j][z1] for j in range(nodp)], label  = f"i={2*(z1-1)}")
for z0 in range(z1,z2+1):
    plt.plot(times, [Es[j][z0] for j in range(nodp)], label = f"i={2*z0}")

plt.xlabel("Time", fontsize = 15)
plt.ylabel("Energy", fontsize = 15)
plt.legend(fontsize=12)
plt.show()

max_time_considered = nodp
# max_time_considered = 90*5

minimal_time_considered = 0 # 18 # 49
minimal_energy_considered = -0.33993
first_maximum = []
for z0 in range(z1, z2+1):
    for j in range(2,max_time_considered):
        if ((Es[j][z0] > Es[j-1][z0]) and (Es[j][z0] > Es[j+1][z0]) and (times[j] > minimal_time_considered) and (Es[j][z0] > minimal_energy_considered)):
            print(f"For z0 = {z0}, we get j = {j}, and t = {times[j]}")
            first_maximum.append(times[j])
            break
        if j == max_time_considered-1:
            first_maximum.append(0)


print(len(range(z1,z2+1)))
print(len(first_maximum))
plt.scatter(range(z1,z2+1), first_maximum)
plt.show()



# Define the model function (polynomial)
def model(x, *params):
    return np.polyval(params, x)

# Initial guess for the polynomial coefficients
p0 = [1.0, 1.0]  # Adjust according to the degree of the polynomial

# Perform the polynomial fit
params, covariance_matrix = curve_fit(model, first_maximum, range(z1,z2+1), p0)

# Calculate the standard deviations from the covariance matrix
std_devs = np.sqrt(np.diag(covariance_matrix))

# Output fitted parameters and their standard deviations
print("Fitted parameters:", 2*params)
print("Standard deviations:", 2*std_devs)

x_values = np.linspace(min(first_maximum), max(first_maximum), 100)
expected_values = model(x_values, *params)
maximal_values = model(x_values, *(params + std_devs))
minimal_values = model(x_values, *(params - std_devs))

plt.scatter(first_maximum, [2*e for e in range(z1,z2+1)], label='Data')

# Plot the fitted curve with expected, maximal, and minimal values
plt.plot(x_values, 2*expected_values, color='black', label = "Fit")
plt.plot(x_values, 2*maximal_values, linestyle='--', color='red', label=r'Fit $\pm$ 1$\sigma$')
plt.plot(x_values, 2*minimal_values, linestyle='--', color='red')
plt.xlabel("Time of first wave arrival", fontsize = 15)
plt.ylabel("Position", fontsize = 15)
plt.legend()
plt.show()