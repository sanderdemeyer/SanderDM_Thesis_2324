import numpy as np
import matplotlib.pyplot as plt


# Define the file path
file_path = "measure_of_eigenvalues.csv"

# Read the data from the file
with open(file_path, "r") as file:
    # Read lines from the file and strip newline characters
    lines = [line.strip() for line in file.readlines()]

# Convert the lines to a 2D NumPy array
traces = np.array([[float(entry) for entry in line.split(",")] for line in lines])

# Print the array
print(traces)

for i in range(5):
    plt.scatter([10*j for j in range(1,11)], traces[i], label = fr"$\Delta(g) = {round(-0.15*(i),2)}$")
plt.xscale('log')
plt.yscale('log')

plt.xlabel(r"System size $N$", fontsize = 15)
plt.ylabel(r"$\frac{2}{N} \text{Tr} \left(C (1-C)\right)$", fontsize = 15)
plt.legend()
plt.show()