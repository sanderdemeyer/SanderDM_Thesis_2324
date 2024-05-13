import scipy.integrate as integrate
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.metrics import r2_score
import csv
from matplotlib.animation import FuncAnimation
from scipy.signal import argrelextrema
import scipy.optimize as opt


file = "occupation_numbers_check"
f = h5py.File(file)


with h5py.File(file, 'r') as file:
    # Access the dataset
    dataset = file['occupation_numbers_check']

    # Create a list to store the vectors
    Es = []

    for ref in dataset[:-1]:
        # Use the 'value' attribute to access the data referenced by the object
        vector = file[ref]
        new_vector = [item[0] for item in vector]

        # Convert complex numbers to Python complex numbers
        Es.append(new_vector)


(datapoints, N) = np.shape(Es)

Es = np.array(Es)
best = np.zeros(N)
for i in range(N):
    if i > 24:
        lijst = [abs(j) for j in Es[:,i]]
    else:
        lijst = [abs(1-j) for j in Es[:,i]]
    best[i] = lijst.index(min(lijst))*(2*np.pi/datapoints)
    print(f'best index for i = {i}, is {lijst.index(min(lijst))}')
    
X = [(2*np.pi)/N*i - np.pi for i in range(N)]

plt.plot(X, best)
plt.show()

print(Es[:,5])
print(Es[:,-1])