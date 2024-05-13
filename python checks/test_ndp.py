import numpy as np


smallest = np.pi/72/24/8

print(smallest)

factor_max = 48
list_of_indices = [0]

for factor in range(1,factor_max+1):
    for j in range(1,5):
        if j*factor not in list_of_indices:
            list_of_indices.append(j*factor*smallest)
            list_of_indices.append(-j*factor*smallest)

print(list_of_indices)
print(len(list_of_indices))