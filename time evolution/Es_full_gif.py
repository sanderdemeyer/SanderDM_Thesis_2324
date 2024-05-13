import numpy as np
import matplotlib.pyplot as plt
import h5py

file = "Es_bhole_trivial_time_evolution_variables_N_140_mass_0.03_delta_g_0.0_ramping_5_dt_0.1_nrsteps_5000_vmax_-2.0_kappa_1.0_trunc_1.5_D_22_savefrequency_5"
f = h5py.File(file, 'r')

with h5py.File(file, 'r') as file:
    # Access the dataset
    dataset = file['Es_full']

    # Create a list to store the vectors
    Es = []

    for ref in dataset[:-1]:
        # Use the 'value' attribute to access the data referenced by the object
        vector = file[ref]
        new_vector = [item for item in vector]

        # Convert complex numbers to Python complex numbers
        Es.append(new_vector)

print(Es)

Es_new = f["Es_full"][:]

print(np.min(Es))
print(np.max(Es))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Initialize an empty figure and axis
fig, ax = plt.subplots()
fig.set_tight_layout(True)

# Initialize an empty plot
line, = ax.plot([], [], 'b-', linewidth=2)

# Set axis limits
ax.set_xlim(0, 69)
ax.set_ylim(-0.33, -0.31)

# title = ax.text(0.5, 1.1, '', transform=ax.transAxes, ha='center')
title_text = ax.text(30, -0.32, '', transform=ax.transAxes, ha='center')

# Function to update the plot for each frame
def update_plot(frame):
    print(frame)
    x = range(69)
    y = Es[frame]
    
    # Update the plot data
    line.set_data(x, y)
    # line.set_title(f"t = {frame}")
    
    plt.title(f"t = {frame}")
    title_text.set_text(f"t = {frame}")

    # Return the line object to update
    return line, title_text

# Create animation
ani = FuncAnimation(fig, update_plot, frames=len(Es), blit=True)

# Save animation as GIF
# ani.save('animated_plot_funcanimation.gif', writer='imagemagick')

plt.show()


fig,ax = plt.subplots(1,1)

def animate(i):
    ax.clear()
    ax.plot(range(69), Es[i])
    ax.set_title(f"t = {i/2}")
    ax.set_ylim([-0.33, -0.31])

ani = FuncAnimation(fig, animate, frames=len(Es))

print('saving')
ani.save('Es_bhole_trivial_time_evolution_variables_N_140_mass_0.03_delta_g_0.0_ramping_5_dt_0.1_nrsteps_5000_vmax_-2.0_kappa_1.0_trunc_1.5_D_22_savefrequency_5.gif')
plt.show()
