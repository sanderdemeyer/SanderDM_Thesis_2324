import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.metrics import r2_score
import csv

delta_g = 0.0
mass = 0.0
v = 1.25

k = np.linspace(-np.pi/6,np.pi/6,11)
k_pos = np.linspace(0, np.pi/2, 1000)
k_neg = np.linspace(-np.pi/2, 0, 1000)


exp_00 = (v/2)*np.sin(2*k_pos) + np.sqrt(mass**2 + np.sin(k_pos)**2)
exp_10 = (v/2)*np.sin(2*k_pos) - np.sqrt(mass**2 + np.sin(k_pos)**2)
exp_11 = (v/2)*np.sin(2*k_neg) + np.sqrt(mass**2 + np.sin(k_neg)**2)
exp_01 = (v/2)*np.sin(2*k_neg) - np.sqrt(mass**2 + np.sin(k_neg)**2)


normal = False


if normal:

    plt.plot(k_pos, exp_00, linestyle='dotted', color = 'blue', label = 'right moving')
    plt.plot(k_pos, exp_10, color = 'blue', label = 'left moving')
    plt.plot(k_neg, exp_01, linestyle='dotted', color = 'blue')
    plt.plot(k_neg, exp_11, color = 'blue')
    # plt.plot(k_pos, exp_00, color = 'blue')
    # plt.plot(k_pos, exp_10, color = 'blue')
    # plt.plot(k_neg, exp_01, color = 'blue')
    # plt.plot(k_neg, exp_11, color = 'blue')
    plt.xlabel('k', fontsize = 15)
    plt.ylabel('energy', fontsize = 15)
    plt.legend(fontsize=13)
    # plt.title(fr"Dispersion relation for $m = {mass}$, $v = {v}$, and $\Delta(g)={delta_g}$")
    plt.savefig(f"Theoretical Dispersion relation for m = {mass} and v = {v}.png")
    plt.show()
else:
    exp_00_switch = [((exp_00[i] > exp_00[i-1]) and (exp_00[i] > exp_00[i+1]))*1 for i in range(1,len(exp_00)-1)]
    exp_01_switch = [((exp_01[i] < exp_01[i-1]) and (exp_01[i] < exp_01[i+1]))*1 for i in range(1,len(exp_01)-1)]

    k_switch00 = list(exp_00_switch).index(1) + 1
    k_switch01 = list(exp_01_switch).index(1) + 1

    if v <= 1.0:
        plt.plot(k_pos[:k_switch00], exp_00[:k_switch00], linestyle='dotted', color = 'blue', label = 'right moving')
        plt.plot(k_pos[k_switch00:], exp_00[k_switch00:], color = 'blue')
        # plt.plot(k_pos, exp_00, linestyle='dotted', color = 'blue', label = 'right moving')
        plt.plot(k_pos, exp_10, color = 'blue', label = 'left moving')
        plt.plot(k_neg[:k_switch01], exp_01[:k_switch01], color = 'blue')
        plt.plot(k_neg[k_switch01:], exp_01[k_switch01:], linestyle='dotted', color = 'blue')
        plt.plot(k_neg, exp_11, color = 'blue')
        # plt.plot(k_pos, exp_00, color = 'blue')
        # plt.plot(k_pos, exp_10, color = 'blue')
        # plt.plot(k_neg, exp_01, color = 'blue')
        # plt.plot(k_neg, exp_11, color = 'blue')
        plt.xlabel('k', fontsize = 15)
        plt.ylabel('energy', fontsize = 15)
        plt.legend(fontsize=13)
        # plt.title(fr"Dispersion relation for $m = {mass}$, $v = {v}$, and $\Delta(g)={delta_g}$")
        plt.savefig(f"Theoretical Dispersion relation for m = {mass} and v = {v}.png")
        plt.show()
    else:
        exp_10_switch = [((exp_10[i] > exp_10[i-1]) and (exp_10[i] > exp_10[i+1]))*1 for i in range(1,len(exp_10)-1)]
        exp_11_switch = [((exp_11[i] < exp_11[i-1]) and (exp_11[i] < exp_11[i+1]))*1 for i in range(1,len(exp_11)-1)]
        k_switch10 = list(exp_10_switch).index(1) + 1
        k_switch11 = list(exp_11_switch).index(1) + 1

        plt.plot(k_pos[:k_switch00], exp_00[:k_switch00], linestyle='dotted', color = 'blue', label = 'right moving')
        plt.plot(k_pos[k_switch00:], exp_00[k_switch00:], color = 'blue')
        # plt.plot(k_pos, exp_00, linestyle='dotted', color = 'blue', label = 'right moving')
        plt.plot(k_pos[:k_switch10], exp_10[:k_switch10], linestyle = 'dotted', color = 'blue')
        plt.plot(k_pos[k_switch10:], exp_10[k_switch10:], color = 'blue', label = 'left moving')
        plt.plot(k_neg[:k_switch01], exp_01[:k_switch01], color = 'blue')
        plt.plot(k_neg[k_switch01:], exp_01[k_switch01:], linestyle='dotted', color = 'blue')
        plt.plot(k_neg[:k_switch11], exp_11[:k_switch11], color = 'blue')
        plt.plot(k_neg[k_switch11:], exp_11[k_switch11:], linestyle='dotted', color = 'blue')
        # plt.plot(k_pos, exp_00, color = 'blue')
        # plt.plot(k_pos, exp_10, color = 'blue')
        # plt.plot(k_neg, exp_01, color = 'blue')
        # plt.plot(k_neg, exp_11, color = 'blue')
        plt.xlabel('k', fontsize = 15)
        plt.ylabel('energy', fontsize = 15)
        plt.legend(fontsize=13)
        # plt.title(fr"Dispersion relation for $m = {mass}$, $v = {v}$, and $\Delta(g)={delta_g}$")
        plt.savefig(f"Theoretical Dispersion relation for m = {mass} and v = {v}.png")
        plt.show()


