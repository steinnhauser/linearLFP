import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.signal as signal

import utils.plotter as pf

T = 2200
dt = 0.0625
Nt = int(T/dt) + 1
xaxis_time_ms = np.arange(Nt)*dt

first_sig = np.zeros(Nt)
np.random.seed(103)
ones = np.random.randint(0,Nt-1, size=11)
first_sig[ones] = 1

size = 11
ylabelx = -0.138
ylabely = 0.39

def transform_to_gaussian_spread(sig, gaussian_width_ms):
    gaussian_sizes_ms = gaussian_width_ms * 10 + 1
    gaussian_width = int(gaussian_width_ms/dt)
    gaussian_sizes = int(gaussian_sizes_ms/dt)

    gauss_kernel = signal.gaussian(gaussian_sizes, std=gaussian_width)
    transformed_firing_rate = np.convolve(gauss_kernel, sig, 'same')
    return transformed_firing_rate

fig = plt.figure(figsize=(7.4, 5.8))
gs = GridSpec(4, 1) # (nrows, ncols)
gs.update(wspace = 0.5, hspace = 0.4)
ax1 = fig.add_subplot(gs[0,:])
ax1.plot(xaxis_time_ms, first_sig)
ax1.grid(axis='y', linestyle='--')
pf.remove_axis_junk(ax1, lines=['top', 'bottom'])
ax1.set_xticks([])
ax1.set_ylabel(f"Raw signal", rotation=0, size=size)
ax1.get_yaxis().set_label_coords(ylabelx, ylabely)

for n, width in enumerate([5, 10, 20]):
    ax1 = fig.add_subplot(gs[n+1,:])
    fr = transform_to_gaussian_spread(first_sig, width)
    ax1.plot(xaxis_time_ms, fr)
    ax1.set_ylabel(f"$\sigma=${width}ms", rotation=0, size=size)
    ax1.get_yaxis().set_label_coords(ylabelx, ylabely)
    ax1.grid(axis='y', linestyle='--')
    if n!=2:
        pf.remove_axis_junk(ax1, lines=['top', 'bottom'])
        ax1.set_xticks([])
    else:
        pf.remove_axis_junk(ax1, lines=['top'])
        ax1.set_xlabel("Time [ms]")


fig.savefig("delta_convolution_example.pdf", bbox_inches='tight')
print("delta_convolution_example.pdf saved.")
plt.close(fig)
