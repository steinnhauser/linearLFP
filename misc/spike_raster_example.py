import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.signal as scpsig


# custom masters utils
import utils.plotter as pf

fig = plt.figure(figsize=(7.4, 3.8))
gs = GridSpec(4, 2) # (nrows, ncols)
gs.update(wspace = 0.12, hspace = 0.2)

Time = 50 # ms
Dt = 0.1 # ms
Nt = int(Time/Dt) + 1
Nn = 500
timespace = np.linspace(0, Time, Nt)

def gaussian_convolution(signal):
    gaussian_width_ms = 0.3 #ms
    gaussian_sizes_ms = gaussian_width_ms * 10 + 1
    gaussian_width = int(gaussian_width_ms/Dt)
    gaussian_sizes = int(gaussian_sizes_ms/Dt)

    gauss_kernel = scpsig.gaussian(gaussian_sizes, std=gaussian_width)
    transformed_firing_rate = np.convolve(signal, gauss_kernel, 'same')
    return transformed_firing_rate

############### Construct the first spike raster example
spikes = np.random.exponential(scale=5, size=(Nn,Nt))
neuron, timesteps = spikes.shape
for t in range(timesteps-1):
    spikes[:, t+1] += spikes[:, t]

tS = []
tG = []
for i in range(neuron):
    for j in spikes[i]:
        if j < Time:
            tS.append(j)
            tG.append(i)

# total spikes per time
delta_signal, _ = np.histogram(tS,bins=Nt,range=(0, Time))
delta_signal = gaussian_convolution(delta_signal)

ax = fig.add_subplot(gs[:3,0])
ax.plot(tS, tG, "r,")
ax.set_xlim(0, Time)
pf.remove_axis_junk(ax, lines=['right', 'top'])
ax.set_ylabel("Neuron ID")
ax.set_xticks([])

ax = fig.add_subplot(gs[3,0])
ax.set_ylabel("Spike no.")
pf.remove_axis_junk(ax, lines=['right', 'top'])
ax.plot(timespace, delta_signal)
ax.set_xlim(0, Time)
ax.set_xlabel("Time [ms]")
ax.grid(axis="y", linestyle="--")

############### Construct the second spike raster example
spikes = np.random.exponential(scale=5, size=(Nn,Nt))
neuron, timesteps = spikes.shape
for t in range(timesteps-1):
    spikes[:, t+1] += spikes[:, t]

tS = []
tG = []
for i in range(neuron):
    for j in spikes[i]:
        if j < Time:
            tS.append(j)
            tG.append(i)

# create a sin signal of synchronicity
sync = np.sin(1.2*timespace)
keeptimes = timespace[np.random.rand(Nt) < sync]
tS_ = []
tG_ = []
for time, gid in zip(tS, tG):
    # round to nearest 0.2 and see if it is in keeptimes
    if round(0.2*np.floor(round(time / 0.2,2)),1) in keeptimes:
        tS_.append(time)
        tG_.append(gid)


delta_signal, _ = np.histogram(tS_,bins=Nt,range=(0, Time))
delta_signal = gaussian_convolution(delta_signal)

ax = fig.add_subplot(gs[:3,1])
ax.plot(tS_, tG_, "r,")
ax.set_xlim(0, Time)
pf.remove_axis_junk(ax, lines=['right', 'top'])
ax.set_yticks([])
ax.set_xticks([])

ax = fig.add_subplot(gs[3,1])
pf.remove_axis_junk(ax, lines=['right', 'top'])
ax.plot(timespace, delta_signal)
ax.set_xlim(0, Time)
ax.set_xlabel("Time [ms]")
ax.grid(axis="y", linestyle="--")
ylow, yhigh = ax.get_ylim()
ax.set_ylim(0, yhigh)
# ax.set_yticks([])

fig.savefig("spike_raster_example.pdf", bbox_inches='tight')
print("spike_raster_example.pdf saved.")
plt.close(fig)

fig.savefig("spike_raster_example.pdf", bbox_inches='tight')
print("spike_raster_example.pdf saved.")
plt.close(fig)
