import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# define time boundaries
tot_time = 1000 # ms
dts_time = 0.1
num_points = int(tot_time/dts_time) + 1

# input signal. Set a few points to 1, rest to zero.
input = np.zeros(num_points)
input[[500, 3000, 6000, 8300]] = 1
xaxis = np.linspace(0, tot_time, num_points)

""" Generate an interesting signal. Sinusoid which goes to zero around. """
kernsig = np.arange(-50, 50, 0.1)
kernel = -np.sin(0.1*kernsig)
kernel[kernsig>0] *= np.exp(-0.2*(kernsig[kernsig>0]))
kernel[kernsig<0] *= np.exp(0.2*kernsig[kernsig<0])

# convolve into the output signal. Return 'same' shape
output = np.convolve(input, kernel, 'same')

# illustrate
fig = plt.figure(figsize=(7.4, 8.8))
gs = GridSpec(3, 1) # (nrows, ncols)
gs.update(wspace = 0.5, hspace = 0.5)

ax = fig.add_subplot(gs[0,:])
ax.set_title("Input signal $x(t)$")
ax.plot(xaxis, input)

ax = fig.add_subplot(gs[1,:])
ax.set_title("Kernel $h(t)$")
ax.plot(kernsig, kernel)
ax.set_ylabel("Amplitudes []", labelpad=16, size=13)

ax = fig.add_subplot(gs[2,:])
ax.set_title(r"Output signal $y(t) = (x \ast h)(t)$")
ax.plot(xaxis, output)
ax.set_xlabel("Time [ms]")

fig.savefig("convolution_example.pdf", bbox_inches='tight')
print("convolution_example.pdf saved.")
plt.close(fig)
