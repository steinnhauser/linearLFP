import numpy as np
import scipy.signal as signal

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# define time boundaries
tot_time = 1000 # ms
dts_time = 1
num_points = int(tot_time/dts_time) + 1

# generate input signal
xaxis = np.linspace(0, tot_time, num_points)
input = np.sin(0.1*xaxis) + np.cos(0.08*xaxis) + np.random.normal(0,1,num_points)

# generate triangular kernel signal
kernel = np.zeros(100)
n = 0
for i in range(45, 55):
    kernel[i] += n
    if i < 50:
        n+=0.2
    else:
        n-=0.2

# convolve into the output signal. Return 'same' shape
output = np.convolve(input, kernel, 'same')

# plot
fig = plt.figure(figsize=(7.4, 8.8))
gs = GridSpec(3, 1) # (nrows, ncols)
gs.update(wspace = 0.5, hspace = 0.5)

ax = fig.add_subplot(gs[0,:])
ax.set_title("Input signal $x(t)$")
ax.plot(xaxis, input)

ax = fig.add_subplot(gs[1,:])
ax.set_title("Kernel $h(t)$")
ax.plot(np.linspace(-50, 50, 100), kernel)
ax.set_ylabel("Amplitudes []", labelpad=16, size=13)

ax = fig.add_subplot(gs[2,:])
ax.set_title(r"Output signal $y(t) = (x \ast h)(t)$")
ax.plot(xaxis, output)
ax.set_xlabel("Time [ms]")


fig.savefig("convolution_intro.pdf", bbox_inches='tight')
print("convolution_intro.pdf saved.")
plt.close(fig)
