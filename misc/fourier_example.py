"""
Example fetched from scipy documentation. Some changes added.
https://docs.scipy.org/doc/scipy/reference/tutorial/fft.html
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.fft import fft, fftfreq

# custom masters utils
import utils.plotter as pf


N = 600
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N, endpoint=False)
y = np.sin(100 * 2.0*np.pi*x) + np.sin(250 * 2.0*np.pi*x)
y += np.random.normal(0,0.8,N) # add some noise

# apply FFT
yf = fft(y)
xf = fftfreq(N, T)[:N//2]

fig = plt.figure(figsize=(7.4, 5.8))
gs = GridSpec(2, 1) # (nrows, ncols)
gs.update(wspace = 0.5, hspace = 0.5)

ax = fig.add_subplot(gs[0,:])
pf.remove_axis_junk(ax, lines=['right', 'top'])
ax.plot(x, y)
ax.grid(axis="y", linestyle="--")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Amplitude []")
ax.set_title("$s(t) = \sin(100 \cdot 2\pi t) + \sin(250 \cdot 2\pi t) + \mathcal{N}(0, 0.8)$")

ax = fig.add_subplot(gs[1,:])
pf.remove_axis_junk(ax, lines=['right', 'top'])
ax.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
ax.grid(axis="y", linestyle="--")
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Amplitude []")
ax.set_title("Fourier Transformed signal $\mathcal{F}(s(t))$")

fig.savefig("fourier_example.pdf", bbox_inches='tight')
print("fourier_example.pdf saved.")
plt.close(fig)
