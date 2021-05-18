import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm
import matplotlib.colors as colors

r1 = (0, 1) # x, y
r2 = (0, -1)
sigma = 1

def dipole(x, y):
    fac = 1./4*np.pi*sigma
    dp1 = 1./(np.sqrt((x-r1[0])**2 + (y-r1[1])**2))
    dp2 = -1./(np.sqrt((x-r2[0])**2 + (y-r2[1])**2))
    return fac*(dp1 + dp2)


dims = np.linspace(-5, 5, 401)

X,Y = np.meshgrid(dims, dims)
Z = dipole(X,Y)

fig, ax = plt.subplots()
im = ax.pcolormesh(X,Y,Z,
    norm=colors.BoundaryNorm(boundaries=np.linspace(-1, 1, 21), ncolors=256),
    shading='gouraud',
    cmap='RdBu_r',)

ax.set_xlabel("X []")
ax.set_ylabel("Y []")
ax.grid(linestyle="--")
fig.colorbar(im, ax=ax)
plt.savefig("dipole_conceptualization.pdf",
    bbox_inches='tight')
