"""
Script which illustrates the Hay cell and shows where the synapses are connected.
Perhaps generalize this to do the same with the BaS cells.
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import neuron
import LFPy
import h5py
import pandas as pd
import numpy as np
from LFPy import NetworkCell, Network, Synapse, RecExtElectrode, TemplateCell
from LFPy.inputgenerators import get_activation_times_from_distribution

import utils.plotter as pf

# fetch synapses connected to the cell.
READPATH = "./true_sim/"
f2 = h5py.File(READPATH + 'synapse_positions.h5', "r")

headers2 = ['gid', 'idx', 'weight', 'delay', 'pre_gid']
f2E_E = pd.DataFrame(np.array(f2.get('E:E')), columns = headers2)
f2E_I = pd.DataFrame(np.array(f2.get('E:I')), columns = headers2)
f2I_E = pd.DataFrame(np.array(f2.get('I:E')), columns = headers2)
f2I_I = pd.DataFrame(np.array(f2.get('I:I')), columns = headers2)
df2 = pd.concat({'E:E':f2E_E,'E:I':f2E_I,'I:E':f2I_E,'I:I':f2I_I})

f2.close()

neuron.load_mechanisms('L5bPCmodelsEH/mod/')
cellParameters = { # excitatory cells are modeled using the Hay model
    'morphology'    : 'L5bPCmodelsEH/morphologies/cell1.asc',
    'templatefile'  : ['L5bPCmodelsEH/models/L5PCbiophys3.hoc',
                       'L5bPCmodelsEH/models/L5PCtemplate.hoc'],
    'templatename'  : 'L5PCtemplate',
    'templateargs'  : 'L5bPCmodelsEH/morphologies/cell1.asc',
    'passive' : False,
    'nsegs_method' : None,
    'pt3d' : True,
    'delete_sections' : False,
}

cell = LFPy.TemplateCell(**cellParameters)
cell.set_rotation(x=4.729, y=-3.166)

fig = plt.figure(figsize=(7.4, 5.8*2))
gs = GridSpec(1, 4) # (nrows, ncols)
gs.update(wspace = 0.5, hspace = 0.5)
ax1 = fig.add_subplot(gs[:,:2])
# fig = plt.figure(dpi=160)
# ax1 = fig.add_axes([0.05, 0.1, 0.55, 0.9], frameon=False)

ax1.set_xlim(-200, 300)
ax1.set_ylim(-200, 1200)
# plot morphology
zips = []
for x, z in cell.get_pt3d_polygons():
    zips.append(list(zip(x, z)))
from matplotlib.collections import PolyCollection
polycol = PolyCollection(zips, edgecolors='none',
                         facecolors='black', zorder=-1, rasterized=False)
ax1.add_collection(polycol)

ax1.set_xlabel("x [µm]", size=13)
ax1.set_ylabel("z [µm]", size=13)

# fetch all the connection indexes for this cell.
populations = [1024, 256]

# Set up loop for all synapses connected to current cell.
vals = {"E" : [], "I" : []}
for _, row in df2.iterrows():
    name = row.name[0]
    pre_type = name[0]
    post_type = name[-1]

    if post_type=="E":
        delay  = row['delay']
        weight = row['weight']
        pregid = int(row['pre_gid'])
        idxnum = int(row['idx'])
        vals[pre_type].append(cell.zmid[idxnum])

ax1.axis('equal')
pf.remove_axis_junk(ax1, lines=['right', 'top'])


ax2 = fig.add_subplot(gs[:,2])
ax2.hist(vals["E"], bins=120, orientation="horizontal", range=(-200, 1200), color="C0")
ax2.grid(axis="x", linestyle="--")
ax2.set_yticks([])
pf.remove_axis_junk(ax2, lines=['right', 'top'])
ax2.set_xlabel("E$_{\mathrm{syn}}$", size=13)

ax3 = fig.add_subplot(gs[:,3])
ax3.hist(vals["I"], bins=120, orientation="horizontal", range=(-200, 1200), color="C1")
ax3.grid(axis="x", linestyle="--")
ax3.set_yticks([])
pf.remove_axis_junk(ax3, lines=['right', 'top'])
ax3.set_xlabel("I$_{\mathrm{syn}}$", size=13)

fig.savefig("./figures/morphology_E.pdf", bbox_inches='tight')
print("./figures/morphology_E.pdf saved.")
plt.close(fig)


""" Perhaps do the other neuron as well? """

cellParameters = { # inhibitory cells are modeled as ball-and-stick for now.
    'morphology' : './init/BallAndStick_active_shortened.hoc',    # use hh for generation.
    'templatefile' : './init/BallAndStickTemplate.hoc',
    'templatename' : 'BallAndStickTemplate',
    'templateargs' : None,
    'delete_sections' : False,
    'pt3d' : True,
}
cell = LFPy.TemplateCell(**cellParameters)

# Set up loop for all synapses connected to current cell.
vals = {"E" : [], "I" : []}
for _, row in df2.iterrows():
    name = row.name[0]
    pre_type = name[0]
    post_type = name[-1]

    if post_type=="I":
        delay  = row['delay']
        weight = row['weight']
        pregid = int(row['pre_gid'])
        idxnum = int(row['idx'])
        vals[pre_type].append(cell.zmid[idxnum])

fig = plt.figure(figsize=(7.4, 5.8))
gs = GridSpec(1, 4) # (nrows, ncols)
gs.update(wspace = 0.5, hspace = 0.5)
ax1 = fig.add_subplot(gs[:,:2])
ax1.set_xlim(-200, 300)
ax1.set_ylim(-200, 1200)

# plot morphology
zips = []
for x, z in cell.get_pt3d_polygons():
    zips.append(list(zip(x, z)))
from matplotlib.collections import PolyCollection
polycol = PolyCollection(zips, edgecolors='none',
                         facecolors='black', zorder=-1, rasterized=False)
ax1.add_collection(polycol)


# ax1.axis("equal")
ax1.set(xlim=(-60, 60), ylim=(-50, 150))
ax1.set_xlabel("x [µm]", size=13)
ax1.set_ylabel("z [µm]", size=13)
pf.remove_axis_junk(ax1, lines=['right', 'top'])
ylims = ax1.get_ylim()

ax2 = fig.add_subplot(gs[:,2])
ax2.hist(vals["E"], bins=120, orientation="horizontal", color="C0", range=ylims)
ax2.grid(axis="x", linestyle="--")
ax2.set_yticks([])
pf.remove_axis_junk(ax2, lines=['right', 'top'])
ax2.set_xlabel("E$_{\mathrm{syn}}$", size=13)

ax3 = fig.add_subplot(gs[:,3])
ax3.hist(vals["I"], bins=120, orientation="horizontal", color="C1", range=ylims)
ax3.grid(axis="x", linestyle="--")
ax3.set_yticks([])
pf.remove_axis_junk(ax3, lines=['right', 'top'])
ax3.set_xlabel("I$_{\mathrm{syn}}$", size=13)

fig.savefig("./figures/morphology_I.pdf", bbox_inches='tight')
print("./figures/morphology_I.pdf saved.")
plt.close(fig)
