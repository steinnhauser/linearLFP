#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Script identical to that of reproduce_LFP.py, though the conductance-based
synapse model is now changed with a current-based one. There are several ways
in which to scale the current based one correctly, so this script will include
a few. Some contenders being:
    * Simply scale it by the resting soma potential ~ -65mV

These values ultimately create the weight scaling factor via the driving force
equation (V-Esyn), where V is one of the factors above and Esyn is the synaptic
resting potential (typically around -10mV for excitatory synapses). """

"""
Code which reads in the spiking times generated by the example_network.py code
from file savedSpikes.txt and attempts to recreate the LFP signal detected
without needing to simulate the biophysically modelled multicompartemental cells.

Run command:
    mpirun --use-hwthread-cpus -np 4 python3 recreate_LFP.py
"""

import numpy as np
import scipy.stats as st
from mpi4py import MPI
import neuron
import json
import os, sys
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from LFPy import NetworkCell, Network, Synapse, RecExtElectrode

# import custom modules
import utils.plotter as pf
import utils.misc as mf
import utils.tester as tf

# Load in the .json setup file.
jfile = open("init/setup.json", "r", encoding="utf-8")
jdict = json.load(jfile)
jfile.close()

args = vars(mf.argument_parser())
n_idx_outside_stim = args["n_idx"]
scale_factor_population = args["scale_factor"]
run_name = args["run_name"]
if run_name == "default_js":    # fetch from setup.json
    run_name = jdict["run_name"]
elif run_name == "default_sf":  # write scale factor into name
    run_name = "run" + str(scale_factor_population)

READPATH = './true_sim/'        # path for read files
OUTPUTPATH = './fake_sim_isyn/'      # path for write files

# Read in some HDF files generated by example_network
f1 = h5py.File(READPATH + 'cell_positions_and_rotations.h5', "r")
f2 = h5py.File(READPATH + 'synapse_positions.h5', "r")

headers1 = ['gid', 'x', 'y', 'z', 'x_rot', 'y_rot','z_rot']
f1E = pd.DataFrame(np.array(f1.get('E')), columns=headers1)
f1I = pd.DataFrame(np.array(f1.get('I')), columns=headers1)
df1 = pd.concat({'E':f1E, 'I':f1I})

headers2 = ['gid', 'idx', 'weight', 'delay', 'pre_gid']
f2E_E = pd.DataFrame(np.array(f2.get('E:E')), columns = headers2)
f2E_I = pd.DataFrame(np.array(f2.get('E:I')), columns = headers2)
f2I_E = pd.DataFrame(np.array(f2.get('I:E')), columns = headers2)
f2I_I = pd.DataFrame(np.array(f2.get('I:I')), columns = headers2)
df2 = pd.concat({'E:E':f2E_E,'E:I':f2E_I,'I:E':f2I_E,'I:I':f2I_I})

f1.close(); f2.close()

# Read in the spikes saved from previous simulations
sph5 = h5py.File(READPATH + "savedSpikes.h5", "r")
spikeTimes = []; spikeGid = []
for pop in jdict["population_names"]:
    group = sph5.get(pop)
    spikeTimes += group.get("spikeTimes")[:].tolist()
    spikeGid += group.get("spikeGid")[:].tolist()
sph5.close()

# Generate [GID, spike time] list
numGids = int(max(spikeGid))+1  # should equal total number of cells
gid_spikes = [[] for _ in range(numGids)]
for time, gid in zip(spikeTimes, spikeGid):
    gid_spikes[int(gid)].append(time)

# Read in the 'outsideSpikes.h5' file generated by 'example_network'.
sph5 = h5py.File(READPATH + "outsideSpikes.h5", "r")
spike_trains = \
[ # create a list where spike_trains[gid] returns a list of all firing times
    [ # the first element of each of these lists is the compartment index.
        firing_rates[:] for firing_rates in sph5.get(f'{cell_no}').values()
    ] for cell_no in range(numGids)]
sph5.close()

"""
Now that the previous spiking data is saved in gid_spikes, df1 and df2,
the network can be built again in the same way as it was before, only
this time the network.connect() function will not be called upon.
The synapse class will be set for all connections manually to recreate
the original LFP signal. Be careful with RNG seed-using functions as they
may de-synchronize the script functionality.
"""

# Set up MPI variables as is done in example_network:
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

GLOBALSEED = 1234
np.random.seed(GLOBALSEED + RANK)

# Set up network and electrode parameters as is done in example_network
cellParameters = dict(
    morphology='./init/BallAndStick_passive.hoc',   # use passive leak for recr.
    templatefile='./init/BallAndStickTemplate.hoc',
    templatename='BallAndStickTemplate',
    templateargs=None,
    delete_sections=False,
)

populationParameters = dict(
    Cell=NetworkCell,
    cell_args=cellParameters,
    pop_args=dict(
        radius=100.,
        loc=0.,
        scale=20.),
    rotation_args=dict(x=0., y=0.),
)

networkParameters = dict(
    dt=2**jdict['dt'],
    tstop=jdict['tstop'],
    v_init=-65.,
    celsius=6.5,
    OUTPUTPATH=OUTPUTPATH
)

electrodeParameters = dict(
    x=np.zeros(13),
    y=np.zeros(13),
    z=np.linspace(1000., -200., 13),
    N=np.array([[0., 1., 0.] for _ in range(13)]),
    r=5.,
    n=50,
    sigma=0.3,
    method='soma_as_point'
)

networkSimulationArguments = dict(
    rec_current_dipole_moment=True,
    rec_pop_contributions=True,
    to_memory=True,
    to_file=False
)

""" Fetch the Network characteristics from setup.json """
population_names = jdict['population_names']
population_sizes = jdict['population_sizes']
for i, _ in enumerate(population_sizes):
    population_sizes[i] = int(population_sizes[i] * scale_factor_population)
nidx = int(jdict['nidx']*n_idx_outside_stim) # Number of outside-stimulating synapses per neuron
numCells = int(sum(population_sizes))

if RANK == 0:
    print("Recreating '" + run_name + "' with current-based synapses:" + \
    f"\n\tPopulation names, sizes: {population_names}, {population_sizes}" + \
    f"\n\tPopulation nidx value: {nidx}" + \
    f"\n\tT = {jdict['tstop']}, dt = 2^({jdict['dt']})")

# Declare Synapse characteristics.
synapseParameters = dict(
    E = dict(tau1=0.2, tau2=1.8, e=0.),
    I = dict(tau1=0.1, tau2=9.0, e=-80.)
)

# Calculate the scaling factors for the current-synapse weights here.
scales = dict(
    E = -(jdict["isyn_v_approx"] - synapseParameters['E']['e']),
    I = -(jdict["isyn_v_approx"] - synapseParameters['I']['e'])
)

if __name__ == "__main__":
    # Main simulation contents
    if not os.path.isdir(OUTPUTPATH):
        if RANK == 0:
            os.mkdir(OUTPUTPATH)
    COMM.Barrier()

    network = Network(**networkParameters)

    # Connect the E and I populations to this network
    for name, size in zip(population_names, population_sizes):
        network.create_population(name=name, POP_SIZE=size,
                                **populationParameters)

    # Assess the network pos/rot setup to be equal to the one in example_network
    tf.assert_established_network_pos_rot(df1, network)

    # Set up the outside-stimulating spikes from spike_trains list
    for name in population_names:
        for cell in network.populations[name].cells:
            for i in range(nidx):
                listobj = spike_trains[cell.gid][i]
                syn = Synapse(cell=cell,
                            idx=int(listobj[0]),
                            syntype='Exp2ISyn',
                            weight=0.002*scales['E'],
                            **dict(tau1=0.2, tau2=1.8, e=0.))
                syn.set_spike_times(np.array(listobj[1:]))

    # Assert the outside stimulus to be equal to that of the previous network
    tf.assert_established_network_syn_out(spike_trains, network)

    # Set up the inside-stimulating spikes caused by neighboring neurons.
    for name in population_names:
        for cell in network.populations[name].cells:
            currentCell = df2[df2['gid']==cell.gid]
            rows, columns = currentCell.shape

            # Set up loop for all synapses connected to current cell.
            for _, row in currentCell.iterrows():

                delay  = row['delay']
                weight = row['weight']
                pregid = int(row['pre_gid'])
                idxnum = int(row['idx'])

                """ Need to generalize the synapse parameters argument to be
                dependent on the pre_gid variable. This is done by acknowlegding
                the name attribute of the 'iterrows' Pandas method. The name
                should include something like 'E:E' or 'I:E', and the synaptic
                parameters should be dependent on the pre-synaptic term (i.e.
                'E' for 'E:I' and 'I' for 'I:E'). """
                name = row.name[0]
                pre_type = name[0]

                # Set up the synapse object to emulate this neural connection
                syn = Synapse(cell=cell,
                    idx=idxnum,
                    syntype='Exp2ISyn',
                    weight=weight*scales[pre_type],
                    **synapseParameters[pre_type])

                """ Set spike times according to pre-synaptic firing rates.
                Make sure to include the characteristic synaptic delay. """
                spike_time_list = [i + delay for i in gid_spikes[pregid]]
                syn.set_spike_times(np.array(spike_time_list))

    # Assert network synapses to be equal to that of the previous network.
    tf.assert_established_network_syn_net(df2, network, nidx)

    # Finally set up the extracellular device and start the simulation
    electrode = RecExtElectrode(**electrodeParameters)
    SPIKES, OUTPUT, DIPOLEMOMENT = network.simulate(
        electrode=electrode,
        **networkSimulationArguments
    )

    # collect somatic potentials across all RANKs to RANK 0:
    if RANK == 0:
        somavs = []
        for i, name in enumerate(population_names):
            somavs.append([])
            somavs[i] += [cell.somav
                          for cell in network.populations[name].cells]
            for j in range(1, SIZE):
                somavs[i] += COMM.recv(source=j, tag=15)
    else:
        somavs = None
        for name in population_names:
            COMM.send([cell.somav for cell in network.populations[name].cells],
                      dest=0, tag=15)

    ############################################################################
    # Plot some output on RANK 0 in the same way as is done in example_network #
    ############################################################################
    if RANK == 0:
        # spike raster
        fig, ax = plt.subplots(1, 1)
        for name, spts, gids in zip(population_names, SPIKES['times'], SPIKES['gids']): # 2 times, E and I
            t = []
            g = []
            for spt, gid in zip(spts, gids):    # 100 times. Loops through each GID for current population
                t = np.r_[t, spt]
                g = np.r_[g, np.zeros(spt.size)+gid]
            ax.plot(t[t >= 200], g[t >= 200], '|', label=name)

        ax.legend(loc=1)
        pf.remove_axis_junk(ax, lines=['right', 'top'])
        ax.set_xlabel('t (ms)')
        ax.set_ylabel('gid')
        ax.set_title('spike raster')
        fig.savefig(os.path.join(OUTPUTPATH, 'spike_raster.pdf'),
                    bbox_inches='tight')
        plt.close(fig)

        # somatic potentials
        fig = plt.figure()
        gs = GridSpec(5, 1)
        ax = fig.add_subplot(gs[:4])
        pf.draw_lineplot(ax, pf.decimate(np.array(somavs, dtype=object)[0], q=16), dt=network.dt*16,
                      T=(200, 1200),
                      scaling_factor=1.,
                      vlimround=16,
                      label='E',
                      scalebar=True,
                      unit='mV',
                      ylabels=False,
                      color='C0',
                      ztransform=True
                      )
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_ylabel('E')
        ax.set_title('somatic potentials')
        ax.set_xlabel('')

        ax = fig.add_subplot(gs[4])
        pf.draw_lineplot(ax, pf.decimate(np.array(somavs, dtype=object)[1], q=16), dt=network.dt*16,
                      T=(200, 1200),
                      scaling_factor=1.,
                      vlimround=16,
                      label='I',
                      scalebar=True,
                      unit='mV',
                      ylabels=False,
                      color='C1',
                      ztransform=True
                     )
        ax.set_yticks([])
        ax.set_ylabel('I')

        fig.savefig(os.path.join(OUTPUTPATH, 'soma_potentials.pdf'),
                    bbox_inches='tight')
        plt.close(fig)


        # extracellular potentials, E and I contributions, sum60
        fig, axes = plt.subplots(1, 3, figsize=(6.4, 4.8))
        fig.suptitle('extracellular potentials')
        for i, (ax, name, label) in enumerate(zip(axes, ['E', 'I', 'imem'],
                                                  ['E', 'I', 'sum'])):
            pf.draw_lineplot(ax, pf.decimate(OUTPUT[0][name], q=16), dt=network.dt*16,
                          T=(200, 1200),
                          scaling_factor=1.,
                          vlimround=None,
                          label=label,
                          scalebar=True,
                          unit='mV',
                          ylabels=True if i == 0 else False,
                          color='C{}'.format(i),
                          ztransform=True
                         )
            ax.set_title(label)
        fig.savefig(os.path.join(OUTPUTPATH, 'extracellular_potential.pdf'),
                    bbox_inches='tight')
        plt.close(fig)

        # Save these extracellular potential values to HDF
        dir = "./data/"
        nm = "isyn_signal_lfp.h5"
        hdf_file = h5py.File(dir+nm,'w')
        for k in range(13):
            key = f"ch{k+1}"
            hdf_file.create_dataset(
                key, data=np.array(OUTPUT[0]['imem'][k])
                )
        hdf_file.close()

        # current-dipole moments, E and I contributions, sum
        fig, axes = plt.subplots(3, 3, figsize=(6.4, 4.8))
        fig.subplots_adjust(wspace=0.45)
        fig.suptitle('current-dipole moments')
        for i, u in enumerate(['x', 'y', 'z']):
            for j, label in enumerate(['E', 'I', 'sum']):
                t = np.arange(DIPOLEMOMENT.shape[0])*network.dt
                inds = (t >= 200) & (t <= 1200)
                if label != 'sum':
                    axes[i, j].plot(t[inds][::16],
                                    pf.decimate(DIPOLEMOMENT[label][inds, i],
                                             q=16),
                                    'C{}'.format(j))
                else:
                    axes[i, j].plot(t[inds][::16],
                                    pf.decimate(DIPOLEMOMENT['E'][inds, i] +
                                             DIPOLEMOMENT['I'][inds, i], q=16),
                                    'C{}'.format(j))

                if j == 0:
                    axes[i, j].set_ylabel(r'$\mathbf{p}\cdot\mathbf{e}_{' +
                                          '{}'.format(u) +'}$ (nA$\mu$m)')
                if i == 0:
                    axes[i, j].set_title(label)
                if i != 2:
                    axes[i, j].set_xticklabels([])
                else:
                    axes[i, j].set_xlabel('t (ms)')
        fig.savefig(os.path.join(OUTPUTPATH, 'current_dipole_moment.pdf'),
                    bbox_inches='tight')
        plt.close(fig)


        """ Generate an analysis of the mean- and median somatic potentials
        to potentially be used as scaling factors for the current-based synapses """
        Emean = np.mean(somavs[0], axis=1)
        Imean = np.mean(somavs[1], axis=1)
        Emedi = np.median(somavs[0], axis=1)
        Imedi = np.median(somavs[1], axis=1)

        # somatic potentials
        fig = plt.figure(figsize=(7.4, 5.8))
        gs = GridSpec(2, 1) # (nrows, ncols)
        gs.update(wspace = 0.5, hspace = 0.5)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.hist(Emean, color="#1f77b4", label="E", ec='black')
        ax1.hist(Imean, color="orange", label="I", ec='black')
        ax1.legend()
        ax1.grid()
        ax1.set_xlabel("Mean Soma Potential [mV]")
        ax1.set_ylabel("Count []")

        ax2 = fig.add_subplot(gs[1, :])
        ax2.hist(Emedi, color="#1f77b4", label="E", ec='black')
        ax2.hist(Imedi, color="orange", label="I", ec='black')
        ax2.legend()
        ax2.grid()
        ax2.set_xlabel("Median Soma Potential [mV]")
        ax2.set_ylabel("Count []")

        fig.savefig(os.path.join(OUTPUTPATH, 'soma_mean_median_potentials.pdf'),
                    bbox_inches='tight')
        plt.close(fig)

        """ Save a quick text file describing the simulation characteristics """
        with open(OUTPUTPATH + "sim_chars.txt", "w") as ofile:
            ofile.write("Simulated '" + run_name + "' with:" + \
                f"\n\tPopulation names, sizes: {population_names}, {population_sizes}" + \
                f"\n\tPopulation nidx value: {nidx}" + \
                f"\n\tT = {jdict['tstop']}, dt = 2^({jdict['dt']})")


    # population illustration (per RANK)
    fig = plt.figure(figsize=(6.4, 4.8*2))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=5)
    ax.plot(electrode.x, electrode.y, electrode.z, 'ko', zorder=0)
    for i, (name, pop) in enumerate(network.populations.items()):
        for cell in pop.cells:
            c = 'C0' if name == 'E' else 'C1'
            ax.plot([cell.xstart[0], cell.xend[0]],
                    [cell.ystart[0], cell.yend[0]],
                    [cell.zstart[0], cell.zend[0]], c,
                    lw=5, zorder=-cell.xmid[0]-cell.ymid[0])
            ax.plot([cell.xstart[1], cell.xend[-1]],
                    [cell.ystart[1], cell.yend[-1]],
                    [cell.zstart[1], cell.zend[-1]], c,
                    lw=0.5, zorder=-cell.xmid[0]-cell.ymid[0])
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r'$y$ ($\mu$m)')
    ax.set_zlabel(r'$z$ ($\mu$m)')
    ax.set_title('network populations')
    fig.savefig(os.path.join(OUTPUTPATH, 'population_RANK_{}.pdf'.format(RANK)),
                bbox_inches='tight')
    plt.close(fig)

    ############################################################################
    # customary cleanup of object references - the psection() function may not
    # write correct information if NEURON still has object references in memory,
    # even if Python references has been deleted. It will also allow the script
    # to be run in successive fashion.
    ############################################################################
    network.pc.gid_clear() # allows assigning new gids to threads
    electrode = None
    syn = None
    synapseModel = None
    for population in network.populations.values():
        for cell in population.cells:
            cell = None
        population.cells = None
        population = None
    pop = None
    network = None
    neuron.h('forall delete_section()')
    #neuron.h.topology()
