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
signal_pop = args["signal"]

if run_name == "default_js":    # fetch from setup.json
    run_name = jdict["run_name"]
elif run_name == "default_sf":  # write scale factor into name
    run_name = "run" + str(scale_factor_population)

READPATH = './true_sim/'            # path for read files
OUTPUTPATH = './fake_sim_isyn/'     # path for write files

# Read in some HDF files generated by example_network
f2 = h5py.File(READPATH + 'synapse_positions.h5', "r")

headers2 = ['gid', 'idx', 'weight', 'delay', 'pre_gid']
f2E_E = pd.DataFrame(np.array(f2.get('E:E')), columns = headers2)
f2E_I = pd.DataFrame(np.array(f2.get('E:I')), columns = headers2)
f2I_E = pd.DataFrame(np.array(f2.get('I:E')), columns = headers2)
f2I_I = pd.DataFrame(np.array(f2.get('I:I')), columns = headers2)
df2 = pd.concat({'E:E':f2E_E,'E:I':f2E_I,'I:E':f2I_E,'I:I':f2I_I})

f2.close()

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
    tstop=jdict["stim_sim_len"],
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
    f"\n\tT = {jdict['stim_sim_len']}, dt = 2^({jdict['dt']})")

# Declare Synapse characteristics.
synapseParameters = dict(
    E = dict(tau1=0.2, tau2=1.8, e=0.),
    I = dict(tau1=0.1, tau2=9.0, e=-80.)
)

populationDict = dict(
    E = population_sizes[0],
    I = population_sizes[1]
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

    # Dont need to set up the outside-stimulating spikes from spike_trains list

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

                # Need to make a input_pop variable which cycles through the
                # populations E:E, E:I, etc. independently.

                """ TODO: Add the delays to the ends. Stim at sync time """
                if str(signal_pop) == name:
                    spike_time_list = [jdict["stim_sim_time"] + delay]
                else:
                    spike_time_list = []
                syn.set_spike_times(np.array(spike_time_list))


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
        # save these LFPs into file to generate later plots.
        # only do the 'signal_pop' population though. Do all channels.
        dir = "./data/"
        nm = f"kernel_{signal_pop}.h5"
        hdf_file = h5py.File(dir+nm,'w')
        norm_val = populationDict[str(signal_pop)[0]]

        for k in range(13):
            key = f"ch{k+1}"
            hdf_file.create_dataset(
                key, data=np.array(OUTPUT[0]['imem'][k])/norm_val
                )
        hdf_file.close()


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
