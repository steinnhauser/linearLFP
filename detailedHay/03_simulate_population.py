#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Demonstrate usage of LFPy.Network with network of ball-and-stick type
morphologies with active HH channels inserted in the somas and passive-leak
channels distributed throughout the apical dendrite. The corresponding
morphology and template specifications are in the files BallAndStick.hoc and
BallAndStickTemplate.hoc.

Execution (w. MPI):
    mpirun --use-hwthread-cpus -np 4 python3 generate_LFP.py

Copyright (C) 2017 Computational Neuroscience Group, NMBU.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

"""
# import modules:
import os, sys
import matplotlib.pyplot as plt
import matplotlib.cbook as cbk
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.signal as ss
import scipy.stats as st
from mpi4py import MPI
import neuron
from LFPy import NetworkCell, Network, Synapse, RecExtElectrode
from LFPy.inputgenerators import get_activation_times_from_distribution
import json
import h5py

# import custom modules
import utils.plotter as pf
import utils.misc as mf

# Load in the .json setup file.
jfile = open("./init/setup.json", "r", encoding="utf-8")
jdict = json.load(jfile)
jfile.close()

args = vars(mf.argument_parser())
n_idx_outside_stim = args["n_idx"]
scale_factor_population = args["scale_factor"]
save_spike_trains = args["save_spike_trains"]
save_lfp_signals = args["save_lfp_signals"]
custom_group_index = args["custom_group_index"]

run_name = args["run_name"]
if run_name == "default_js":    # fetch from setup.json
    run_name = jdict["run_name"]
elif run_name == "default_sf":  # write scale factor into name
    run_name = "run" + str(scale_factor_population)

# set up MPI variables:
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

# avoid same sequence of random numbers from numpy and neuron on each RANK,
# e.g., in order to draw unique cell and synapse locations and random synapse
# activation times
GLOBALSEED = 1234
np.random.seed(GLOBALSEED + RANK)

################################################################################
# Set up shared and population-specific parameters
################################################################################
# relative path for simulation output:
OUTPUTPATH = './true_sim/'

neuron.load_mechanisms('L5bPCmodelsEH/mod/')
cellParameters = {
    'E' : { # excitatory cells are modeled using the Hay model
        'morphology'    : 'L5bPCmodelsEH/morphologies/cell1.asc',
        'templatefile'  : ['L5bPCmodelsEH/models/L5PCbiophys3.hoc',
                           'L5bPCmodelsEH/models/L5PCtemplate.hoc'],
        'templatename'  : 'L5PCtemplate',
        'templateargs'  : 'L5bPCmodelsEH/morphologies/cell1.asc',
        'passive' : False,
        'nsegs_method' : None,
        'pt3d' : True,
        'delete_sections' : False,
    },
    'N' : { # inhibitory cells are modeled as ball-and-stick for now.
        'morphology' : './init/BallAndStick_active_shortened.hoc',    # use hh for generation.
        'templatefile' : './init/BallAndStickTemplate.hoc',
        'templatename' : 'BallAndStickTemplate',
        'templateargs' : None,
        'delete_sections' : False,
    },
    'I' : { # inhibitory cells are modeled as ball-and-stick for now.
        'morphology' : './init/BallAndStick_active_shortened.hoc',    # use hh for generation.
        'templatefile' : './init/BallAndStickTemplate.hoc',
        'templatename' : 'BallAndStickTemplate',
        'templateargs' : None,
        'delete_sections' : False,
    }
}

# class NetworkPopulation parameters:
populationParameters = {
    'E' : dict(
        Cell=NetworkCell,
        pop_args=dict(
            radius=100.,
            loc=0.,
            scale=20.),
        rotation_args=dict(x=4.729, y=-3.166), # Hay-cell specific rotations
    ),
    'I' : dict(
        Cell=NetworkCell,
        pop_args=dict(
            radius=100.,
            loc=0.,
            scale=20.),
        rotation_args=dict(x=0., y=0.),
    )
}

# class Network parameters:
networkParameters = dict(
    dt=2**jdict['dt'],
    tstop=jdict['tstop'],
    v_init=-65.,
    celsius=6.5,
    OUTPUTPATH=OUTPUTPATH,
)

# class RecExtElectrode parameters:
electrodeParameters = dict(
    x=np.zeros(13),
    y=np.zeros(13),
    z=np.linspace(1000., -200., 13),
    N=np.array([[0., 1., 0.] for _ in range(13)]),
    r=5.,
    n=50,
    sigma=0.3,
    method="soma_as_point"
)

# method Network.simulate() parameters:
networkSimulationArguments = dict(
    rec_current_dipole_moment=True,
    rec_pop_contributions=True,
    to_memory=True,
    to_file=False,
)

# n_idx_outside_stim
# scale_factor_population
""" Fetch the Network characteristics from setup.json """
population_names = jdict['population_names']
population_sizes = jdict['population_sizes']
for i, _ in enumerate(population_sizes):
    population_sizes[i] = int(population_sizes[i]*scale_factor_population)
nidx = int(jdict['nidx']*n_idx_outside_stim) #outside-stimulating synapses per neuron
save_stim_synapses = True               # boolean to save network synapses
numCells = int(sum(population_sizes))
connectionProbability = [[0.1, 0.1], [0.1, 0.1]]
additional_neuron_stim = [float(i) for i in jdict['additional_neuron_stim']]

if RANK == 0:
    print("Simulating '" + run_name + "' with:" + \
    f"\n\tPopulation names, sizes: {population_names}, {population_sizes}" + \
    f"\n\tnidx value: {nidx}, Scale factors {additional_neuron_stim}" + \
    f"\n\tWeigts: E={args['custom_e_weight']}, I={args['custom_i_weight']}" + \
    f"\n\tT = {jdict['tstop']}, dt = 2^({jdict['dt']})")

# description which is saved to network output
desc = "Regular parameter configuration."

# synapse model. All corresponding parameters for weights,
# connection delays, multapses and layerwise positions are
# set up as shape (2, 2) nested lists for each possible
# connection on the form:
# [["E:E", "E:I"],
#  ["I:E", "I:I"]].
synapseModel = neuron.h.Exp2Syn
synapseParameters = [[dict(tau1=0.2, tau2=1.8, e=0.),
                      dict(tau1=0.2, tau2=1.8, e=0.)],
                     [dict(tau1=0.1, tau2=9.0, e=-80.),
                      dict(tau1=0.1, tau2=9.0, e=-80.)]]

# synapse max. conductance (function, mean, st.dev., min.):
# try dividing these weights by the population scale to renormalize
weightFunction = np.random.normal
excitatory = float(args['custom_e_weight'])/scale_factor_population
inhibitory = float(args['custom_i_weight'])/scale_factor_population
# vary the scale of the distribution by a tenth of the actual mean
weightArguments = [[dict(loc=excitatory, scale=excitatory/10.),
                    dict(loc=excitatory, scale=excitatory/10.)],
                   [dict(loc=inhibitory, scale=inhibitory/10.),
                    dict(loc=inhibitory, scale=inhibitory/10.)]]
minweight = 0.

# conduction delay (function, mean, st.dev., min.):
delayFunction = np.random.normal
delayArguments = [[dict(loc=1.5, scale=0.3),
                   dict(loc=1.5, scale=0.3)],
                  [dict(loc=1.5, scale=0.3),
                   dict(loc=1.5, scale=0.3)]]
mindelay = 0.3
multapseFunction = np.random.normal
multapseArguments = [[dict(loc=2., scale=.5), dict(loc=2., scale=.5)],
                     [dict(loc=5., scale=1.), dict(loc=5., scale=1.)]]

# method NetworkCell.get_rand_idx_area_and_distribution_norm
# parameters for layerwise synapse positions:
# synapsePositionArguments = [[dict(section=['soma', 'apic'],
#                                   fun=[st.norm, st.norm],
#                                   funargs=[dict(loc=500., scale=100.),
#                                            dict(loc=500., scale=100.)],
#                                   funweights=[0.5, 1.]
#                                  ) for _ in range(2)],
#                             [dict(section=['soma', 'apic'],
#                                   fun=[st.norm, st.norm],
#                                   funargs=[dict(loc=0., scale=100.),
#                                            dict(loc=0., scale=100.)],
#                                   funweights=[1., 0.5]
#                                  ) for _ in range(2)]]

# e_e should be how Hay cells connect to them selves. Mostly Dendritic (apic).
e_e = dict(section=['dend', 'apic'], fun=[st.norm, st.norm], funargs=[ \
    dict(loc=0., scale=100.), dict(loc=500., scale=100.)], funweights=[1., 1.])

# e_i should be how Hay cells connect to ball. Should be some dend but mostly apic
e_i = dict(section=['dend'], fun=[st.norm], funargs=[ \
    dict(loc=100., scale=100.)], funweights=[1.])

# i_e connects to the Hay cell somatic region.
i_e = dict(section=['soma', 'dend'], fun=[st.norm], funargs=[ \
    dict(loc=0., scale=100.)], funweights=[1.])

# i_i connects to its own somatic and dendritic region.
i_i = dict(section=['soma', 'dend'], fun=[st.norm], funargs=[ \
    dict(loc=0., scale=100.)], funweights=[1.])

synapsePositionArguments = [[e_e, e_i],
                            [i_e, i_i]]

"""
TODO:
    - Figure which double checks that these are initialized regularly.
    - Rename apic to dend in .hoc file of inhibitory.
"""

if __name__ == '__main__':
    ############################################################################
    ############################## Main simulation #############################
    ############################################################################

    # create directory for output:
    if not os.path.isdir(OUTPUTPATH):
        if RANK == 0:
            os.mkdir(OUTPUTPATH)
    COMM.Barrier()

    # create directory for data:
    if not os.path.isdir("./data/"):
        if RANK == 0:
            os.mkdir("./data/")
    COMM.Barrier()

    # instantiate Network:
    network = Network(**networkParameters)

    # create E and I populations:
    for name, size in zip(population_names, population_sizes):
        network.create_population(
            name=name,
            POP_SIZE=size,
            cell_args=cellParameters[name], # insert the difference between E and I cell models.
            **populationParameters[name]
        )

    # create connectivity matrices and connect populations:
    for i, pre in enumerate(population_names):
        for j, post in enumerate(population_names):
            # boolean connectivity matrix between pre- and post-synaptic neurons
            # in each population (postsynaptic on this RANK)
            connectivity = network.get_connectivity_rand(
                pre=pre, post=post,
                connprob=connectionProbability[i][j]
                )

            # connect network:
            (conncount, syncount) = network.connect(
                pre=pre, post=post,
                connectivity=connectivity,
                syntype=synapseModel,
                synparams=synapseParameters[i][j],
                weightfun=weightFunction,
                weightargs=weightArguments[i][j],
                minweight=minweight,
                delayfun=delayFunction,
                delayargs=delayArguments[i][j],
                mindelay=mindelay,
                multapsefun=multapseFunction,
                multapseargs=multapseArguments[i][j],
                syn_pos_args=synapsePositionArguments[i][j],
                save_connections=True, # Add a 'save network' arg
                )

    # Set up numpy array to save all outside-stimulating synapses
    saved_outside_input_per_pop = { # make customizable for different nidx per population.
        'E' : np.zeros((int(population_sizes[0]), int(nidx*additional_neuron_stim[0])), dtype=list),
        'I' : np.zeros((int(population_sizes[1]), int(nidx*additional_neuron_stim[1])), dtype=list),
    }
    saved_outside_input = np.zeros((numCells, nidx), dtype=list)

    fac = { # misc factor for increasing stimulus to Hay cell
        'E' : 1,
        'I' : 1
    }

    # https://www.humanbrainproject.eu/en/follow-hbp/news/hbp-researchers-achieve-breakthrough-in-modelling-nerve-cells/

    for id, (name, size) in enumerate(zip(population_names, population_sizes)):
        # create excitatory background synaptic activity for each cell
        # with Poisson statistics
        for numcell, cell in enumerate(network.populations[name].cells):
            idx = cell.get_rand_idx_area_norm(section='allsec',
                nidx=int(nidx*additional_neuron_stim[id])) # make sure to pick out correct amount of nidx
            for n, i in enumerate(idx):
                syn = Synapse(
                            cell=cell,
                            idx=i,
                            syntype='Exp2Syn',
                            weight=0.002,
                            **dict(tau1=0.2, tau2=1.8, e=0.))

                # Generate a poisson distribution to be set and saved.
                times = get_activation_times_from_distribution(n=1, tstart=0.,
                        tstop=jdict['tstop'], distribution=st.expon,
                        rvs_args=dict(loc=0., scale=100.))[0]
                syn.set_spike_times(np.array(times))

                # Add these to the saved outside input list.
                saved_outside_input_per_pop[name][numcell, n] = [i, times]

    if RANK == 0:
        # establish the file for writing to.
        nm = "outsideSpikes.h5"
        hdf_file = h5py.File(OUTPUTPATH+nm,'w')


    # loop through each population, processing as before and saving to outer pop group.
    for ct, pop in enumerate(population_names):
        saved_outside_input = saved_outside_input_per_pop[pop]
        pop_size = population_sizes[ct]

        if save_stim_synapses:
            # remove the large sections of zeros caused by parallelization
            del_idcs = []
            for cell in range(pop_size):
                if all(saved_outside_input[cell,:] == 0):
                    del_idcs.append(cell)
            useful_items = np.delete(saved_outside_input, del_idcs, axis=0)

            # Gather all the values into one final list before saving in RANK 0
            master_list = []
            master_list = COMM.gather(useful_items, root=0)

            if RANK == 0:
                # Need to flatten the list. i.e [[1,2],[3,4]] -> [1,2,3,4]
                flat_list = [item for sublist in master_list for item in sublist]

                # Problem: The list should have a chronology following GID.
                # for numCells = 10 and SIZE = 4, the list has chronology
                # mixed list = [0, 4, 8, 1, 5, 9, 2, 6, 3, 7]
                # Generate an arbitrary list like this for any given (SIZE, numCells)
                mixedList = []
                ctr = 0; coreNo = 0
                while len(mixedList) < pop_size:
                    if ctr < pop_size:
                        # Counter can go up to 9 in case numCells = 10
                        mixedList.append(ctr)
                        ctr+=SIZE
                    else:
                        # If the counter passes over numCells, move to next core.
                        coreNo+=1
                        ctr=coreNo

                # Now to adjust the flat_list to have the right chronology.
                #This is done by sorting flat_list with respect to mixedList
                right_list = [i for _,i in sorted(zip(mixedList, flat_list))]

                ###### write to the h5 file. establish group for pop first.
                grp1 = hdf_file.create_group(f'{pop}')
                for i in range(pop_size):
                    grp2 = grp1.create_group(f'{i}')
                    # array to contain the values with shape nidx*pop_scale
                    nidx_pop = saved_outside_input.shape[1]
                    vals = np.zeros(nidx_pop, dtype=object)
                    for j in range(nidx_pop):
                        lst = right_list[i][j][1].tolist()
                        idx = int(right_list[i][j][0])
                        lst.insert(0,idx)
                        grp2.create_dataset(f"{j}",data=lst)

    if RANK==0:
        hdf_file.close()

    # set up extracellular recording device:
    electrode = RecExtElectrode(**electrodeParameters)

    # run simulation:
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
    # Plot some output on RANK 0
    ############################################################################
    if RANK == 0:
        # spike raster
        fig, ax = plt.subplots(1, 1)
        totSpikes = []
        totGids = []

        ordered_populations_totSpikes = {}
        ordered_populations_totGids = {}
        for name, spts, gids in zip(population_names, SPIKES['times'], SPIKES['gids']): # 2 times, E and I
            t = []
            g = []
            for spt, gid in zip(spts, gids):    # 100 times. Loops through each GID for current population
                t = np.r_[t, spt]
                g = np.r_[g, np.zeros(spt.size)+gid]
            ax.plot(t, g, ',', label=name)

            totSpikes = np.r_[totSpikes, t]
            totGids = np.r_[totGids, g]

            # save in a 'population ordered' fashion for kernel method.
            ordered_populations_totSpikes[name] = t
            ordered_populations_totGids[name] = g

        ax.legend(loc=1)
        pf.remove_axis_junk(ax, lines=['right', 'top'])
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel('Gid []')
        ax.set_title('spike raster')
        fig.savefig(os.path.join("./figures/", 'spike_raster.pdf'),
                    bbox_inches='tight')
        plt.close(fig)


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

        fig.savefig(os.path.join("./figures/", 'soma_mean_median_potentials_real.pdf'),
                    bbox_inches='tight')
        plt.close(fig)


        """
        TODO:
            Use these spike times and plots and assign each spike to the according GID and time and save to a file.
            totSpikes : List of all spike times
            totGids : List of all the GIDS corresponding with the spikes.
            Transform these two lists into one including both the pieces of information.
        """

        if save_lfp_signals:
            # if we're in the lfp signal per nidx scheme, append to a data analysis frame
            dir = "./analysis/prepdata/"
            nm = "lfp_signal_real.h5"
            hdf_file = h5py.File(dir+nm, 'a') # append to already existing list

            if custom_group_index == -1:
                grp = hdf_file.create_group(f"nidx_{nidx}")
            else: # if the study does not change nidx value, need to save as custom.
                grp = hdf_file.create_group(f'{custom_group_index}')

            for k in range(13):
                key = f"ch{k+1}"
                grp.create_dataset(
                    key, data=np.array(OUTPUT[0]['imem'][k])
                    )
            hdf_file.close()


        if save_spike_trains:
            # indicates that this is an analysis for simply spike train analysis purposes. Skip rest of things

            """ Using the custom save_spike_trains bool input, we add the spike train
            information into a separate directory for later analysis """
            dir = "./analysis/prepdata/"
            nm = "firing_rates.h5"
            hdf_file = h5py.File(dir+nm, 'a') # append to already existing list


            if custom_group_index == -1:
                grp = hdf_file.create_group(f"nidx_{nidx}")
            else: # if the study does not change nidx value, need to save as custom.
                grp = hdf_file.create_group(f'{custom_group_index}')

            grp.create_dataset("spikeTimes", data=totSpikes)
            grp.create_dataset("spikeGid", data=totGids)

            hdf_file.close()

        # save these instead into a H5 file, one key is 'spikeTimes' and another is 'spikeGid'.
        nm = "savedSpikes.h5"
        hdf_file = h5py.File(OUTPUTPATH + nm,'w')
        for pop_name in population_names:
            grp = hdf_file.create_group(f"{pop_name}")
            grp.create_dataset("spikeTimes",
                data=ordered_populations_totSpikes[pop_name])
            grp.create_dataset("spikeGid",
                data=ordered_populations_totGids[pop_name])
        hdf_file.close()

        # save the extracellular potential, overwriting.
        dir = "./data/"
        nm = "real_signal_lfp.h5"
        hdf_file = h5py.File(dir+nm,'w')
        for k in range(13):
            key = f"ch{k+1}"
            hdf_file.create_dataset(
                key, data=np.array(OUTPUT[0]['imem'][k]))
        hdf_file.close()

        # save the soma potential as well.
        nm = "soma_potentials.h5"
        hdf_file = h5py.File(OUTPUTPATH + nm,'w')

        hdf_file.create_dataset('E', data=np.array(somavs[0]))
        hdf_file.create_dataset('I', data=np.array(somavs[1]))
        hdf_file.close()

        """ Save a quick text file describing the simulation characteristics """
        with open(OUTPUTPATH + "sim_chars.txt", "w") as ofile:
            ofile.write("Simulated '" + run_name + "' with:" + \
                f"\n\tPopulation names, sizes: {population_names}, {population_sizes}" + \
                f"\n\tPopulation nidx value: {nidx}" + \
                f"\n\tT = {jdict['tstop']}, dt = 2^({jdict['dt']})" + \
                f"\n\tOther description/comments: {desc}")





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