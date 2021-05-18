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

# import custom modules
import utils.plotter as pf
import utils.misc as mf

# Load in the .json setup file.
jfile = open("./init/setup.json", "r", encoding="utf-8")
jdict = json.load(jfile)
jfile.close()

args = vars(mf.argument_parser())
scale_factor_population = args["scale_factor"]
run_name = args["run_name"]
signal_pop = args["signal"]
custom_group_index = args["custom_group_index"]

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

# class NetworkCell parameters:
cellParameters = dict(
    morphology='./init/BallAndStick_active.hoc',    # use hh for generation.
    templatefile='./init/BallAndStickTemplate.hoc',
    templatename='BallAndStickTemplate',
    templateargs=None,
    delete_sections=False,
)

# class NetworkPopulation parameters:
populationParameters = dict(
    Cell=NetworkCell,
    cell_args=cellParameters,
    pop_args=dict(
        radius=100.,
        loc=0.,
        scale=20.),
    rotation_args=dict(x=0., y=0.),
)

# class Network parameters:
networkParameters = dict(
    dt=2**jdict['dt'],
    tstop=jdict["stim_sim_len"],
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
# scale the populations according to the scale factor.
for i, _ in enumerate(population_sizes):
    population_sizes[i] = int(population_sizes[i]*scale_factor_population)

numCells = int(sum(population_sizes))

save_stim_synapses = True               # boolean to save network synapses
connectionProbability = [[0.1, 0.1], [0.1, 0.1]]

if RANK == 0:
    print("Initializing '" + run_name + "' with:" + \
    f"\n\tPopulation names, sizes: {population_names}, {population_sizes}" + \
    f"\n\tWeigts: E={args['custom_e_weight']}, I={args['custom_i_weight']}")

# description which is saved to network output
desc = ""


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
synapsePositionArguments = [[dict(section=['soma', 'apic'],
                                  fun=[st.norm, st.norm],
                                  funargs=[dict(loc=500., scale=100.),
                                           dict(loc=500., scale=100.)],
                                  funweights=[0.5, 1.]
                                 ) for _ in range(2)],
                            [dict(section=['soma', 'apic'],
                                  fun=[st.norm, st.norm],
                                  funargs=[dict(loc=0., scale=100.),
                                           dict(loc=0., scale=100.)],
                                  funweights=[1., 0.5]
                                 ) for _ in range(2)]]

if __name__ == '__main__':
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
            **populationParameters
        )

    # create connectivity matrices and connect populations
    for i, pre in enumerate(population_names):
        for j, post in enumerate(population_names):
            connectivity = network.get_connectivity_rand(
                pre=pre, post=post,
                connprob=connectionProbability[i][j]
            )

            # connect network and save connections
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
