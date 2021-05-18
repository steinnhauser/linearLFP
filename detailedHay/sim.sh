#!/bin/bash

# Bash script to run the g, r, i scripts for all populations E:E, E:I, etc.
declare -a arr=("E:E" "E:I" "I:E" "I:I")

mpirun python3 ./01_kernel_model_setup.py -p $1 -i $2 -n run0 -s 0
for i in "${arr[@]}"
do
  # same synapses but set only population i to fire at certain times
  mpirun python3 ./02_generate_kernels.py -p $1 -i $2 -n $i -s $i
done

# conduct the true simulation with active conductances
mpirun python3 ./03_simulate_population.py -p $1 -i $2 -n $3

mpirun python3 ./04_recreate_isyn.py -p $1 -i $2 -n $3

# Kernel approximation. Not parallelized for the time being
python3 ./05_recreate_kernels.py -p $1 -i $2 -n $3

mpirun python3 ./06_recreate_gsyn.py -p $1 -i $2 -n $3
# python3 visualize_data.py
