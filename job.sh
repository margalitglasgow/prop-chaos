#!/bin/bash

#SBATCH -p mit_normal
#SBATCH --mem=64G

source ./bin/activate
mkdir results/"$2"
mkdir results/"$2"/data
mkdir results/"$2"/plotdata

# Example code
# python3 main_nn.py -d 32 -mb 128 -ms 3 -t 64 -p "$1" -l 0.1 -a "$2" -j $SLURM_JOB_ID # He4 test
# python3 cleanup.py -d 32 -mb 128 -ms 3 -p "$1" -a "$2" # He4 test
# python3 plotting.py -p "$1" -mb 128 -ms 3 -t 64 -a "$2"


# Experimental Jobs
# mkdir results/XOR_4_lpath
# mkdir results/XOR_4_lpath/data
# mkdir results/XOR_4_lang
# mkdir results/XOR_4_lang/data
# python3 main_nn.py -d 32 -mb 512 -ms 7 -t 400 -p XOR_4 -l 0.25 -v 0.001 -a XOR_4_lang # For Hop_1_3, XOR_4 langevin
#python3 main_nn.py -d 32 -mb 512 -ms 7 -t 400 -p XOR_4 -l 0.25 -r 0.1 -a XOR_4_lpath # For Hop_1_3, XOR_4 + lpath

# BIG M JOBS

#python3 main_nn.py -d 128 -mb 16384 -ms 2 -t 600 -p "$1" -l 0.25 -a "$2" -j $SLURM_JOB_ID # For Hop_1_3, XOR_4
#python3 main_nn.py -d 32 -mb 32768 -ms 1 -t 600 -p "$1" -l 0.25 -a "$2" -j $SLURM_JOB_ID # For XOR_4
#python3 main_nn.py -d 32 -mb 16384 -ms 1 -t 32 -p "$1" -l 0.005 -a "$2" -j $SLURM_JOB_ID # For He4_misspecfied and Man_big_2
python3 main_nn.py -d 32 -mb 32768 -ms 1 -t 64 -p "$1" -l 0.02 -a "$2" -j $SLURM_JOB_ID # For He4, Ge4

# Smal M JOBS

# He4, He4_orth, He4_nonorth, Man_2, He4_random_8_8
# python3 main_nn.py -d 16 -mb 128 -ms 3 -t 64 -p "$1" -l 0.01
# python3 main_nn.py -d 32 -mb 128 -ms 3 -t 64 -p "$1" -l 0.01
# python3 main_nn.py -d 64 -mb 128 -ms 3 -t 64 -p "$1" -l 0.01
# python3 main_nn.py -d 128 -mb 128 -ms 3 -t 64 -p "$1" -l 0.01

# He4_misspecfied and Man_big_2
# python3 main_nn.py -d 16 -mb 128 -ms 3 -t 64 -p "$1" -l 0.01
# python3 main_nn.py -d 32 -mb 128 -ms 3 -t 64 -p "$1" -l 0.005
# python3 main_nn.py -d 64 -mb 128 -ms 3 -t 64 -p "$1" -l 0.0025
# python3 main_nn.py -d 128 -mb 128 -ms 3 -t 64 -p "$1" -l 0.0025 # Too unstable for He4_misspecfied

# Boolean problems (may want to run for longer for XOR_4)
# python3 main_nn.py -d 16 -mb 128 -ms 3 -t 800 -p "$1" -l 0.25
# python3 main_nn.py -d 32 -mb 128 -ms 3 -t 800 -p "$1" -l 0.25
# python3 main_nn.py -d 64 -mb 128 -ms 3 -t 800 -p "$1" -l 0.25
# python3 main_nn.py -d 128 -mb 128 -ms 3 -t 800 -p "$1" -l 0.25

# python3 main_nn.py -d 32 -mb 512 -ms 7 -t 200 -p "$1" -l 0.1 # This was final XOR_bigg & requires 20G of memory

# For gaussian problems I have been running with LR 0.01 and Times 16 16 32 64
# For boolean problems I have been running with LR 0.25 and Times 100 400 1600 3200
# Most problem require 4G memory