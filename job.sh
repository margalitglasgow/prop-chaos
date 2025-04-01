#!/bin/bash

#SBATCH -p mit_normal
#SBATCH --mem=32G

source ./bin/activate
mkdir results/"$1"
mkdir results/"$1"/data

# BIG M JOBS

#python3 main_nn.py -d 64 -mb 512 -ms 7 -t 400 -p "$1" -l 0.25 # For Hop_1_3
python3 main_nn.py -d 32 -mb 512 -ms 7 -t 32 -p "$1" -l 0.005 # For He4_misspecfied and Man_big_2
# python3 main_nn.py -d 32 -mb 512 -ms 7 -t 64 -p "$1" -l 0.01 # For He4

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

#python3 plotting.py -p "$1" -mb 128 -ms 3 -t 8 -k "$2"

# For gaussian problems I have been running with LR 0.01 and Times 16 16 32 64
# For boolean problems I have been running with LR 0.25 and Times 100 400 1600 3200
# Most problem require 4G memory