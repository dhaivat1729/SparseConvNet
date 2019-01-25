#!/usr/bin/env bash

# Source bashrc
#source $HOME/.bashrc

# Activate the environment
module load miniconda3
module load cuda
module load python/3.6
source activate my_pytorch

# Run the script
sbatch --time=6:0:0 --ntasks=1 --account=def-bengioy --nodes=1 --mem=8000M python unet2.py
