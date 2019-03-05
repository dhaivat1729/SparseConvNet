#!/bin/bash
#SBATCH --account=def-bengioy
#SBATCH --gres=gpu:1		# request GPU "generic resource"
#SBATCH --cpus-per-task=1	# maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham
#SBATCH --mem=8000M		# memory per node
#SBATCH --time=0-06:00		# time (DD-HH:MM)
#SBATCH --output=%N-%j.out	# %N for node name, %j for jobID
#SBATCH --requeue
#SBATCH --begin=now+60
#SBATCH --mail-user=dhaivat1994@gmail.com
#SBATCH --mail-type=ALL


module load miniconda3
module load python/3.6
source activate my_pytorch

echo Running on $HOSTNAME

cd ~/SparseConvNet/examples/road_segmentation
python small_unet.py
