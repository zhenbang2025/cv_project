#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

module load Anaconda3/2023.09-0
source activate ms_sd_infer

which python
conda info --envs
echo "CONDA_PREFIX: $CONDA_PREFIX"