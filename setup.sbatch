#!/bin/bash
#SBATCH --job-name=test          # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --gpus=1                 # number of GPUs per node(only valid under large/normal partition)
#SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=normal           # partition(large/normal/cpu) where you submit
#SBATCH --account=mscaisuperpod      # only require for multiple projects

srun --pty bash
module purge
module load Anaconda3/2023.09-0
conda create -n ms_sd_infer python=3.9 -y
# 激活新环境
conda activate ms_sd_infer
pip install -r requirements.txt