#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

module purge
module load Anaconda3/2023.09-0

source activate ms_sd_infer
pip install 'accelerate>=0.26.0'
# 卸载当前版本

python dual_vlm_sd.py
# nvidia-smi