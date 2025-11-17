#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --partition=normal
#SBATCH --account=mscaisuperpod

module purge
module load Anaconda3/2023.09-0
echo "=== Job Start: $(date) ==="
source activate ms_sd_infer
# 卸载当前版本

python pipeline.py
# nvidia-smi