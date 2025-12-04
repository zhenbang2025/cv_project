#!/bin/bash
#SBATCH --job-name=msdiff_inference  # Create a short name for your job
#SBATCH --output=logs/msdiff_output_%j.log  # Log output file, saved in the logs directory (%j is the job ID)
#SBATCH --error=logs/msdiff_error_%j.log   # Error log file
#SBATCH --nodes=1                # Number of nodes
#SBATCH --gpus=1                 # Number of GPUs per node (only valid for large/normal partitions)
#SBATCH --time=00:20:00         # Total run time limit (HH:MM:SS) (2 hours, sufficient for one inference)
#SBATCH --partition=normal  # Partition (large/normal/cpu) to submit to
#SBATCH --account=mscaisuperpod      # Required only for multiple projects

# Navigate to the project directory
cd /home/hxiaoap/MS-Diffusion

# Load environment
module purge                     # Clear inherited environment modules
module load Anaconda3/2023.09-0  # Load the required modules
echo "Initializing and activating Conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate msdiff
echo "当前激活的 Conda 环境：$(conda info --envs | grep '*' | awk '{print $1}')"
echo "当前使用的 Python 路径：$(which python)"
echo "当前 Python 版本：$(python --version)"
pip install diffusers==0.35.2 transformers accelerate
pip install -q ultralytics
pip install controlnet_aux
pip install gradio
pip uninstall numpy -y
pip install numpy==1.24.4 scikit-image

echo ""
echo "=== Checking installed packages ==="
python - <<'PYCODE'
import torch, diffusers, transformers, accelerate
print(f"torch version        : {torch.__version__}")
print(f"diffusers version    : {diffusers.__version__}")
print(f"transformers version : {transformers.__version__}")
print(f"accelerate version   : {accelerate.__version__}")
PYCODE
echo "==================================="

# Prepare log directory
mkdir -p logs

# Execute inference command
echo "Starting inference with image input..."
python ./exp_controlnet_refine.py


if [ $? -eq 0 ]; then
  # Output on success
  echo "Inference task completed."
else
  # Output on failure
  echo "Inference task failed! Check the error log: cat logs/msdiff_error_"$SLURM_JOB_ID".log"
fi