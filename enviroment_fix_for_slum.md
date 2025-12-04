# 下载 ms-diffusion 模型


```sh
#!/bin/bash

# ==================== 配置区 ====================
# 设置您要存放所有模型的根目录
MODEL_ROOT="/home/hxiaoap/model"
# ===============================================

# 设置脚本在遇到错误时立即退出
set -e

# --- 步骤 0: 清理旧目录 (带安全确认) ---
echo "=============================================="
echo "步骤 0: 清理旧模型目录"
echo "=============================================="
if [ -d "$MODEL_ROOT" ]; then
    echo "⚠️  警告：即将删除整个目录及其所有内容: $MODEL_ROOT"
    # -n 1: 只读取一个字符; -r: 禁止反斜杠转义; -p: 显示提示
    read -p "您确定要继续吗？ (y/N) " -n 1 -r
    echo "" # 换行
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在删除旧目录..."
        rm -rf "$MODEL_ROOT"
        echo "✅ 旧目录已删除。"
    else
        echo "操作已取消，脚本退出。"
        exit 0
    fi
else
    echo "✅ 目标目录 '$MODEL_ROOT' 不存在，无需清理。"
fi
echo ""


# --- 步骤 1: 检查依赖 huggingface-cli ---
echo "=============================================="
echo "步骤 1: 检查依赖项"
echo "=============================================="
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ 错误: 'huggingface-cli' 命令未找到。"
    echo "请先安装它，通常使用以下命令:"
    echo "pip install -U \"huggingface_hub[cli]\""
    exit 1
fi
echo "✅ 依赖 'huggingface-cli' 已满足。"
echo ""


# --- 步骤 2: 创建目录结构 ---
echo "=============================================="
echo "步骤 2: 正在创建目录结构..."
echo "=============================================="
# 虽然 huggingface-cli 会自动创建目标目录，但为了结构清晰，我们先创建父目录
mkdir -p "$MODEL_ROOT/stabilityai"
mkdir -p "$MODEL_ROOT/laion"
mkdir -p "$MODEL_ROOT/MS-Diffusion"
echo "✅ 目录结构已在 '$MODEL_ROOT' 中准备就绪。"
echo ""


# --- 步骤 3: 下载 Stable Diffusion XL Base 1.0 ---
echo "=============================================="
echo "步骤 3: 正在处理 Stable Diffusion XL Base 1.0"
echo "=============================================="
SDXL_REPO_ID="stabilityai/stable-diffusion-xl-base-1.0"
SDXL_PATH="$MODEL_ROOT/stabilityai/stable-diffusion-xl-base-1.0"

if [ -f "$SDXL_PATH/model_index.json" ]; then
    echo "✅ 模型已存在于: $SDXL_PATH。跳过下载。"
else
    echo "模型不存在，开始使用 huggingface-cli 下载到: $SDXL_PATH"
    echo "（这可能需要一些时间，但会比 git clone 快很多...）"
    # --local-dir-use-symlinks False 确保文件被直接下载到目标目录，而不是链接到缓存
    huggingface-cli download "$SDXL_REPO_ID" \
        --local-dir "$SDXL_PATH" \
        --local-dir-use-symlinks False
    echo "🎉 Stable Diffusion XL 下载完成！"
fi
echo ""


# --- 步骤 4: 下载 CLIP-ViT-bigG-14 ---
echo "=============================================="
echo "步骤 4: 正在处理 CLIP-ViT-bigG-14"
echo "=============================================="
CLIP_REPO_ID="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
CLIP_PATH="$MODEL_ROOT/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

if [ -f "$CLIP_PATH/config.json" ]; then
    echo "✅ 模型已存在于: $CLIP_PATH。跳过下载。"
else
    echo "模型不存在，开始使用 huggingface-cli 下载到: $CLIP_PATH"
    huggingface-cli download "$CLIP_REPO_ID" \
        --local-dir "$CLIP_PATH" \
        --local-dir-use-symlinks False
    echo "🎉 CLIP-ViT-bigG-14 下载完成！"
fi
echo ""


# --- 步骤 5: 下载 MS-Diffusion Checkpoint ---
echo "=============================================="
echo "步骤 5: 正在处理 MS-Diffusion Checkpoint"
echo "=============================================="
MS_CKPT_PATH="$MODEL_ROOT/MS-Diffusion/ms_adapter.bin"
MS_CKPT_URL="https://huggingface.co/doge1516/MS-Diffusion/resolve/main/ms_adapter.bin"

if [ -f "$MS_CKPT_PATH" ]; then
    echo "✅ Checkpoint 文件已存在: $MS_CKPT_PATH。跳过下载。"
else
    echo "Checkpoint 不存在，开始下载到: $MS_CKPT_PATH"
    wget -O "$MS_CKPT_PATH" "$MS_CKPT_URL"
    echo "🎉 MS-Diffusion Checkpoint 下载完成！"
fi
echo ""


# --- 步骤 6: 最终验证与空间占用报告 ---
echo "=============================================="
echo "步骤 6: 验证并报告磁盘空间占用"
echo "=============================================="
echo "在 '$MODEL_ROOT' 目录下的最终文件结构："
# 使用 tree 命令（如果已安装）会更直观，ls -R 作为备选
if command -v tree &> /dev/null; then
    tree -L 2 "$MODEL_ROOT"
else
    ls -R "$MODEL_ROOT"
fi
echo ""
echo "--- 各模型文件夹磁盘占用大小 ---"
du -sh "$MODEL_ROOT"/*
echo "------------------------------------"
echo ""
echo "🎉🎉🎉 所有模型已通过高效方式成功下载到 '$MODEL_ROOT' 目录下！🎉🎉🎉"
```


```sh
chmod +x download_ms_diffusion.sh
./download_ms_diffusion.sh
```



# python环境

配置环境
```
cd /home/hxiaoap/MS-Diffusion

# txt文件配置环境
conda create -n msdiff python=3.10.0
conda activate msdiff
pip install -r requirements.txt

# 原版本huggingface-hub-0.36.0不兼容
pip uninstall huggingface_hub
pip install huggingface_hub==0.16.4
pip show huggingface_hub
```

卸载环境
```
conda deactivate
conda remove -n msdiff --all
conda env list
```

# 报错 NVIDIA H800 with CUDA capability sm_90 is not compatible with the current PyTorch installation.


The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70 sm_75 sm_80 sm_86.
If you want to use the NVIDIA H800 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/


**不可用**
```
# 查看当前版本
conda activate msdiff
nvidia-smi


# 清除原先用pip现在的旧版本
pip uninstall torch torchvision torchaudio -y
# 确认清除后，没有 /pypi_0 的 torch 残影
conda list | grep torch
# 升级版本兼容
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia



conda install --force-reinstall pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```

# diffusers 版本兼容 MultiControlNetModel

```
pip uninstall diffusers transformers accelerate -y
pip install diffusers>=0.30.0 transformers accelerate

# !pip install "diffusers==0.27.2" "transformers==4.37.2" "accelerate==0.26.0" 
# pip install diffusers==0.25.1 transformers==4.36.2 accelerate==0.25.0

pip uninstall huggingface_hub
pip install huggingface-hub==0.25.2
pip install huggingface_hub==0.20.3
python -c "from diffusers import MultiControlNetModel; print('Import successful!')"
python -c "from diffusers import MultiControlNetModel"
```


# sbatch执行代码

Script File: test.sbatch
```bash
#!/bin/bash
#SBATCH --job-name=msdiff_inference  # Create a short name for your job
#SBATCH --output=logs/msdiff_output_%j.log  # Log output file, saved in the logs directory (%j is the job ID)
#SBATCH --error=logs/msdiff_error_%j.log   # Error log file
#SBATCH --nodes=1                # Number of nodes
#SBATCH --gpus=1                 # Number of GPUs per node (only valid for large/normal partitions)
#SBATCH --time=00:20:00         # Total run time limit (HH:MM:SS) (2 hours, sufficient for one inference)
#SBATCH --partition=normal  # Partition (large/normal/cpu) to submit to
#SBATCH --account=mscaisuperpod      # Required only for multiple projects

# Navigate to the project directory (replace with your gai code path)
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

# Prepare log directory
mkdir -p logs

# Execute inference command
echo "Starting inference with image input..."
python ./python inference.py

if [ $? -eq 0 ]; then
  # Output on success
  echo "Inference task completed."
else
  # Output on failure
  echo "Inference task failed! Check the error log: cat logs/msdiff_error_"$SLURM_JOB_ID".log"
fi
```

Submit the Script
```shell
cd /home/hxiaoap/MS-Diffusion
sed -i 's/\r$//' test.sbatch
sbatch test.sbatch
```

Check Job Status
```shell
squeue -u hxiaoap  # Check if the job is running (R) or pending (PD)
```

View Real-Time Logs: Once the job starts running (status changes to R), use these commands to monitor progress. This will show real-time output from script echo statements and Python code. You should see model download progress bars if applicable. Press Ctrl + C to stop monitoring.
```shell
# Replace 123456 with your actual job ID
tail -f logs/msdiff_output_123456.log  # Monitor standard output
tail -f logs/msdiff_error_123456.log   # Monitor errors (if any)
```

View Full Logs After Job Completion
```shell
# Replace 123456 with your actual job ID
cat logs/msdiff_output_123456.log  # View full output log
cat logs/msdiff_error_123456.log   # View full error log (if any)
```





