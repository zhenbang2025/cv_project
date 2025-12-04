export HF_ENDPOINT=https://hf-mirror.com

#!/bin/bash
# ==============================================
# Using huggingface-cli download command
# ==============================================

# the base directory to save the downloaded models
BASE_DIR="./hf_models"
mkdir -p ${BASE_DIR}

# --------------------------
# --------------------------
echo "===== Download SDXL Base model ====="
huggingface-cli download \
  --resume-download \
  stabilityai/stable-diffusion-xl-base-1.0 \
  --local-dir ${BASE_DIR}/stabilityai/stable-diffusion-xl-base-1.0 \
  --local-dir-use-symlinks False

# --------------------------
# 2. SDXL Refiner model
# --------------------------
echo -e "\n===== Download SDXL Refiner model ====="
huggingface-cli download \
  --resume-download \
  stabilityai/stable-diffusion-xl-refiner-1.0 \
  --local-dir ${BASE_DIR}/stabilityai/stable-diffusion-xl-refiner-1.0 \
  --local-dir-use-symlinks False

# --------------------------
# 3. CLIP image encoder
# --------------------------
echo -e "\n===== Download CLIP-ViT-bigG-14 encoder model===="
huggingface-cli download \
  --resume-download \
  laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
  --local-dir ${BASE_DIR}/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \
  --local-dir-use-symlinks False

# --------------------------
# 4. ControlNet model - Depth
# --------------------------
echo -e "\n===== Download Depth ControlNet (SDXL) ====="
huggingface-cli download \
  --resume-download \
  diffusers/controlnet-depth-sdxl-1.0 \
  --local-dir ${BASE_DIR}/diffusers/controlnet-depth-sdxl-1.0 \
  --local-dir-use-symlinks False

# --------------------------
# 5. ControlNet model - Canny
# --------------------------
echo -e "\n===== 下载 Canny ControlNet (SDXL) ====="
huggingface-cli download \
  --resume-download \
  diffusers/controlnet-canny-sdxl-1.0 \
  --local-dir ${BASE_DIR}/diffusers/controlnet-canny-sdxl-1.0 \
  --local-dir-use-symlinks False

# --------------------------
# 6. ControlNet model - SoftEdge
# --------------------------
echo -e "\n===== download SoftEdge ControlNet (SDXL) ====="
huggingface-cli download \
  --resume-download \
  diffusers/controlnet-softedge-sdxl-1.0 \
  --local-dir ${BASE_DIR}/diffusers/controlnet-softedge-sdxl-1.0 \
  --local-dir-use-symlinks False

# --------------------------

# --------------------------
# --------------------------
echo -e "\n===== Done ====="
echo "Model saves in ${BASE_DIR}"
echo -e "\n example usage:"
echo "python script_name.py \\"
echo "  --base_model_path ${BASE_DIR}/stabilityai/stable-diffusion-xl-base-1.0 \\"
echo "  --refiner_model_path ${BASE_DIR}/stabilityai/stable-diffusion-xl-refiner-1.0 \\"
echo "  --image_encoder_path ${BASE_DIR}/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k \\"
echo "  --controlnet_depth_path ${BASE_DIR}/diffusers/controlnet-depth-sdxl-1.0 \\"
echo "  --controlnet_canny_path ${BASE_DIR}/diffusers/controlnet-canny-sdxl-1.0 \\"
echo "  --controlnet_softedge_path ${BASE_DIR}/diffusers/controlnet-softedge-sdxl-1.0 \\"
echo "  --ms_ckpt ${BASE_DIR}/MS-Diffusion/ms_adapter.bin"