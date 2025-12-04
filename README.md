# MS-Diffusion: Multi-Control Image Generation with SDXL

A flexible and powerful image generation pipeline combining **MS-Adapter**, **ControlNet (Multi-Control support)**, **SDXL Base/Refiner**, and **FreeU enhancement** for precise, high-quality image synthesis.

## 🌟 Key Features
- Support for multiple ControlNet types (Depth, Canny, SoftEdge)
- Seamless SDXL Base + Refiner integration for photorealistic results
- FreeU enhancement for improved detail and contrast
- Grounded generation with phrase-level control (via MS-Adapter)
- Command-line configurable parameters (no hardcoded paths!)
- Hugging Face Hub model auto-download (or local path support)
- Breakpoint resume for model downloads

## 📷 Generation Theme
**Focus: Fashion & Object Composition**  
Generate studio-style images of people with controlled clothing, accessories, and backgrounds. Example use cases:
- Person with specified clothing (dresses, jackets, etc.)
- Controlled placement of accessories (bags, suitcases, hats)
- Consistent background styling (solid colors, minimal scenes)
- Precise object positioning via bounding boxes

## 🎨 Experiment in **controlnet** visualization
<img src="res/example_cat.jpg" alt="cat Image" width="250">
<img src="res/example_dog.jpg" alt="dog Image" width="250">


| ControlNet Combination | Result Preview |
|---------------|----------------|
| Baseline | ![Baseline](res/baseline.png) |
| Depth + Canny |![Depth+Canny](res/group1_depth_canny.png) |
| Depth + SoftEdge |![Depth+SoftEdge](res/Group_M2_Depth_Softedge​.png) |
| Depth + Canny + SoftEdge |![Depth+Canny+SoftEdge](res/Group_M3_Depth_Canny_Softedge​.png) |

## 🎨 Evaluation

DINO – similarity between reference images to target image​
CLIP-T – similarity between prompt and target image​


Baseline: DINO: 0.36; CLIP-T: 0.12​

Improve DINO from 0.36 to 0.61 and improve CLIP-T from 0.12 to 0.32​

<img src="res/DINO_CLIP_evaluate.png" alt="evaluation" width="800">

## 🎨 Experiment in **FreeU** visualization
<img src="res/freeu_exp2.png" alt="cat Image" width="500">


## 🎨 Experiment in **Multi-object** visualization
| object1                          | object2                          | object3                          | controlnet input                | result                              |
|----------------------------------|----------------------------------|----------------------------------|----------------------------------|-------------------------------------|
| <img src="res/bag_woman.png" width="100"> | <img src="res/blue_shirt.png" width="100"> | <img src="res/white_skirt.png" width="100"> |  ❌ | <img src="res/controlnet_3_object.png" width="100"> |
| <img src="res/bag_woman.png" width="100"> | <img src="res/blue_shirt.png" width="100"> | <img src="res/white_skirt.png" width="100"> | <img src="res/woman_openpose.png" width="100"> | <img src="res/no_controlnet_3_object.jpg" width="100"> |

| object1                          | object2                          | object3                          | object4                | result                              |
|----------------------------------|----------------------------------|----------------------------------|----------------------------------|-------------------------------------|
| <img src="res/blue_bag.png" width="100"> | <img src="res/laggage.png" width="100"> | <img src="res/model_face.png" width="100"> | <img src="res/pink_skirt.png" width="100"> | <img src="res/4_objectresult.png" width="100"> |

### Methodology: Multi-Subject Image Generation Pipeline
This section details the **5-stage pipeline** for generating consistent, layout-aligned multi-subject images (used in Experiment 1: 2-subject generation of a grey cat + Corgi).


#### Stage 1: VLM-Enhanced Prompt Construction
| Input               | Method                                                                 | Purpose                                                                 |
|---------------------|-----------------------------------------------------------------------|-------------------------------------------------------------------------|
| Reference subject images (grey cat, Corgi) | Use VLM (e.g., CLIP/GPT-4V) to extract semantic keywords (e.g., *"short-haired grey cat, fluffy white-brown Corgi"*) | Enrich prompt with subject-specific attributes to improve fidelity      |


#### Stage 2: Base Image Generation via T2I Model
**Input**: Enhanced prompt (e.g., *"The grey cat and Corgi are playing by the seaside, with the cat's paw stroking the dog's head"*)  
**Method**: Use a high-quality T2I model (e.g., SDXL Base) to generate an initial base image.  
**Key Constraint**: Prioritize *visual quality* (lighting, spatial relations, action) over subject consistency (no subject constraints here).  
**Output**: Baseline image (see Experiment 1 results) with correct layout but potential subject deviation.


#### Stage 3: Conditional Control Map Extraction
**Input**: Base image (Stage 2)  
**Method**: Extract control maps from the base image:
- Depth map: Encodes 3D spatial layout (foreground/background separation)
- Canny edge map: Captures object contours (e.g., cat/Corgi body shapes)
- Soft edge map: Smooths edge textures for natural transitions  
**Purpose**: Provide structural guidance to constrain layout consistency in later stages.


#### Stage 4: Subject Bounding Box Extraction
**Input**: Base image  
**Method**: Use object detection (e.g., YOLOv8) to get *normalized (0–1 scale)* bounding boxes for each subject (e.g., `[0.13, 0.10, 0.56, 0.85]` for the cat).  
**Purpose**: Define spatial regions for subjects to enforce layout alignment.


#### Stage 5: Final Image Generation via MS-Diffusion
**Input**:
- Enhanced prompt (Stage 1)
- Control maps (Stage 3)
- Subject bounding boxes (Stage 4)
- Tunable parameters (Experiment 1 variables: `Control scale` (0.2–0.9), `MS-Diffusion scale` (0.6–0.9), `Control types` (depth, canny, soft edge))

**Method**: MS-Diffusion integrates inputs to balance:
1. **Semantic content** (prompt)
2. **Structural consistency** (control maps via ControlNet)
3. **Spatial layout** (bounding boxes via MS-Diffusion guidance)

**Experiment 1 Variables**:
| Group       | Control Type Combination       | Tuned Scales                                  |
|-------------|--------------------------------|-----------------------------------------------|
| M1          | Depth + Canny                  | Control scale (0.2–0.9); MS scale (0.6/0.7/0.9) |
| M2          | Depth + Softedge               | Same as M1                                    |
| M3          | Depth + Canny + Softedge       | Same as M1                                    |

**Output**: Final images (e.g., M1-1 to M1-8) with aligned layout + consistent subject appearance.


### 3.6 Pipeline Validation (Experiment 1 Results)
| Condition                  | Outcome                                                                 |
|----------------------------|-------------------------------------------------------------------------|
| Baseline (MS-Diffusion w/o ControlNet) | Subject inconsistency (e.g., cat fur color deviates from reference)     |
| ControlNet-enhanced (M1–M3) | Preserves **subject fidelity** (cat/Corgi features) + **layout consistency** (bounding box alignment) |

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/ms-diffusion.git
```
### 2. Install Dependencies
```bash
cd ms-diffusion
pip install -r requirements.txt
```
# Download Models
```bash
chmod +x download_models.sh
# Run the download script
# Linux/macOS
./download_models.sh
```

### 4. Prepare Control Images
Place your control images in the examples/ directory (or specify custom paths via CLI):
Depth map: examples/depth_v6.png
Canny edge: examples/canny_v6.png
Soft edge: examples/soft_edge_v6.png
### 5. Run Generation
Use the command-line interface to configure your experiment. Example commands:

Basic Run (Default Parameters)
```bash
python main.py \
  --base_model_path ./hf_models/stabilityai/stable-diffusion-xl-base-1.0 \
  --refiner_model_path ./hf_models/stabilityai/stable-diffusion-xl-refiner-1.0 \
  --control_types "depth,softedge" \
  --cn_scales "0.9,0.7" \
  --ms_scale 0.8 \
  --use_refiner \
  --use_freeu \
  --prompt "A woman wearing a red dress, holding a black bag, grey background" \
  --phrases "woman,dress,bag,background" \
  --boxes "0.13,0.10,0.56,0.85;0.13,0.10,0.56,0.85;0.30,0.34,0.42,0.49;0.0,0.0,1.0,1.0" \
  --exp_id "custom-red-dress" \
  --num_samples 2
```

#### model settings
| Model | Hugging Face Repo | Local Path (after download) |
|-------|-------------------|------------------------------|
| SDXL Base | stabilityai/stable-diffusion-xl-base-1.0 | `./hf_models/stabilityai/stable-diffusion-xl-base-1.0` |
| SDXL Refiner | stabilityai/stable-diffusion-xl-refiner-1.0 | `./hf_models/stabilityai/stable-diffusion-xl-refiner-1.0` |
| CLIP Encoder | laion/CLIP-ViT-bigG-14-laion2B-39B-b160k | `./hf_models/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` |
| Depth ControlNet | diffusers/controlnet-depth-sdxl-1.0 | `./hf_models/diffusers/controlnet-depth-sdxl-1.0` |
| Canny ControlNet | diffusers/controlnet-canny-sdxl-1.0 | `./hf_models/diffusers/controlnet-canny-sdxl-1.0` |
| SoftEdge ControlNet | diffusers/controlnet-softedge-sdxl-1.0 | `./hf_models/diffusers/controlnet-softedge-sdxl-1.0` |

### Custom Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--control_types` | Comma-separated ControlNet types (depth/canny/softedge) | `depth,softedge` |
| `--cn_scales` | ControlNet strength scales (one per control type) | `0.9,0.7` |
| `--ms_scale` | MS-Adapter grounding strength | `0.8` |
| `--use_refiner` | Enable SDXL Refiner for high-quality output | `False` |
| `--use_freeu` | Enable FreeU enhancement for better details | `False` |
| `--num_samples` | Number of images to generate per run | `1` |
| `--prompt` | Text prompt for image generation | `Fashion-focused default` |
| `--boxes` | Normalized bounding boxes for object positioning | `Default fashion layout` |
