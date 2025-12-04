import os
import torch
import numpy as np
from PIL import Image
from huggingface_hub import InferenceClient
from ultralytics import YOLO
# ControlNet Preprocessors
from controlnet_aux import ZoeDetector, CannyDetector, HEDdetector
# Diffusers & MS-Diffusion
from diffusers import (
    StableDiffusionXLPipeline, 
    StableDiffusionXLControlNetPipeline, 
    ControlNetModel, 
    MultiControlNetModel,
    DiffusionPipeline
)
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from msdiffusion.models.projection import Resampler
from msdiffusion.models.model import MSAdapter
from msdiffusion.utils import get_phrase_idx, get_eot_idx

# ==========================================
# 1. Global Configuration
# ==========================================

# Base working directory
WORK_DIR = "./pipeline_output"
os.makedirs(WORK_DIR, exist_ok=True)

# Model path configuration (Please modify to your actual paths)
MODEL_PATHS = {
    "base_sdxl": "/home/hxiaoap/model/stabilityai/stable-diffusion-xl-base-1.0",
    "refiner": "/home/hxiaoap/model/stabilityai/stable-diffusion-xl-refiner-1.0",
    "image_encoder": "/home/hxiaoap/model/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    "ms_adapter_ckpt": "/home/hxiaoap/model/MS-Diffusion/ms_adapter.bin",
    "controlnets": {
        "depth": "/home/hxiaoap/controlnet/controlnet-depth-sdxl-1.0",
        "canny": "/home/hxiaoap/controlnet/controlnet-canny-sdxl-1.0",
        "softedge": "/home/hxiaoap/controlnet/EcomXL_controlnet_softedge",
    }
}

# Core Workflow Configuration
WORKFLOW_CONFIG = {
    # Unified Prompt, used for T2I generation of Base Image and MS-Diffusion
    "prompt": "Astronaut riding a horse, high quality, photorealistic, 8k",
    
    # Define subjects, corresponding YOLO classes, and reference images for MS-Diffusion
    # MS-Diffusion will strictly map Box and Reference Image based on this list order
    "subjects": [
        {
            "phrase": "astronaut",           # Phrase used for MS-Diffusion
            "yolo_class": "person",          # Category to look for during YOLO detection
            "ref_image": "./examples/example_astronaut.jpg" # Subject reference image path
        },
        {
            "phrase": "horse",
            "yolo_class": "horse",
            "ref_image": "./examples/example_horse.jpg"
        }
    ],
    
    # Base image settings
    "base_image_name": "base_image_pipeline.png",
}

# Experiment parameter configuration
EXPERIMENTS = [
    {"exp_id": "TEST_01", "control_types": ["depth", "canny"], "ms_scale": 0.7, "cn_scales": [0.9, 0.9], 
     "use_refiner": True, "use_freeu": False},
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 2. Helper Functions: Preprocessing Stage (Step 1-3)
# ==========================================

def prepare_assets(config):
    print("\n>>> Stage 1: Preparing Base Image and Control Maps")
    
    base_img_path = os.path.join(WORK_DIR, config["base_image_name"])
    
    # 2.1 Get/Generate Base Image
    if not os.path.exists(base_img_path):
        print(f"Local file not found at {base_img_path}, calling API to generate...")
        try:
            client = InferenceClient(provider="nebius", api_key=os.environ["HF_TOKEN"])
            base_image = client.text_to_image(
                config["prompt"],
                model="black-forest-labs/FLUX.1-dev",
            )
            base_image.save(base_img_path)
            print("Base Image generated and saved.")
        except Exception as e:
            raise RuntimeError(f"API generation failed: {e}")
    else:
        print("Loading local Base Image.")
        base_image = Image.open(base_img_path).convert("RGB")

    # 2.2 Generate ControlNet Maps
    paths = {}
    # Depth
    depth_path = os.path.join(WORK_DIR, "map_depth.png")
    if not os.path.exists(depth_path):
        print("Generating Depth Map...")
        zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
        depth_map = zoe(base_image)
        depth_map.save(depth_path)
        del zoe # Release VRAM
    paths["depth"] = depth_path

    # Canny
    canny_path = os.path.join(WORK_DIR, "map_canny.png")
    if not os.path.exists(canny_path):
        print("Generating Canny Map...")
        canny = CannyDetector()
        canny_map = canny(base_image, low_threshold=100, high_threshold=200)
        canny_map.save(canny_path)
        del canny
    paths["canny"] = canny_path

    # SoftEdge
    softedge_path = os.path.join(WORK_DIR, "map_softedge.png")
    if not os.path.exists(softedge_path):
        print("Generating SoftEdge Map...")
        hed = HEDdetector.from_pretrained('lllyasviel/Annotators')
        hed_map = hed(base_image)
        hed_map.save(softedge_path)
        del hed
    paths["softedge"] = softedge_path

    # 2.3 YOLO Object Detection and Box Extraction
    print("\n>>> Stage 2: YOLO Object Extraction")
    model = YOLO("yolov8m.pt")
    results = model(base_image, conf=0.3, save=False, verbose=False)[0]
    
    w, h = base_image.size
    sorted_boxes = [] # Will store according to the order in config['subjects']
    
    # Get all YOLO detection results
    detected_objects = []
    if results.boxes:
        for box, cls, conf in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.cls.cpu().numpy(), results.boxes.conf.cpu().numpy()):
            detected_objects.append({
                "label": model.names[int(cls)],
                "box": box, # xyxy
                "conf": conf,
                "area": (box[2]-box[0]) * (box[3]-box[1])
            })

    print(f"Raw objects detected by YOLO: {[d['label'] for d in detected_objects]}")

    # Critical Logic: Match Subjects in Order
    for subj in config["subjects"]:
        target_cls = subj["yolo_class"]
        phrase = subj["phrase"]
        
        # Filter all boxes matching the current category
        candidates = [d for d in detected_objects if d["label"] == target_cls]
        
        if not candidates:
            print(f"Warning: Could not detect '{target_cls}' (for '{phrase}') in Base Image. Using default full-image box.")
            sorted_boxes.append([0.0, 0.0, 1.0, 1.0]) # Fallback
        else:
            # If multiple objects of the same class exist, pick the one with highest confidence (or largest area)
            best_cand = sorted(candidates, key=lambda x: x["conf"], reverse=True)[0]
            x1, y1, x2, y2 = best_cand["box"]
            # Normalize
            rel_box = [round(x1/w, 4), round(y1/h, 4), round(x2/w, 4), round(y2/h, 4)]
            sorted_boxes.append(rel_box)
            print(f"Match successful: '{phrase}' -> '{target_cls}' box: {rel_box}")
            
            # Remove selected object from list to prevent different subjects grabbing the same box (if needed)
            detected_objects.remove(best_cand)

    # Clean up VRAM
    del model
    torch.cuda.empty_cache()
    
    return {
        "base_image": base_image,
        "map_paths": paths,
        "boxes": [sorted_boxes], # MS-Diffusion format requires an extra nesting layer: [ [box1, box2] ]
        "subject_images": [Image.open(s["ref_image"]).convert("RGB").resize((512, 512)) for s in config["subjects"]],
        "phrases": [[s["phrase"] for s in config["subjects"]]] # Also nested
    }


# ==========================================
# 3. Helper Functions: Generation Stage (Step 4 - MS Diffusion)
# ==========================================

def load_ms_models():
    print("\n>>> Stage 3: Loading MS-Diffusion Models")
    
    # Load Pipelines
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_PATHS["base_sdxl"], torch_dtype=torch.float16
    ).to(device)
    
    refiner = DiffusionPipeline.from_pretrained(
        MODEL_PATHS["refiner"],
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(device)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        MODEL_PATHS["image_encoder"]
    ).to(device, dtype=torch.float16)
    
    image_processor = CLIPImageProcessor()
    
    # Load ControlNets
    cnet_models = {}
    for key, path in MODEL_PATHS["controlnets"].items():
        cnet_models[key] = ControlNetModel.from_pretrained(path, torch_dtype=torch.float16)
        
    return pipe, refiner, image_encoder, image_processor, cnet_models

def get_phrases_idx_wrapper(tokenizer, phrases, prompt):
    res = []
    phrase_cnt = {}
    for phrase in phrases:
        if phrase in phrase_cnt:
            phrase_cnt[phrase] += 1
        else:
            phrase_cnt[phrase] = 1
        # Get token index
        res.append(get_phrase_idx(tokenizer, phrase, prompt, num=phrase_cnt[phrase])[0])
    return res

def run_ms_generation(prepped_data, ms_components, experiments):
    pipe, refiner, image_encoder, image_processor, cnet_models = ms_components
    
    # Prepare common inputs
    prompt = WORKFLOW_CONFIG["prompt"]
    phrases = prepped_data["phrases"]         # e.g. [["astronaut", "horse"]]
    boxes = prepped_data["boxes"]             # e.g. [[[0.1,0.1,0.5,0.5], ...]]
    input_subject_images = prepped_data["subject_images"]
    
    # Prepare Maps (Load into memory and resize)
    control_maps = {}
    for k, v in prepped_data["map_paths"].items():
        control_maps[k] = Image.open(v).resize((1024, 1024))

    # Calculate Text Embeddings index
    phrase_idxes = [get_phrases_idx_wrapper(pipe.tokenizer, phrases[0], prompt)]
    eot_idxes = [[get_eot_idx(pipe.tokenizer, prompt)] * len(phrases[0])]

    print(f"Prompt: {prompt}")
    print(f"Phrases: {phrases}")
    print(f"Boxes: {boxes}")

    for exp in experiments:
        print(f"\nRunning experiment: {exp['exp_id']}")
        
        # 1. Dynamically Assemble ControlNet
        active_cn_types = exp["control_types"]
        loaded_cns = [cnet_models[t].to(device) for t in active_cn_types]
        
        # Switch ControlNet in Pipeline
        cn_wrapper = MultiControlNetModel(loaded_cns) if len(loaded_cns) > 1 else loaded_cns[0]
        current_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            MODEL_PATHS["base_sdxl"],
            controlnet=cn_wrapper,
            unet=pipe.unet, 
            vae=pipe.vae,
            text_encoder=pipe.text_encoder,
            text_encoder_2=pipe.text_encoder_2,
            tokenizer=pipe.tokenizer,
            tokenizer_2=pipe.tokenizer_2,
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(device)
        
        # Configure FreeU
        if exp["use_freeu"]:
            current_pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
        else:
            current_pipe.disable_freeu()

        # 2. Initialize MS-Adapter
        image_proj_model = Resampler(
            dim=1280, depth=4, dim_head=64, heads=20, num_queries=16,
            embedding_dim=image_encoder.config.hidden_size,
            output_dim=current_pipe.unet.config.cross_attention_dim,
            ff_mult=4, latent_init_mode="grounding",
            phrase_embeddings_dim=current_pipe.text_encoder.config.projection_dim,
        ).to(device, dtype=torch.float16)

        ms_model = MSAdapter(
            current_pipe.unet, image_proj_model, 
            ckpt_path=MODEL_PATHS["ms_adapter_ckpt"], 
            device=device, num_tokens=16
        ).to(device, dtype=torch.float16)

        # 3. Prepare Control Images
        current_control_images = [control_maps[t] for t in active_cn_types]
        if len(current_control_images) == 1: 
            current_control_images = current_control_images[0]

        # 4. Generate
        steps = 40 if exp["use_refiner"] else 30
        denoise_end = 0.8 if exp["use_refiner"] else None
        output_type = "latent" if exp["use_refiner"] else "pil"

        # Base Generation
        gen_output = ms_model.generate(
            pipe=current_pipe, 
            pil_images=[input_subject_images], 
            num_samples=1,
            num_inference_steps=steps, 
            seed=42, 
            prompt=[prompt], 
            scale=exp["ms_scale"], 
            image_encoder=image_encoder, 
            image_processor=image_processor, 
            boxes=boxes,
            image_proj_type="resampler", 
            image_encoder_type="clip", 
            phrases=phrases, 
            drop_grounding_tokens=[0],
            phrase_idxes=phrase_idxes, 
            eot_idxes=eot_idxes, 
            height=1024, width=1024, 
            image=current_control_images, 
            controlnet_conditioning_scale=exp["cn_scales"],
            output_type=output_type,
            denoising_end=denoise_end
        )

        # Refiner Generation (Optional)
        if exp["use_refiner"]:
            final_images = refiner(
                prompt=prompt,
                image=gen_output,
                guidance_scale=7.5,
                num_inference_steps=steps,
                denoising_start=denoise_end,
            ).images
        else:
            final_images = gen_output

        # 5. Save Results
        save_dir = os.path.join(WORK_DIR, "results", exp["exp_id"])
        os.makedirs(save_dir, exist_ok=True)
        final_images[0].save(os.path.join(save_dir, "result.png"))
        print(f"Experiment {exp['exp_id']} completed. Image saved to {save_dir}")


# ==========================================
# 4. Main Entry Point
# ==========================================
if __name__ == "__main__":
    # 1. Preprocessing: Get Image, Generate Maps, Extract Boxes
    #    This step loads YOLO and ControlNet Aux, releases them after use
    prepped_data = prepare_assets(WORKFLOW_CONFIG)
    
    # 2. Load Generation Models: SDXL, MS-Adapter
    #    This step consumes significant VRAM
    ms_components = load_ms_models()
    
    # 3. Run Experiments
    run_ms_generation(prepped_data, ms_components, EXPERIMENTS)
    
    print("All tasks completed!")