import os
import argparse
import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    StableDiffusionXLControlNetPipeline, 
    ControlNetModel, 
    MultiControlNetModel,
    DiffusionPipeline
)
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from msdiffusion.models.projection import Resampler
from msdiffusion.models.model import MSAdapter
from msdiffusion.utils import get_phrase_idx, get_eot_idx
import itertools

def parse_args():
    parser = argparse.ArgumentParser(description='MS-Diffusion with ControlNet and Refiner')
    
    parser.add_argument('--base_model_path', type=str, 
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help='Path or HF repo name for SDXL base model')
    parser.add_argument('--refiner_model_path', type=str,
                        default="stabilityai/stable-diffusion-xl-refiner-1.0",
                        help='Path or HF repo name for SDXL refiner model')
    parser.add_argument('--image_encoder_path', type=str,
                        default="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                        help='Path or HF repo name for CLIP image encoder')
    parser.add_argument('--ms_ckpt', type=str,
                        default="/home/hxiaoap/model/MS-Diffusion/ms_adapter.bin",
                        help='Path to MS-Adapter checkpoint (local file only)')
    
    parser.add_argument('--controlnet_depth_path', type=str,
                        default="diffusers/controlnet-depth-sdxl-1.0",
                        help='Path or HF repo name for Depth ControlNet')
    parser.add_argument('--controlnet_canny_path', type=str,
                        default="diffusers/controlnet-canny-sdxl-1.0",
                        help='Path or HF repo name for Canny ControlNet')
    parser.add_argument('--controlnet_softedge_path', type=str,
                        default="diffusers/controlnet-softedge-sdxl-1.0",
                        help='Path or HF repo name for SoftEdge ControlNet')
    
    parser.add_argument('--depth_image_path', type=str,
                        default="./examples/depth_v6.png",
                        help='Path to depth control image (local file)')
    parser.add_argument('--canny_image_path', type=str,
                        default="./examples/canny_v6.png",
                        help='Path to canny control image (local file)')
    parser.add_argument('--softedge_image_path', type=str,
                        default="./examples/soft_edge_v6.png",
                        help='Path to softedge control image (local file)')
    
    parser.add_argument('--boxes', type=str,
                        default="0.1356,0.1006,0.5625,0.8558;0.1356,0.1006,0.5625,0.8558;0.3090,0.3465,0.4216,0.4955;0.4950,0.4104,0.7242,0.9026",
                        help='Bounding boxes in format x1,y1,x2,y2;x1,y1,x2,y2...')
    
    parser.add_argument('--input_images', type=str,
                        default="./examples/woman.png,./examples/dress.png,./examples/bag.png,./examples/suitcase.jpg",
                        help='Input images path (comma separated, local files only)')
    
    parser.add_argument('--use_freeu', action='store_true', default=False,
                        help='Whether to enable FreeU enhancement')
    parser.add_argument('--use_refiner', action='store_true', default=False,
                        help='Whether to use SDXL refiner')
    
    parser.add_argument('--result_path', type=str, default="./res",
                        help='Result save path (local directory)')
    parser.add_argument('--log_id', type=str, default="test_optimized",
                        help='Log ID for experiment')
    parser.add_argument('--load_type', type=str, default="checkpoint-xxxxxx",
                        help='Load type identifier')
    
    parser.add_argument('--ms_scale', type=float, default=0.8,
                        help='MS-Adapter scale')
    parser.add_argument('--cn_scales', type=str, default="0.9,0.7",
                        help='ControlNet scales (comma separated)')
    parser.add_argument('--control_types', type=str, default="depth,softedge",
                        help='ControlNet types to use (comma separated: depth,canny,softedge)')
    parser.add_argument('--exp_id', type=str, default="M4-8",
                        help='Experiment ID')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate')
    
    parser.add_argument('--prompt', type=str,
                        default="Sitting on a bench with a grey background, the woman is wearing a solid-colored dress without any patterns, holding a bag, and a rolling suitcase lies next to her.",
                        help='Generation prompt')
    parser.add_argument('--phrases', type=str, default="woman,dress,bag,bench",
                        help='Phrases for grounding (comma separated)')
    
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Cache directory for HF models (default: HF default cache)')
    parser.add_argument('--revision', type=str, default="fp16",
                        help='Model revision to use (e.g., fp16, main)')
    parser.add_argument('--use_safetensors', action='store_true', default=True,
                        help='Use safetensors format if available')
    
    return parser.parse_args()

def get_phrases_idx(tokenizer, phrases, prompt):
    res = []
    phrase_cnt = {}
    for phrase in phrases:
        if phrase in phrase_cnt:
            cur_cnt = phrase_cnt[phrase]
            phrase_cnt[phrase] += 1
        else:
            cur_cnt = 0
            phrase_cnt[phrase] = 1
        res.append(get_phrase_idx(tokenizer, phrase, prompt, num=cur_cnt)[0])
    return res

def parse_boxes(box_str):
    boxes = []
    box_groups = []
    for box_part in box_str.split(';'):
        coords = list(map(float, box_part.split(',')))
        box_groups.append(coords)
    boxes.append(box_groups)
    return boxes

def main():
    args = parse_args()
    
    # --- Fix Environment Settings ---
    base_model_path = args.base_model_path
    refiner_model_path = args.refiner_model_path
    image_encoder_path = args.image_encoder_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result_path = args.result_path
    log_id = args.log_id
    load_type = args.load_type
    ms_ckpt = args.ms_ckpt

    hf_kwargs = {
        "torch_dtype": torch.float16,
        "cache_dir": args.cache_dir,
        "revision": args.revision,
        "use_safetensors": args.use_safetensors
    }

    print("Loading Refiner Model from HF Hub...")
    try:
        temp_pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            **hf_kwargs
        )
        
        refiner_pipe = DiffusionPipeline.from_pretrained(
            refiner_model_path,
            text_encoder_2=temp_pipe.text_encoder_2,
            vae=temp_pipe.vae,
            **hf_kwargs
        ).to(device)
        
        del temp_pipe
        print("Refiner 模型加载完成。")
    except Exception as e:
        print(f"Warning: Refiner loading failed. {e}")
        refiner_pipe = None

    controlnet_path_dict = {
        "depth": args.controlnet_depth_path,
        "canny": args.controlnet_canny_path,
        "softedge": args.controlnet_softedge_path,
    }

    controlnet_models = {}
    for name, path in controlnet_path_dict.items():
        if path:
            try:
                controlnet_models[name] = ControlNetModel.from_pretrained(
                    path,
                    **hf_kwargs
                )
                print(f"Loaded ControlNet {name} from: {path}")
            except Exception as e:
                print(f"Failed to load ControlNet {name} from {path}: {e}")
        else:
            print(f"ControlNet path for {name} is empty")

    boxes = parse_boxes(args.boxes)
    
    control_image_groups = {
        "v6-new": {
            "depth": Image.open(args.depth_image_path).resize((1024, 1024)) if os.path.exists(args.depth_image_path) else None,
            "canny": Image.open(args.canny_image_path).resize((1024, 1024)) if os.path.exists(args.canny_image_path) else None,
            "softedge": Image.open(args.softedge_image_path).resize((1024, 1024)) if os.path.exists(args.softedge_image_path) else None,
            "boxes": boxes,
        }
    }

    # --- Preload other components ---
    print(f"Loading CLIP image encoder from: {image_encoder_path}")
    image_processor = CLIPImageProcessor()
    image_encoder_type = "clip"
    
    try:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            image_encoder_path,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16
        ).to(device)
        print("CLIP image encoder loaded successfully")
    except Exception as e:
        print(f"Failed to load CLIP image encoder: {e}")
        raise
    
    image_encoder_projection_dim = image_encoder.config.projection_dim
    num_tokens = 16
    image_proj_type = "resampler"
    latent_init_mode = "grounding"

    input_image_paths = args.input_images.split(',')
    input_images = []
    for img_path in input_image_paths:
        if os.path.exists(img_path):
            input_images.append(Image.open(img_path).convert("RGB").resize((512, 512)))
        else:
            print(f"Warning: Input image path {img_path} does not exist")

    phrases_list = [args.phrases.split(',')]
    common_inputs = {
        "prompt": args.prompt,
        "phrases": phrases_list,
        "input_images": input_images,
        "num_samples": args.num_samples,
    }

    control_types = args.control_types.split(',') if args.control_types else []
    control_types = [ct for ct in control_types if ct in controlnet_models and controlnet_models[ct] is not None]
    
    try:
        cn_scales = list(map(float, args.cn_scales.split(','))) if args.cn_scales else None
        if cn_scales and len(cn_scales) != len(control_types) and len(cn_scales) == 1:
            cn_scales = cn_scales * len(control_types)
    except:
        cn_scales = None
        print("Warning: Invalid cn_scales format, using None")

    exp_config = {
        "exp_id": args.exp_id,
        "control_types": control_types,
        "ms_scale": args.ms_scale,
        "cn_scales": cn_scales,
        "use_refiner": args.use_refiner,
        "use_freeu": args.use_freeu,
    }

    # ---- Reusable experiment run functions ----
    def run_experiment(exp_config, base_image_version):
        exp_id = exp_config["exp_id"]
        control_types = exp_config["control_types"]
        ms_scale = exp_config["ms_scale"]
        cn_scales = exp_config.get("cn_scales", None)
        use_refiner = exp_config.get("use_refiner", False)
        use_freeu = exp_config.get("use_freeu", False)
        
        print("\n" + "="*50)
        print(f"--- Experiment: {exp_id} ---")
        print(f"ControlNet: {control_types if control_types else 'None'}")
        print(f"MS-Adapter Scale: {ms_scale}")
        if cn_scales:
            print(f"ControlNet Scales: {cn_scales}")
        print(f"FreeU Enabled: {use_freeu}")
        print(f"Refiner Enabled: {use_refiner}")
        
        # --- Dynamically Construct ControlNet and Pipeline ---
        if not isinstance(control_types, list):
            control_types = [control_types]
        
        loaded_cns = []
        for t in control_types:
            if t in controlnet_models and controlnet_models[t] is not None:
                loaded_cns.append(controlnet_models[t].to(device))
        
        controlnet = None
        if len(loaded_cns) > 1:
            print(f"组合 {len(loaded_cns)} 个 ControlNet...")
            controlnet = MultiControlNetModel(loaded_cns)
        elif len(loaded_cns) == 1:
            print("使用单个 ControlNet...")
            controlnet = loaded_cns[0]
        else:
            print("不使用 ControlNet。")
        
        # 构建Pipeline（支持HF repo）
        try:
            pipeline_kwargs = {
                "pretrained_model_name_or_path": base_model_path,
                "controlnet": controlnet,
                "torch_dtype": torch.float16,
                "add_watermarker": False,
                "cache_dir": args.cache_dir,
                "revision": args.revision,
                "use_safetensors": args.use_safetensors
            }
            
            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(**pipeline_kwargs).to(device)
        except Exception as e:
            print(f"Failed to create pipeline: {e}")
            return

        if use_freeu:
            try:
                pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.3, b2=1.4)
                print("FreeU enabled with s1=0.9, s2=0.2, b1=1.3, b2=1.4")
            except Exception as e:
                print(f"Failed to enable FreeU: {e}")

        try:
            image_proj_model = Resampler(
                dim=1280,
                depth=4,
                dim_head=64,
                heads=20,
                num_queries=num_tokens,
                embedding_dim=image_encoder.config.hidden_size,
                output_dim=pipe.unet.config.cross_attention_dim,
                ff_mult=4,
                latent_init_mode=latent_init_mode,
                phrase_embeddings_dim=pipe.text_encoder.config.projection_dim,
            ).to(device, dtype=torch.float16)
        except Exception as e:
            print(f"Failed to create Resampler: {e}")
            return
        
        try:
            ms_model = MSAdapter(pipe.unet, image_proj_model, ckpt_path=ms_ckpt, device=device, num_tokens=16)
            ms_model.to(device, dtype=torch.float16)
        except Exception as e:
            print(f"Failed to create MSAdapter: {e}")
            return

        control_image_dict = control_image_groups[base_image_version]
        control_input_images = []
        for t in control_types:
            if t in control_image_dict and control_image_dict[t] is not None:
                control_input_images.append(control_image_dict[t])
        
        if control_input_images and len(control_input_images) == 1:
            control_input_images = control_input_images[0]
        elif not control_input_images:
            control_input_images = None

        prompt = common_inputs["prompt"]
        phrases = common_inputs["phrases"]
        input_images = common_inputs["input_images"]
        boxes = control_image_dict["boxes"]
        
        try:
            phrase_idxes = [get_phrases_idx(pipe.tokenizer, phrases[0], prompt)]
            eot_idxes = [[get_eot_idx(pipe.tokenizer, prompt)] * len(phrases[0])]
        except Exception as e:
            print(f"Failed to process phrase indexes: {e}")
            return

        total_steps = 40 if use_refiner else 30
        denoising_end_step = 0.8 if use_refiner else None
        output_type = "latent" if use_refiner else "pil"
        
        try:
            drop_grounding_tokens = [0]
            base_output = ms_model.generate(
                pipe=pipe, pil_images=[input_images], num_samples=common_inputs["num_samples"],
                num_inference_steps=total_steps, seed=0, prompt=[prompt], scale=ms_scale,
                image_encoder=image_encoder, image_processor=image_processor, boxes=boxes,
                image_proj_type="resampler", image_encoder_type="clip",
                phrases=phrases, drop_grounding_tokens=drop_grounding_tokens,
                phrase_idxes=phrase_idxes, eot_idxes=eot_idxes, height=1024, width=1024,
                image=control_input_images if control_types else None,
                controlnet_conditioning_scale=cn_scales if control_types else None,
                output_type=output_type,
                denoising_end=denoising_end_step
            )
        except Exception as e:
            print(f"Failed to generate base output: {e}")
            return

        if use_refiner and refiner_pipe is not None:
            try:
                latents = base_output
                images = refiner_pipe(
                    prompt=prompt,
                    image=latents,
                    guidance_scale=7.5,
                    num_inference_steps=total_steps,
                    denoising_start=denoising_end_step,
                ).images
                print("Refiner processing completed")
            except Exception as e:
                print(f"Refiner failed: {e}, using base output instead")
                images = base_output
        else:
            images = base_output

        try:
            save_name = f"{base_image_version}"
            save_path = os.path.join(result_path, log_id, load_type, save_name)
            os.makedirs(save_path, exist_ok=True)
            
            enhancement_str = ""
            if use_refiner:
                enhancement_str += "_refiner"
            if use_freeu:
                enhancement_str += "_freeu"
            
            for j, image in enumerate(images):
                save_filename = f"{exp_id}_{'_'.join(control_types)}_{j}{enhancement_str}.jpg"
                image_path = os.path.join(save_path, save_filename)
                image.save(image_path)
                print(f"Saved image to: {image_path}")
        except Exception as e:
            print(f"Failed to save images: {e}")
        
        del pipe
        del ms_model
        del image_proj_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for version_name in control_image_groups.keys():
        run_experiment(exp_config, base_image_version=version_name)

if __name__ == "__main__":
    main()