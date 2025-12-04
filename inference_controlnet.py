import os
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline, ControlNetModel
from PIL import Image
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor

from msdiffusion.models.projection import Resampler
from msdiffusion.models.model import MSAdapter
from msdiffusion.utils import get_phrase_idx, get_eot_idx


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


# base_model_path = "/path/to/your/model"
# image_encoder_path = "/path/to/your/image_encoder"
# device = "cuda"
# result_path = "./res"
# log_id = "test"
# load_type = "checkpoint-xxxxxx"
# ms_ckpt = f"./output/{log_id}/{load_type}/ms_adapter.bin"
base_model_path = "/home/hxiaoap/model/stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "/home/hxiaoap/model/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
result_path = "./res"
log_id = "test"
load_type = "checkpoint-xxxxxx"
ms_ckpt = f"/home/hxiaoap/model/MS-Diffusion/ms_adapter.bin"

image_processor = CLIPImageProcessor()

# controlnet
# controlnet_path = "/home/hxiaoap/controlnet/controlnet-depth-sdxl-1.0"
# controlnet_path = "/home/hxiaoap/controlnet/control_v11p_sd15_softedge" # 维度不匹配，好像用不了
# controlnet_path = "/home/hxiaoap/controlnet/controlnet-sd-xl-1.0-softedge-dexined" # 好像也用不了，精度有区别
# controlnet_path = "/home/hxiaoap/controlnet/controlnet-canny-sdxl-1.0"
controlnet_path = "/home/hxiaoap/controlnet/EcomXL_controlnet_softedge"
# load SDXL pipeline
# controlnet = ControlNetModel.from_pretrained(controlnet_path, variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to(device)
controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=True, torch_dtype=torch.float16).to(device)

# controlnet = ControlNetModel.from_pretrained("/home/hxiaoap/controlnet/controlnet-sd-xl-1.0-softedge-dexined") # 它没有fp16版本

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    use_safetensors=True,
    torch_dtype=torch.float16,
    add_watermarker=False,
).to(device)

image_encoder_type = "clip"
image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(device, dtype=torch.float16)
image_encoder_projection_dim = image_encoder.config.projection_dim
num_tokens = 16
image_proj_type="resampler"
latent_init_mode="grounding"
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
ms_model = MSAdapter(pipe.unet, image_proj_model, ckpt_path=ms_ckpt, device=device, num_tokens=num_tokens)
ms_model.to(device, dtype=torch.float16)

image0 = Image.open("./examples/example_cat.jpg")
image1 = Image.open("./examples/example_dog.jpg")
# input_images = [image0]
input_images = [image0, image1]
input_images = [x.convert("RGB").resize((512, 512)) for x in input_images]
# control_image = Image.open("./examples/depth.png").resize((1024, 1024))
control_image = Image.open("./examples/canny(1).png").resize((1024, 1024))

# generation configs
# num_samples = 5
# prompt = "best quality, high quality, a dog on the beach"
# print(prompt)
# boxes = [[[0.25, 0.25, 0.75, 0.75]]]  # dog
# # boxes = [[[0., 0.25, 0.4, 0.75], [0.6, 0.25, 1., 0.75]]]  # dog+cat
# # boxes = [[[0., 0., 0., 0.], [0., 0., 0., 0.]]]  # used if you want no layout guidance
# phrases = [["dog"]]
# # phrases = [["dog", "cat"]]
# drop_grounding_tokens = [0]  # set to 1 if you want to drop the grounding tokens
# controlnet_conditioning_scale = 0.7

# # used to get the attention map, return zero if the phrase is not in the prompt
# phrase_idxes = [get_phrases_idx(pipe.tokenizer, phrases[0], prompt)]
# eot_idxes = [[get_eot_idx(pipe.tokenizer, prompt)] * len(phrases[0])]
# print(phrase_idxes, eot_idxes)

# images = ms_model.generate(pipe=pipe, pil_images=[input_images], num_samples=num_samples, num_inference_steps=30, seed=0,
#                            prompt=[prompt], scale=0.6, image_encoder=image_encoder, image_processor=image_processor, boxes=boxes,
#                            image_proj_type=image_proj_type, image_encoder_type=image_encoder_type, phrases=phrases, drop_grounding_tokens=drop_grounding_tokens,
#                            phrase_idxes=phrase_idxes, eot_idxes=eot_idxes, height=1024, width=1024, 
#                            image=control_image, controlnet_conditioning_scale=controlnet_conditioning_scale)

# save_name = "dog_depth"
# save_path = os.path.join(result_path, log_id, load_type, save_name)
# os.makedirs(save_path, exist_ok=True)
# for i, image in enumerate(images):
#     image.save(os.path.join(save_path, f"{i}.jpg"))



# generation configs
num_samples = 1
prompt = "The cat and the dog are playing by the seaside, with the cat's paw stroking the dog's head"
print(prompt)
# 第一个boxes是猫在左狗在右，第二个box是不使用box控制
# boxes = [[[0.0, 0.1, 0.5, 0.9], [0.5, 0.2, 1.0, 1.0]]]
# boxes = [[[0.07, 0.14, 0.55, 0.91], [0.32, 0.33, 0.67, 0.96]]] # imagen
boxes = [[[0.01, 0.15, 0.57, 0.97], [0.47, 0.32, 0.84, 0.94]]] # Qwen

phrases = [["cat", "dog"]]
drop_grounding_tokens = [0]  # set to 1 if you want to drop the grounding tokens
# controlnet_conditioning_scale = 0.7
# ms_controlnet_scales = [[0.6, 0.5], [0.6, 0.3], [0.9, 0.6]]

# for c, scales in enumerate(ms_controlnet_scales):
#     ms_scale = scales[0]
#     controlnet_conditioning_scale = scales[1]
#     phrase_idxes = [get_phrases_idx(pipe.tokenizer, phrases[0], prompt)]
#     eot_idxes = [[get_eot_idx(pipe.tokenizer, prompt)] * len(phrases[0])]
#     print(phrase_idxes, eot_idxes)

#     images = ms_model.generate(pipe=pipe, pil_images=[input_images], num_samples=num_samples, num_inference_steps=30, seed=0,
#                            prompt=[prompt], scale=ms_scale, image_encoder=image_encoder, image_processor=image_processor, boxes=boxes,
#                            image_proj_type=image_proj_type, image_encoder_type=image_encoder_type, phrases=phrases, drop_grounding_tokens=drop_grounding_tokens,
#                            phrase_idxes=phrase_idxes, eot_idxes=eot_idxes, height=1024, width=1024, 
#                            image=control_image, controlnet_conditioning_scale=controlnet_conditioning_scale)

#     # save_name = "only_depth"
#     # save_name = "only_soft_edge"
#     save_name = "only_canny"
#     save_path = os.path.join(result_path, log_id, load_type, save_name)
#     os.makedirs(save_path, exist_ok=True)
#     for j, image in enumerate(images):
#         image.save(os.path.join(save_path, f"scales{c}.jpg"))

ms_controlnet_scales = [[0.6, 0.7], [0.6, 0.8], [0.4, 0.9]]
# ms_controlnet_scales = [[0.6, 0.7]]
for c, scales in enumerate(ms_controlnet_scales):
    ms_scale = scales[0]
    controlnet_conditioning_scale = scales[1]
    phrase_idxes = [get_phrases_idx(pipe.tokenizer, phrases[0], prompt)]
    eot_idxes = [[get_eot_idx(pipe.tokenizer, prompt)] * len(phrases[0])]
    print(phrase_idxes, eot_idxes)

    images = ms_model.generate(pipe=pipe, pil_images=[input_images], num_samples=num_samples, num_inference_steps=30, seed=0,
                           prompt=[prompt], scale=ms_scale, image_encoder=image_encoder, image_processor=image_processor, boxes=boxes,
                           image_proj_type=image_proj_type, image_encoder_type=image_encoder_type, phrases=phrases, drop_grounding_tokens=drop_grounding_tokens,
                           phrase_idxes=phrase_idxes, eot_idxes=eot_idxes, height=1024, width=1024, 
                           image=control_image, controlnet_conditioning_scale=controlnet_conditioning_scale)

    # save_name = "only_depth"
    # save_name = "only_soft_edge"
    # save_name = "only_canny"
    save_name = "only_softedge"
    save_path = os.path.join(result_path, log_id, load_type, save_name)
    os.makedirs(save_path, exist_ok=True)
    for j, image in enumerate(images):
        image.save(os.path.join(save_path, f"Qwen_boxes_scales{c}.jpg"))
    
    
    # for i in range(2):
    #     boxes = boxesList[i]
    #     # used to get the attention map, return zero if the phrase is not in the prompt
    #     phrase_idxes = [get_phrases_idx(pipe.tokenizer, phrases[0], prompt)]
    #     eot_idxes = [[get_eot_idx(pipe.tokenizer, prompt)] * len(phrases[0])]
    #     print(phrase_idxes, eot_idxes)

    #     images = ms_model.generate(pipe=pipe, pil_images=[input_images], num_samples=num_samples, num_inference_steps=30, seed=0,
    #                        prompt=[prompt], scale=ms_scale, image_encoder=image_encoder, image_processor=image_processor, boxes=boxes,
    #                        image_proj_type=image_proj_type, image_encoder_type=image_encoder_type, phrases=phrases, drop_grounding_tokens=drop_grounding_tokens,
    #                        phrase_idxes=phrase_idxes, eot_idxes=eot_idxes, height=1024, width=1024, 
    #                        image=control_image, controlnet_conditioning_scale=controlnet_conditioning_scale)

    #     save_name = "only_depth"
    #     save_path = os.path.join(result_path, log_id, load_type, save_name)
    #     os.makedirs(save_path, exist_ok=True)
    #     for j, image in enumerate(images):
    #         image.save(os.path.join(save_path, f"scales{c}_boxes{i}.jpg"))

