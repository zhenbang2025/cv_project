import os
import torch
import json
import numpy as np
import re
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from transformers import CLIPVisionModelWithProjection
from transformers import CLIPImageProcessor
from transformers import pipeline

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


base_model_path = "/home/rjiangas/models/stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "/home/rjiangas/models/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
device = "cuda"
result_path = "./res"
log_id = "test"
load_type = "/home/rjiangas/models/doge1516/MS-Diffusion/ms_adapter.bin"
ms_ckpt = f"/home/rjiangas/models/doge1516/MS-Diffusion/ms_adapter.bin"

image_processor = CLIPImageProcessor()

# load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.to(device)

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


llava_pipe = pipeline("image-text-to-text", model="/home/rjiangas/models/llava-hf/llava-1.5-7b-hf",device_map="cuda" )

image0 = Image.open("./examples/1.png")
# image1 = Image.open("./examples/example_dog.jpg")
input_images = [image0]
# input_images = [image0, image1]

# generation configs
num_samples = 5
ACCEPT_THRESHOLD = 8
prompt = "A figurine raises the hands"
# prompt = "The cat and the dog are playing by the seaside, with the cat's paw stroking the dog's head."
print(prompt)
# boxes = [[[0.25, 0.25, 0.75, 0.75]]]  # dog
# boxes = [[[0., 0.25, 0.4, 0.75], [0.6, 0.25, 1., 0.75]]]  # dog+cat
# boxes = [[[0., 0., 0., 0.], [0., 0., 0., 0.]]]  # used if you want no layout guidance
boxes = [[[0., 0., 0., 0.]]]  # used if you want no layout guidance
# phrases = [["dog"]]
phrases = [["figurine"]]
MAX_ITERATIONS = 4


def generate_image(prompt,boxes,phrases,input_images):
    input_images = [x.convert("RGB").resize((512, 512)) for x in input_images]
    drop_grounding_tokens = [0]  # set to 1 if you want to drop the grounding tokens

    # used to get the attention map, return zero if the phrase is not in the prompt
    phrase_idxes = [get_phrases_idx(pipe.tokenizer, phrases[0], prompt)]
    eot_idxes = [[get_eot_idx(pipe.tokenizer, prompt)] * len(phrases[0])]
    print(phrase_idxes, eot_idxes)

    images = ms_model.generate(pipe=pipe, pil_images=[input_images], num_samples=num_samples, num_inference_steps=30, seed=0,
                            prompt=[prompt], scale=0.6, image_encoder=image_encoder, image_processor=image_processor, boxes=boxes,
                            image_proj_type=image_proj_type, image_encoder_type=image_encoder_type, phrases=phrases, drop_grounding_tokens=drop_grounding_tokens,
                            phrase_idxes=phrase_idxes, eot_idxes=eot_idxes, height=1024, width=1024)

    # save_path = os.path.join(save_path, save_name)
    # os.makedirs(save_path, exist_ok=True)
    # for i, image in enumerate(images):
    #     image.save(os.path.join(save_path, f"{i}.jpg"))

    return images
    


def parse_boxes_and_phrases(boxes, phrases):
    """
    解析boxes（归一化坐标[x1, y1, x2, y2]）和phrases，生成评估规则：
    返回：(主体要求字符串, 布局要求字符串)
    """
    # 提取当前批次的boxes和phrases（支持单批次多主体）
    batch_boxes = boxes[0]  # boxes格式：[[[x1,y1,x2,y2], ...]]
    batch_phrases = phrases[0]  # phrases格式：[[subject1, subject2, ...]]
    
    # 校验：boxes和phrases数量必须一致
    if len(batch_boxes) != len(batch_phrases):
        raise ValueError(f"boxes数量（{len(batch_boxes)}）与phrases数量（{len(batch_phrases)}）不匹配！")
    
    # 1. 生成主体要求（必须包含所有phrases，且清晰可辨）
    subject_requirement = f"必须包含以下所有主体，且每个主体清晰可辨：{', '.join(batch_phrases)}"
    
    # 2. 生成布局要求（将归一化坐标转换为人类可理解的位置描述）
    layout_requirements = []
    for idx, (box, subject) in enumerate(zip(batch_boxes, batch_phrases)):
        x1, y1, x2, y2 = box
        # 计算宽度占比（x轴：0=左边界，1=右边界）
        box_width = x2 - x1
        box_center_x = (x1 + x2) / 2
        
        # 转换x坐标为位置描述（左/中/右/具体占比）
        if box_center_x < 0.3:
            pos_x = "左半部分"
        elif box_center_x < 0.7:
            pos_x = "中间部分"
        else:
            pos_x = "右半部分"
        
        # 补充具体占比（更精准）
        pos_detail = f"（图像{x1:.0%}-{x2:.0%}宽度）"
        layout_requirements.append(f"'{subject}' 应位于 {pos_x}{pos_detail}")
    
    layout_requirement = "布局要求：" + "；".join(layout_requirements)
    
    return subject_requirement, layout_requirement


def evaluate_and_optimize_prompt_with_layout(generated_images, current_prompt):
    """LLaVA Evaluation (Layout + Subject + Action + Prompt Matching) & Prompt Optimization (English for better accuracy)"""
    eval_image = generated_images[0]

    # Dynamically parse boxes and phrases to generate layout requirements (keep flexibility)
    subject_req, layout_req = parse_boxes_and_phrases(boxes, phrases)
    # Convert layout requirements to English (align with LLaVA's training data)

    # Evaluation prompt (full English, optimized for LLaVA's understanding)
    eval_prompt = f"""
    Task: Evaluate if the image matches the Prompt, layout requirements, subject definitions, action interaction, and scene, then optimize the Prompt.
    1. Current Prompt: {current_prompt}
    2. Subject Requirement: Must include one subjects ("figurine"), both clearly visible, no missing or blurry details.
    3. Action Interaction Requirement: Must accurately present "the figurine next to the crystal ball and raises right hand" — the figurine must raises right handand the action is natural and distinguishable.
    5. Scoring Standard: 0-10 points (10 points = perfect match), ≥{ACCEPT_THRESHOLD} points = acceptable.

    Please output strictly in the following JSON format, NO extra text outside JSON:
    {{
        "match_score": integer score,
        "is_acceptable": true/false,
        "feedback": "Detailed feedback (must cover 5 aspects: 1. Overall Prompt matching; 2. Whether layout meets requirements; 3. Whether subjects are complete and clear; 4. Whether action interaction is accurately implemented; 5. Whether the scene is seaside)",
        "optimized_prompt": "Optimized Prompt (core rules: 1. Keep the core elements 'figurine+the figurine raises right hand'; 2. Add details for action, fur texture; 3. Length ≤ 80 words; 4. Do not change layout or core interaction)"
    }}
    """

    try:
        # Call LLaVA with English messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": eval_image},
                    {"type": "text", "text": eval_prompt.strip()}  # Remove extra newlines for stability
                ]
            }
        ]

        out = llava_pipe(text=messages, max_new_tokens=450, temperature=0.15)  # Lower temp for stability
        generated_text = out[0]["generated_text"][1]['content']
        print(f"LLaVA Raw Output (English): {generated_text}")  # Debug: check raw output

        # Parse JSON (fault-tolerant handling for minor format deviations)
        json_match = re.search(r"\{.*\}", generated_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"LLaVA output does not contain valid JSON: {generated_text}")
        
        eval_result = json.loads(json_match.group())
        # Ensure feedback and optimized_prompt are in English (LLaVA will naturally output English)
        return eval_result, eval_image

    except Exception as e:
        print(f"LLaVA Evaluation Failed: {e}")
        # Fallback optimized prompt (English, keep core elements)
        # default_optimized_prompt = (
        #     f"{current_prompt}, cat's paw gently stroking dog's head, detailed fur texture, "
        #     f"sunny seaside with golden sand and calm ocean waves, sharp focus, realistic lighting, high resolution"
        # )
        return {
            "match_score": 5,
            "is_acceptable": False,
            "feedback": f"LLaVA evaluation failed: {str(e)}. Fallback to default optimization (added action and environment details).",
            "optimized_prompt": default_optimized_prompt[:80]  # Limit length
        }, eval_image
    
    
def iterative_layout_guided_generation(initial_prompt, boxes,phrases,save_path,save_name):
    current_prompt = initial_prompt
    iteration = 1

    while iteration <= MAX_ITERATIONS:
        print(f"\n===== 迭代 {iteration}/{MAX_ITERATIONS} =====")
        print(f"当前Prompt：{current_prompt}")

        generated_images = generate_image(current_prompt, boxes, phrases,input_images)
        if not generated_images:
            print("生图失败，终止迭代")
            print("Can not find files in ",os.path.join(save_path,save_name))
            return None

        save_path_current = os.path.join(save_path, f"iteration_{iteration}")
        os.makedirs(save_path_current, exist_ok=True)
        for i, img in enumerate(generated_images):
            img.save(os.path.join(save_path_current, f"sample_{i}.jpg"))
        print(f"本次生成的{num_samples}张图像已保存至：{save_path_current}")

        # 2. LLaVA评估（用第一张图）+ 优化Prompt
        eval_result, eval_image = evaluate_and_optimize_prompt_with_layout(generated_images, current_prompt)
        print(f"匹配度：{eval_result['match_score']}/10")
        print(f"评估反馈：{eval_result['feedback']}")

        # 3. 保存评估用的图像（标记为当前迭代的代表图）
        eval_image.save(os.path.join(save_path_current, "eval_sample.jpg"))
        print(f"评估用图像已保存至：{os.path.join(save_path_current, 'eval_sample.jpg')}")

        # 4. 判断是否合格
        if eval_result["is_acceptable"]:
            print(f"\n迭代成功！最终Prompt：{current_prompt}")
            print(f"最终图像路径：{save_path_current}")
            return generated_images, current_prompt

        # 5. 更新Prompt继续迭代
        current_prompt = eval_result["optimized_prompt"]
        print(f"优化后Prompt：{current_prompt}")

        iteration += 1

    print(f"\n 已达最大迭代次数，输出最后一次生成结果")
    return generated_images, current_prompt

if __name__ == "__main__":
    save_path = '/home/rjiangas/cv_project/MS-Diffusion/res/llava'
    save_name = 'llava_figurine'
    final_images, final_prompt = iterative_layout_guided_generation(
        initial_prompt=prompt,
        boxes=boxes,
        phrases=phrases,
        save_path=save_path,
        save_name=save_name,

    )
    # img = Image.open("/home/rjiangas/cv_project/MS-Diffusion/res/test/models/doge1516/MS-Diffusion/ms_adapter.bin/dog_cat/0.jpg")
    # img = np.array([img])
    # eval_result, eval_image = evaluate_and_optimize_prompt_with_layout(img, prompt)