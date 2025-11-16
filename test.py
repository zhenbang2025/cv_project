import os
import re
import json
from PIL import Image
from transformers import pipeline
import torch

# -------------------------- 原有生图相关配置（保留你的全部参数） --------------------------
# 图像输入（多图+预处理）
image0 = Image.open("./examples/1.png")
image1 = Image.open("./examples/2.png")
input_images = [image0, image1]
input_images = [x.convert("RGB").resize((512, 512)) for x in input_images]

# 生图核心参数（布局引导+主体定义）
num_samples = 5  # 每次生成5张图
boxes = [[[0., 0.25, 0.4, 0.75], [0.6, 0.25, 1., 0.75]]]  # dog(左) + cat(右) 布局
phrases = [["dog", "cat"]]  # 对应布局的主体
drop_grounding_tokens = [0]  # 不丢弃布局引导token
height = 1024
width = 1024
num_inference_steps = 30
seed = 0
scale = 0.6
image_proj_type = "your_image_proj_type"  # 替换为你的实际参数（如"clip"）
image_encoder_type = "your_image_encoder_type"  # 替换为你的实际参数（如"vit"）

# 结果保存路径
result_path = "./generation_results"
log_id = "iterative_optimization"
load_type = "multi_image_layout"
os.makedirs(os.path.join(result_path, log_id, load_type), exist_ok=True)

# -------------------------- 新增：LLaVA配置 + 迭代参数 --------------------------
# LLaVA-1.5 初始化（本地开源VLM，评估图像+布局+Prompt匹配）
llava_pipe = pipeline(
    "image-text-to-text",
    model="llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"  # 自动分配GPU/CPU（显存不足用4-bit量化版，见之前优化建议）
)

# 迭代配置
MAX_ITERATIONS = 3  # 最大迭代次数
INITIAL_PROMPT = "best quality, high quality, a dog and cat on the beach"  # 初始Prompt
# INITIAL_PROMPT = "The figurine hold the crystal ball in its gowpen, while dancing on the volcano."  # 可选初始Prompt
NEGATIVE_PROMPT = "blurry, low quality, ugly, distorted, dark, wrong layout, missing subject"  # 新增"布局错误/缺失主体"负面Prompt
ACCEPT_THRESHOLD = 8  # 匹配度≥8分视为合格

# -------------------------- 原有生图辅助函数（保留） --------------------------
def get_phrases_idx(tokenizer, phrases, prompt):
    """获取phrases在prompt中的token索引（你的原有逻辑）"""
    prompt_tokens = tokenizer.tokenize(prompt)
    phrase_idxes = []
    for phrase in phrases:
        phrase_tokens = tokenizer.tokenize(phrase)
        # 简化匹配逻辑（请替换为你的实际实现）
        for i in range(len(prompt_tokens) - len(phrase_tokens) + 1):
            if prompt_tokens[i:i+len(phrase_tokens)] == phrase_tokens:
                phrase_idxes.append((i, i+len(phrase_tokens)-1))
                break
    return phrase_idxes

def get_eot_idx(tokenizer, prompt):
    """获取EOT token索引（你的原有逻辑）"""
    return len(tokenizer.tokenize(prompt))

# -------------------------- 新增：适配布局引导的生图函数 --------------------------
def generate_image_with_layout(current_prompt, pipe, image_encoder, image_processor):
    """结合多图输入+布局引导生成图像（复用你的生图逻辑）"""
    # 计算phrase索引和EOT索引（基于当前优化后的Prompt）
    phrase_idxes = [get_phrases_idx(pipe.tokenizer, phrases[0], current_prompt)]
    eot_idxes = [[get_eot_idx(pipe.tokenizer, current_prompt)] * len(phrases[0])]
    print(f"当前Prompt的phrase索引：{phrase_idxes}, EOT索引：{eot_idxes}")

    try:
        # 调用你的ms_model生成图像（参数完全复用你的配置）
        images = ms_model.generate(
            pipe=pipe,
            pil_images=[input_images],
            num_samples=num_samples,
            num_inference_steps=num_inference_steps,
            seed=seed,
            prompt=[current_prompt],
            negative_prompt=[NEGATIVE_PROMPT],  # 新增负面Prompt
            scale=scale,
            image_encoder=image_encoder,
            image_processor=image_processor,
            boxes=boxes,
            image_proj_type=image_proj_type,
            image_encoder_type=image_encoder_type,
            phrases=phrases,
            drop_grounding_tokens=drop_grounding_tokens,
            phrase_idxes=phrase_idxes,
            eot_idxes=eot_idxes,
            height=height,
            width=width
        )
        return images  # 返回生成的num_samples张图像
    except Exception as e:
        print(f"生图失败（布局引导模式）：{e}")
        return None

# -------------------------- 新增：适配布局引导的LLaVA评估+Prompt优化 --------------------------
def evaluate_and_optimize_prompt_with_layout(generated_images, current_prompt):
    """LLaVA评估（含布局+主体+Prompt匹配）并优化Prompt"""
    # 选择生成的第一张图进行评估（也可选择最优图，见扩展建议）
    eval_image = generated_images[0]

    # 评估指令（重点：加入布局和主体检查，适配你的boxes/phrases）
    eval_prompt = f"""
    任务：评估图像是否符合Prompt、布局要求和主体定义，然后优化Prompt。
    1. 当前Prompt：{current_prompt}
    2. 布局要求：左半部分（图像0-40%宽度）显示"dog"，右半部分（60%-100%宽度）显示"cat"
    3. 主体要求：必须包含"dog"和"cat"两个主体，且清晰可辨
    4. 负面要求：{NEGATIVE_PROMPT}
    5. 评分标准：0-10分（10分完全匹配），≥{ACCEPT_THRESHOLD}分为合格

    请严格按以下JSON格式输出，不要添加任何额外文字：
    {{
        "match_score": 整数分数,
        "is_acceptable": true/false,
        "feedback": 详细反馈（需说明：1.Prompt匹配度；2.布局是否符合要求；3.主体是否完整）,
        "optimized_prompt": 优化后的Prompt（核心要求：保留"dog+cat+beach"主题和左右布局，补充细节/调整风格，长度≤60词）
    }}
    """

    try:
        # 调用LLaVA评估（直接传入PIL图像）
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": eval_image},
                    {"type": "text", "text": eval_prompt}
                ]
            }
        ]

        out = llava_pipe(text=messages, max_new_tokens=350, temperature=0.2)
        generated_text = out[0]["generated_text"].strip()

        # 解析JSON（容错处理LLaVA的格式偏差）
        json_match = re.search(r"\{.*\}", generated_text, re.DOTALL)
        if not json_match:
            raise ValueError(f"LLaVA输出格式错误：{generated_text}")
        
        eval_result = json.loads(json_match.group())
        return eval_result, eval_image  # 返回评估结果+用于评估的图像
    except Exception as e:
        print(f"LLaVA评估失败：{e}")
        # 容错：返回默认结果继续迭代
        return {
            "match_score": 5,
            "is_acceptable": False,
            "feedback": "LLaVA评估失败，默认优化Prompt",
            "optimized_prompt": f"{current_prompt}, clear layout (dog left, cat right), detailed fur, vibrant beach scenery, high resolution"
        }, eval_image

def iterative_layout_guided_generation(initial_prompt, pipe, image_encoder, image_processor):
    current_prompt = initial_prompt
    iteration = 1

    while iteration <= MAX_ITERATIONS:
        print(f"\n===== 迭代 {iteration}/{MAX_ITERATIONS} =====")
        print(f"当前Prompt：{current_prompt}")

        # 1. 多图+布局引导生图
        generated_images = generate_image_with_layout(current_prompt, pipe, image_encoder, image_processor)
        if not generated_images:
            print("生图失败，终止迭代")
            return None

        # 保存本次所有生成的图像
        save_path = os.path.join(result_path, log_id, load_type, f"iteration_{iteration}")
        os.makedirs(save_path, exist_ok=True)
        for i, img in enumerate(generated_images):
            img.save(os.path.join(save_path, f"sample_{i}.jpg"))
        print(f"本次生成的{num_samples}张图像已保存至：{save_path}")

        # 2. LLaVA评估（用第一张图）+ 优化Prompt
        eval_result, eval_image = evaluate_and_optimize_prompt_with_layout(generated_images, current_prompt)
        print(f"匹配度：{eval_result['match_score']}/10")
        print(f"评估反馈：{eval_result['feedback']}")

        # 3. 保存评估用的图像（标记为当前迭代的代表图）
        eval_image.save(os.path.join(save_path, "eval_sample.jpg"))
        print(f"评估用图像已保存至：{os.path.join(save_path, 'eval_sample.jpg')}")

        # 4. 判断是否合格
        if eval_result["is_acceptable"]:
            print(f"\n✅ 迭代成功！最终Prompt：{current_prompt}")
            print(f"最终图像路径：{save_path}")
            return generated_images, current_prompt

        # 5. 更新Prompt继续迭代
        current_prompt = eval_result["optimized_prompt"]
        print(f"优化后Prompt：{current_prompt}")

        iteration += 1

    print(f"\n❌ 已达最大迭代次数，输出最后一次生成结果")
    return generated_images, current_prompt

# -------------------------- 执行主流程（需传入你的pipe/image_encoder/image_processor） --------------------------
if __name__ == "__main__":
    # 注意：需替换为你的实际SD pipe、image_encoder、image_processor（原有生图逻辑中的实例）
    # 示例：假设你已初始化好以下组件（请根据你的实际代码调整）
    # from your_sd_pipeline import pipe  # 你的SD pipe（含tokenizer）
    # from your_image_encoder import image_encoder, image_processor  # 你的图像编码器和处理器
    # from your_ms_model import ms_model  # 你的ms_model生图模型

    # 执行迭代生图（请确保上述组件已正确初始化）
    final_images, final_prompt = iterative_layout_guided_generation(
        initial_prompt=INITIAL_PROMPT,
        pipe=pipe,  # 你的SD pipe实例
        image_encoder=image_encoder,  # 你的图像编码器实例
        image_processor=image_processor  # 你的图像处理器实例
    )