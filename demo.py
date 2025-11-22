import gradio as gr
import os
import time
from PIL import Image
import numpy as np
import cv2
from controlnet_aux import CannyDetector, HEDdetector, OpenposeDetector

# ----------------- 核心配置 -----------------
# 假设你的 inference_controlnet.py 和相关模型在同一目录或Python路径下
import inference_controlnet

# 定义生成结果的保存根目录
RESULT_ROOT_PATH = './res/gradio_demo_results'
os.makedirs(RESULT_ROOT_PATH, exist_ok=True)

# 定义 ControlNet 类型选项
CONTROLNET_TYPES = [
    "canny",
    "openpose"
]

# ----------------- 辅助函数 -----------------

def preprocess_control_image(image, control_type):
    """
    根据选择的 ControlNet 类型，对输入的控制图进行预处理。
    """
    if image is None:
        return None
        
    image_np = np.array(image)
    processed_image = None

    if control_type == "canny":
        print("Preprocessing control image with Canny...")
        canny_detector = CannyDetector()
        processed_image = canny_detector(image_np, low_threshold=100, high_threshold=200)
    elif control_type == "openpose":
        print("Preprocessing control image with OpenPose...")
        openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        # openpose_detector 返回一个PIL Image
        processed_image = openpose_detector(image_np)
    
    # 确保返回一个PIL Image
    if processed_image is not None and not isinstance(processed_image, Image.Image):
        processed_image = Image.fromarray(processed_image)
        
    return processed_image

# ----------------- Gradio 核心函数 -----------------

def generate(
    object_images, 
    control_image, 
    prompt, 
    controlnet_type, 
    object_names, 
    progress=gr.Progress(track_tqdm=True)
):
    """
    当用户点击生成按钮时调用的主函数。
    """
    status_messages = []
    status_messages.append("🔍 Starting image generation process...")
    progress(0, desc=status_messages[-1])
    time.sleep(1)

    # 1. 输入验证
    if not object_images:
        error_msg = "❌ Error: Please upload at least one object image."
        status_messages.append(error_msg)
        return None, "\n".join(status_messages)
    
    if not prompt:
        prompt = "A photo of the given objects arranged in a scene."
        status_messages.append(f"⚠️ Warning: No prompt provided. Using default: '{prompt}'")
        progress(0.1, desc=status_messages[-1])
        time.sleep(0.5)

    # 2. 准备 phrases
    phrases = []
    if object_names:
        names_list = [name.strip() for name in object_names.split(',') if name.strip()]
        if len(names_list) == len(object_images):
            phrases = [names_list]
            status_messages.append(f"📝 Using object names: {names_list}")
        else:
            status_messages.append(f"⚠️ Warning: Number of object names ({len(names_list)}) does not match number of images ({len(object_images)}). Using default phrases.")
            phrases = [[f"object_{i+1}" for i in range(len(object_images))]]
    else:
        status_messages.append("📝 No object names provided. Using default phrases.")
        phrases = [[f"object_{i+1}" for i in range(len(object_images))]]
    
    status_messages.append(f"📋 Prepared phrases: {phrases}")
    progress(0.2, desc=status_messages[-1])
    time.sleep(0.5)

    # 3. 准备 control_image
    final_control_image = None
    if control_image:
        status_messages.append(f"🎨 Preprocessing control image with {controlnet_type}...")
        progress(0.3, desc=status_messages[-1])
        final_control_image = preprocess_control_image(control_image, controlnet_type)
        if final_control_image:
            status_messages.append("✅ Control image preprocessing completed.")
        else:
            status_messages.append("❌ Failed to preprocess control image. Proceeding without it...")
    else:
        status_messages.append("ℹ️ No control image provided. Proceeding without ControlNet...")
    
    progress(0.5, desc=status_messages[-1])
    time.sleep(0.5)

    # 4. 准备 boxes (核心简化点)
    # 假设每个物体都占据整个画面，或者让AI决定
    boxes = [[[0.0, 0.0, 1.0, 1.0] for _ in object_images]]
    status_messages.append(f"📦 Using default boxes for objects: {boxes}")
    progress(0.6, desc="🚀 Calling MS-SD generation function...")
    time.sleep(0.5)

    # 5. 准备其他参数
    controlnet_conditioning_scale = 0.9
    msadapter_scale = 0.9
    
    # 6. 调用生成函数
    try:
        # 为每次生成创建一个唯一的保存路径
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_path = os.path.join(RESULT_ROOT_PATH, timestamp)
        os.makedirs(result_path, exist_ok=True)
        
        save_name = f"generated_image_cn_{controlnet_type}_cs_{controlnet_conditioning_scale}_ms_{msadapter_scale}.jpg"
        
        status_messages.append(f"💾 Will save result to: {os.path.join(result_path, save_name)}")
        
        # 调用核心生成函数
        inference_controlnet.ms_sd_generate_image(
            input_images=object_images,
            prompt=prompt,
            phrases=phrases,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            msadapter_scale=msadapter_scale,
            control_image=final_control_image,
            boxes=boxes,
            result_path=result_path,
            save_name=save_name
        )
        
        status_messages.append("🎉 Image generation completed successfully!")
        progress(1, desc=status_messages[-1])
        
        # 返回生成的图片路径和状态信息
        generated_image_path = os.path.join(result_path, save_name)
        return generated_image_path, "\n".join(status_messages)

    except Exception as e:
        error_msg = f"💥 Error during image generation: {e}"
        status_messages.append(error_msg)
        return None, "\n".join(status_messages)


with gr.Blocks(title="ControlNet Object Composer") as demo:
    gr.Markdown(
        """
        # 🎭 ControlNet Object Composer Demo
        
        This demo allows you to combine multiple object images into a single coherent scene using a text prompt and a control image (for ControlNet guidance).
        
        **How to use:**
        1. **Upload Object Images**: Upload one or more images of objects you want to combine.
        2. **Upload Control Image**: (Optional) Upload an image to guide the composition (e.g., a pose for a person, or a scene layout for Canny edges).
        3. **Enter Prompt**: Describe the desired final scene in detail.
        4. **Select ControlNet Type**: Choose the type of ControlNet to use (if a control image is provided).
        5. **Enter Object Names**: (Optional) Provide comma-separated names for your objects (e.g., `shirt, shoes, hat`). The order must match the uploaded images.
        6. **Click "Generate"** and wait for the magic!
        """
    )
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Object Images")
            object_input = gr.Files(
                label="Upload one or more object images (PNG with transparent background recommended)",
                file_types=["image"]
            )
            
            gr.Markdown("### 🎨 Control Image (Optional)")
            control_input = gr.Image(
                label="Upload a control image (e.g., pose, scene sketch)",
                type="pil"
            )
            
            controlnet_type_dropdown = gr.Dropdown(
                choices=CONTROLNET_TYPES,
                value=CONTROLNET_TYPES[0],
                label="ControlNet Type",
                info="Select the preprocessing method for the control image."
            )

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Prompt")
            prompt_input = gr.Textbox(
                label="Describe your desired scene",
                lines=3,
                placeholder="e.g., A woman wearing a blue shirt and white skirt, carrying a black handbag, standing in a park."
            )
            
            gr.Markdown("### 🏷️ Object Names (Optional)")
            object_names_input = gr.Textbox(
                label="Comma-separated names for objects (order must match images)",
                placeholder="e.g., blue shirt, white skirt, black handbag"
            )
            
            submit_btn = gr.Button("🚀 Generate Combined Image", variant="primary")
            
            gr.Markdown("### 📊 Generation Status")
            status_output = gr.Textbox(
                label="Process Log",
                interactive=False,
                lines=10
            )

    with gr.Row():
        gr.Markdown("### 🖼️ Generated Result")
        image_output = gr.Image(
            label="The final composed image",
            height=512
        )

    # 绑定按钮点击事件
    submit_btn.click(
        fn=generate,
        inputs=[object_input, control_input, prompt_input, controlnet_type_dropdown, object_names_input],
        outputs=[image_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)