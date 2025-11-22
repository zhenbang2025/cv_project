from rembg import remove
from PIL import Image
import os
import cv2
from controlnet_aux import OpenposeDetector

output_dir = "./examples"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def remove_background(image_path, output_suffix="_nobg"):
    try:
        file_name = os.path.basename(image_path)
        name, ext = os.path.splitext(file_name)
        
        output_path = os.path.join(output_dir, f"{name}{output_suffix}.png")
        
        print(f"正在处理: {file_name}")
        input_image = Image.open(image_path)
        output_image = remove(input_image)
        
        output_image.save(output_path, "PNG")
        print(f"处理完成: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"✗ 处理失败 {file_name}: {e}")
        return None


def process_image(image_path, output_path):
    try:
        openpose = OpenposeDetector.from_pretrained("/home/rjiangas/models/lllyasviel/ControlNet")
        base_image = Image.open(image_path).convert("RGB")
        if base_image is None:
            print(f"无法读取图像: {image_path}")
            return
        pose_image = openpose(base_image)

        pose_image.save(output_path)
        print(f"姿态图已成功保存到: {output_path}")
        
    except Exception as e:
        print(f"错误: {e}")

    
    
if __name__ == "__main__":
    # files = ["./examples/bench.png","./examples/luggage.jpeg","./examples/bag.png","./examples/model.png"]
    # for file in files:
    #     remove_background(file)

    process_image("./examples/pose_girl.png", "./res/openpose/open_pose_girl.jpg")


    
    
    