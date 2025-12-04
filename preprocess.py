import os
from huggingface_hub import InferenceClient
from PIL import Image
from ultralytics import YOLO
from controlnet_aux import ZoeDetector, CannyDetector, HEDdetector

# 1. 设置路径 (建议使用当前目录，避免权限问题)
base_path = './examples'
os.makedirs(base_path, exist_ok=True) # 自动创建文件夹

input_image_filename = 'base_image_v5.png'
input_image_path = os.path.join(base_path, input_image_filename)

output_depth_path = os.path.join(base_path, 'depth_v5.png')
output_canny_path = os.path.join(base_path, 'canny_v5.png')
output_softedge_path = os.path.join(base_path, 'soft_edge_v5.png')
output_yolo_path = os.path.join(base_path, 'yolo_result_v5.png')

# 2. 图像获取逻辑
# 检查本地是否有图，没有则调用 API 生成，避免重复消耗 token
if not os.path.exists(input_image_path):
    print("本地未找到图像，正在调用 FLUX.1-dev 生成...")
    client = InferenceClient(
        provider="nebius",
        api_key=os.environ["HF_TOKEN"],
    )
    
    try:
        # 生成图像
        base_image = client.text_to_image(
            "Astronaut riding a horse, high quality, photorealistic",
            model="black-forest-labs/FLUX.1-dev",
        )
        # 保存生成的原图！这一步很重要
        base_image.save(input_image_path)
        print(f"图像已生成并保存至: {input_image_path}")
    except Exception as e:
        print(f"API 生成图像失败: {e}")
        exit() # 如果生成失败，无法继续后续步骤
else:
    print(f"加载本地现有图像: {input_image_path}")
    base_image = Image.open(input_image_path).convert("RGB")

# 3. 处理图像 (ControlNet Preprocessors)

# --- 深度图 (Zoe Depth) ---
try:
    print("正在加载 ZoeDetector 并生成深度图...")
    depth_estimator = ZoeDetector.from_pretrained("lllyasviel/Annotators")
    depth_map = depth_estimator(base_image)
    # 修复：保存深度图
    depth_map.save(output_depth_path)
    print(f"深度图已保存: {output_depth_path}")
except Exception as e:
    print(f"生成深度图失败: {e}")

# --- Canny 边缘图 ---
try:
    print("正在生成 Canny 边缘图...")
    canny_detector = CannyDetector()
    canny_image = canny_detector(base_image, low_threshold=100, high_threshold=200)
    canny_image.save(output_canny_path)
    print(f"Canny 图已保存: {output_canny_path}")
except Exception as e:
    print(f"生成 Canny 图失败: {e}")

# --- Soft Edge (HED) ---
try:
    print("正在加载 HEDdetector 并生成 Soft Edge 图...")
    hed_detector = HEDdetector.from_pretrained('lllyasviel/Annotators')
    soft_edge_image = hed_detector(base_image)
    soft_edge_image.save(output_softedge_path)
    print(f"Soft Edge 图已保存: {output_softedge_path}")
except Exception as e:
    print(f"生成 Soft Edge 图失败: {e}")

# YOLOv8 对象检测
try:
    # 加载模型
    model = YOLO("yolov8m.pt")

    # conf=0.3 过滤低置信度
    results = model(base_image, conf=0.3, save=False, verbose=False)

    boxes_rel = []
    names = []
    
    # 获取图像尺寸用于归一化
    w, h = base_image.size
    
    # COCO数据集常见类别: 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'...
    target_classes = ["person", "horse", "cat", "dog"] 

    # 解析结果
    result = results[0] # 第一张图的结果
    
    if result.boxes:
        for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            label = model.names[int(cls)]
            
            # 筛选特定类别
            if label in target_classes:
                x1, y1, x2, y2 = box
                # 计算相对坐标 (0-1)
                rel_box = [round(x1/w, 4), round(y1/h, 4), round(x2/w, 4), round(y2/h, 4)]
                
                boxes_rel.append(rel_box)
                names.append(f"{label} ({conf:.2f})")

    print(f"   检测到的对象: {names}")
    print(f"   相对坐标 Box (xyxy 0-1): {boxes_rel}")

    # --- 可视化与保存 ---
    # plot() 返回的是 BGR 格式的 numpy 数组
    annotated_bgr = result.plot() 
    
    # 转换 BGR -> RGB 以便 PIL 保存和 Matplotlib 显示
    annotated_rgb = annotated_bgr[..., ::-1]
    
    # 保存检测结果图
    Image.fromarray(annotated_rgb).save(output_yolo_path)
    print(f"   YOLO 结果图已保存 -> {output_yolo_path}")

except Exception as e:
    print(f"   !! YOLO 检测过程发生错误: {e}")

