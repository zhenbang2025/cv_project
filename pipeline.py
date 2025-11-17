
import inference_controlnet
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from controlnet_aux import CannyDetector, HEDdetector
import os
from ultralytics import YOLO

image0 = Image.open("./examples/example_dog.jpg")
image1 = Image.open("./examples/example_cat.jpg")
pose_images_dir = "./examples/canny_yolo_pipeline_dataset"
position_image_paths = []
items = os.listdir(pose_images_dir)
for item in items:
    position_image_path = os.path.join(pose_images_dir, item)
    position_image_paths.append(position_image_path)
output_canny_dir = './res/canny'

# # generate canny 边缘图
print("正在生成 Canny 边缘图...")
try:
    for i,position_image_path in enumerate(position_image_paths):
        position_image = Image.open(os.path.join(pose_images_dir, position_image_path)).convert("RGB")
        canny_detector = CannyDetector()
        canny_image = canny_detector(position_image, low_threshold=100, high_threshold=200)
        os.makedirs(output_canny_dir, exist_ok=True)
        output_canny_path = os.path.join(output_canny_dir,f"dog_cat_{i}.jpg")
        canny_image.save(output_canny_path)
        print(f"Canny 边缘图已成功保存到: {output_canny_path}")
except Exception as e:
    print(f"生成 Canny 边缘图时发生错误: {e}")


# box generation
model = YOLO("/home/rjiangas/models/youlov8/yolov8m.pt")
labels = ["cat", "dog"]
yolo_detect_result_dir = './res/yolo_detect'
boxes_total = []
for i, base_path in enumerate(position_image_paths):
    image = Image.open(base_path).convert("RGB")

    results = model(base_path, conf=0.7, save=False)  
    boxes_rel = []
    names = []
    w, h = image.size
    for box, cls in zip(results[0].boxes.xyxy.cpu().numpy(), results[0].boxes.cls.cpu().numpy()):
        label = model.names[int(cls)]
        if label in labels:
            x1, y1, x2, y2 = box
            boxes_rel.append([x1/w, y1/h, x2/w, y2/h])
            names.append(label)
    boxes_total.append(boxes_rel)
    annotated = results[0].plot()   
    plt.imshow(annotated[..., ::-1]) 
    plt.axis("off")
    plt.show()
    yolo_detect_path = os.path.join(yolo_detect_result_dir,f'yolo_{i}.jpg')
    cv2.imwrite(yolo_detect_path, annotated)
    print(f"yolo 边缘图已成功保存到: {yolo_detect_path}")

# #sd generate image with controlnet
print("ms sd generating...")

input_images = [image0, image1]
input_images = [x.convert("RGB").resize((512, 512)) for x in input_images]

control_image = Image.open("./examples/depth.png").resize((1024, 1024))
phrases = [["cat", "dog"]]
controlnet_conditioning_scale = 0.7
msadapter_scale = 0.6
result_path = './res/action_result/dog_cat'
# prompt = "The cat and the dog are playing by the seaside, with the cat's paw stroking the dog's head."
# save_name = f'controlnet_conditioning_scale_{controlnet_conditioning_scale}_canny_1.jpg'
# control_image = Image.open("./res/canny/dog_cat_1.jpg").resize((1024, 1024))
# boxes = [[boxes_rel[1]]]
# inference_controlnet.ms_sd_generate_image(input_images,prompt,phrases,controlnet_conditioning_scale,control_image=control_image,boxes=boxes,result_path= result_path,save_name = save_name)

# Experiment for controlnet_conditioning_scale and msadapter_scale

controlnet_conditioning_scales = [0.1,0.5,1,1.5,2.0]
msadapter_scales = [0.1,0.5,1,1.5]

for i in range(len(controlnet_conditioning_scales)):
    prompt = "Dogs and cats playing on the beach, with the dog on top of the cat"
    save_name = f'controlnet_conditioning_scale_{controlnet_conditioning_scales[i]}_{msadapter_scales[i]}_experiment_1_dog_cat_0_{i}.jpg'
    control_image = Image.open("./res/canny/dog_cat_0.jpg").resize((1024, 1024))
    boxes = [boxes_total[0]]
    inference_controlnet.ms_sd_generate_image(input_images,prompt,phrases,controlnet_conditioning_scale,msadapter_scale,control_image,boxes=boxes,result_path= result_path,save_name = save_name)

    save_name = f'controlnet_conditioning_scale_{controlnet_conditioning_scales[i]}_{msadapter_scales[i]}_experiment_1_dog_cat_2_{i}.jpg'
    control_image = Image.open("./res/canny/dog_cat_2.jpg").resize((1024, 1024))
    boxes = [boxes_total[1]]
    inference_controlnet.ms_sd_generate_image(input_images,prompt,phrases,controlnet_conditioning_scale,msadapter_scale,control_image,boxes=boxes,result_path= result_path,save_name = save_name)


