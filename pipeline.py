
import inference_controlnet
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from controlnet_aux import CannyDetector, HEDdetector
import os
from ultralytics import YOLO

# input

# image0 = Image.open("./examples/example_dog.jpg")
# image1 = Image.open("./examples/example_cat.jpg")
# image0 = Image.open("./examples/bench_nobg.png")
# image1 = Image.open("./examples/luggage_nobg.png")
# image2 = Image.open("./examples/bag_nobg.png")
# image3 = Image.open("./examples/model_nobg.png")
image0 = Image.open("./examples/white_skirt.png")
image1 = Image.open("./examples/blue_shirt.png")
image2 = Image.open("./examples/bag_woman.png")

# output
output_canny_dir = './res/canny/three_object'
pose_images_dir = None
pose_images_path = "./examples/pose_girl.png"

# setting 
pose_process = 'openpose'
yolo = None
phrases = [["shirt", "skirt","bag"]]


if pose_images_dir:
    position_image_paths = []
    items = os.listdir(pose_images_dir)
    for item in items:
        position_image_path = os.path.join(pose_images_dir, item)
        position_image_paths.append(position_image_path)

if not pose_images_dir and os.path.exists(pose_images_path):
    position_image_paths = [pose_images_path]

# # generate canny 边缘图
print("正在生成 Canny 边缘图...")
try:
    for i,position_image_path in enumerate(position_image_paths):
        position_image = Image.open(os.path.join(position_image_path)).convert("RGB")
        os.makedirs(output_canny_dir, exist_ok=True)
        output_canny_path = os.path.join(output_canny_dir,f"four_objects{i}.jpg")
        if pose_process == 'canny':
            canny_detector = CannyDetector()
            pose_image = canny_detector(position_image, low_threshold=100, high_threshold=200)
        elif pose_process == "openpose":
            output_canny_path = './res/openpose/openpose.png'
            break
        pose_image.save(output_canny_path)

        print(f"Canny 边缘图已成功保存到: {output_canny_path}")
except Exception as e:
    print(f"生成 Canny 边缘图时发生错误: {e}")

# box generation
boxes_total = None
if yolo:
    model = YOLO("/home/rjiangas/models/youlov8/yolov8m.pt")
    labels = ["shirt", "skirt","bag"]
    yolo_detect_result_dir = './res/yolo_detect/four_objects'
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
        os.makedirs(yolo_detect_result_dir, exist_ok=True)
        yolo_detect_path = os.path.join(yolo_detect_result_dir,f'four_objects_{i}.jpg')
        cv2.imwrite(yolo_detect_path, annotated)
        print(f"yolo 分割图已成功保存到: {yolo_detect_path}")

# #sd generate image with controlnet
print("ms sd generating...")
input_images = [image0, image1,image2]
input_images = [x.convert("RGB").resize((512, 512)) for x in input_images]

result_path = './res/action_result/three_objects'
# prompt = "The cat and the dog are playing by the seaside, with the cat's paw stroking the dog's head."
# save_name = f'controlnet_conditioning_scale_{controlnet_conditioning_scale}_canny_1.jpg'
# control_image = Image.open("./res/canny/dog_cat_1.jpg").resize((1024, 1024))
# boxes = [[boxes_rel[1]]]
# inference_controlnet.ms_sd_generate_image(input_images,prompt,phrases,controlnet_conditioning_scale,control_image=control_image,boxes=boxes,result_path= result_path,save_name = save_name)
# Experiment for controlnet_conditioning_scale and msadapter_scale

controlnet_conditioning_scales = [0.9]
msadapter_scales = [0.9]
if boxes_total is None:
    boxes_total = [[[0., 0., 0., 0.] for _ in range(len(input_images))]]
for i in range(len(msadapter_scales)):
    for j in range(len(controlnet_conditioning_scales)):
        boxes = [boxes_total[0]]
        prompt = "An Korean women wearing a blue button-down shirt and a white skirt, carrying a sleek black handbag"
        # prompt = "Image of a tourist sitting on a bench with a backpack, and a suitcase next to the bench."
        print("prompt:",prompt)
        print("boxes:",boxes)
        save_name = f'controlnet_conditioning_scale_{controlnet_conditioning_scales[j]}_{msadapter_scales[i]}_experiment_1_three_objects_1_{i}.jpg'
        control_image = Image.open("./res/openpose/openpose_2.png").resize((1024, 1024))
        inference_controlnet.ms_sd_generate_image(input_images,prompt,phrases,controlnet_conditioning_scales[j],msadapter_scales[i],control_image,boxes=boxes,result_path= result_path,save_name = save_name)
        save_name = f'controlnet_conditioning_scale_{controlnet_conditioning_scales[j]}_{msadapter_scales[i]}_experiment_1_two_objects_1_{i}.jpg'
        boxes_total = [[[0., 0., 0., 0.] for _ in range(2)]]
        boxes = [boxes_total[0]]
        inference_controlnet.ms_sd_generate_image(input_images[:2],prompt,[phrases[0][:2]],controlnet_conditioning_scales[j],msadapter_scales[i],control_image,boxes=boxes,result_path= result_path,save_name = save_name)
        save_name = f'controlnet_conditioning_scale_{controlnet_conditioning_scales[j]}_{msadapter_scales[i]}_experiment_1_one_objects_1_{i}.jpg'
        boxes_total = [[[0., 0., 0., 0.] for _ in range(1)]]
        boxes = [boxes_total[0]]
        inference_controlnet.ms_sd_generate_image(input_images[1],prompt,[phrases[0][1]],controlnet_conditioning_scales[j],msadapter_scales[i],control_image,boxes=boxes,result_path= result_path,save_name = save_name)


