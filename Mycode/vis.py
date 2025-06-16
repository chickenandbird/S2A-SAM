import os
import cv2
import numpy as np
from tqdm import tqdm

# ===== 1. 配置路径 =====
image_dir = "instances2/train/images"
label_dir = "instances2/train/labelTxt"
output_dir = "instances2/train/vis_images"
os.makedirs(output_dir, exist_ok=True)

# ===== 2. 定义类别颜色映射 =====
class_colors = {
    'storage_tank': (0, 255, 0),      # 绿色 - 存储罐
    'vehicle': (255, 0, 0),           # 红色 - 车辆
    'aircraft': (0, 0, 255),          # 蓝色 - 飞机
    'ship': (255, 255, 0),            # 青色 - 船舶
    'bridge': (128, 0, 128),          # 紫色 - 桥梁
    'sports_facility': (255, 165, 0), # 橙色 - 体育设施
    'roundabout': (0, 255, 255),      # 黄色 - 环岛
    'harbor': (0, 128, 128)           # 深青色 - 港口
}

# ===== 3. 可视化函数 =====
def visualize_dota_annotation(image_path, label_path, output_path):
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return
    
    # 解析标注文件（跳过前2行元数据）
    if not os.path.exists(label_path):
        return
    
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()[2:] if line.strip()]
    
    # 绘制每个实例
    for line in lines:
        parts = line.split()
        if len(parts) < 9:
            continue
        
        # 提取旋转框坐标和类别
        points = np.array(list(map(float, parts[:8]))).reshape(4, 2).astype(int)
        cls_name = parts[8].lower().replace(' ', '_')  # 统一为小写+下划线格式
        
        # 跳过未定义类别
        if cls_name not in class_colors:
            continue
            
        # 绘制旋转框
        color = class_colors[cls_name]
        cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)
        
        # 标记类别名称
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))
        cv2.putText(img, cls_name, (center_x-20, center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 保存可视化结果
    cv2.imwrite(output_path, img)

# ===== 4. 批量处理所有图像 =====
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
for img_file in tqdm(image_files, desc="可视化进度"):
    base_name = os.path.splitext(img_file)[0]
    img_path = os.path.join(image_dir, img_file)
    label_path = os.path.join(label_dir, f"{base_name}.txt")
    output_path = os.path.join(output_dir, f"vis_{base_name}.jpg")
    
    visualize_dota_annotation(img_path, label_path, output_path)

print(f"可视化完成！结果已保存至: {output_dir}")