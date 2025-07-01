import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from PIL import Image
from collections import defaultdict

# 配置路径
json_path = "submit_0.3_new.json"
image_dir = "instances/test/images"
output_dir = "/root/autodl-tmp/json_img"
os.makedirs(output_dir, exist_ok=True)

# 加载JSON数据
with open(json_path, 'r') as f:
    annotations = json.load(f)

# 按图像ID分组标注
annotations_by_image = defaultdict(list)
for ann in annotations:
    image_id = ann["image_id"]
    annotations_by_image[image_id].append(ann)

# 创建类别颜色映射
category_colors = {
    1: (1, 0, 0, 0.5),    # 红色
    2: (0, 1, 0, 0.5),    # 绿色
    3: (0, 0, 1, 0.5),    # 蓝色
    6: (1, 1, 0, 0.5),    # 黄色
    7: (1, 0, 1, 0.5),    # 紫色
    # 其他类别使用默认颜色
}

# 处理每个图像
image_id = 101
img_anns = annotations_by_image[image_id]
# for image_id, img_anns in annotations_by_image.items():
# 构造图像路径
img_path = os.path.join(image_dir, f"{image_id}.png")

# 加载图像
if not os.path.exists(img_path):
    print(f"Image not found: {img_path}")
    

img = np.array(Image.open(img_path))
height, width = img.shape[:2]

# 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
fig.suptitle(f"Image ID: {image_id}", fontsize=16)

# 左侧：边缘点连接形式（修改部分）
ax1.imshow(img)
for ann in img_anns:
    cat_id = ann["category_id"]
    color = category_colors.get(cat_id, (0.5, 0.5, 0.5, 0.5))
    
    # 转换分割点
    if "segmentation" in ann and ann["segmentation"]:
        seg = ann["segmentation"][0]
        points = np.array(seg).reshape(-1, 2)
        x = points[:, 0]
        y = points[:, 1]
        
        # 绘制边缘点连接线（关键修改）
        ax1.plot(x, y, color=color, linewidth=1.5, alpha=0.9, marker='o', 
                markersize=4, markerfacecolor=color, markeredgewidth=0.5)
        
        # 标记起点和终点
        ax1.plot(x[0], y[0], 'go', markersize=8, label='Start Point' if ann == img_anns[0] else "")
        ax1.plot(x[-1], y[-1], 'ro', markersize=8, label='End Point' if ann == img_anns[0] else "")
        
        # 添加点序号（仅显示前10个点）
        for i, (px, py) in enumerate(points[:10]):
            ax1.text(px, py, str(i), color='white', fontsize=8, 
                    bbox=dict(boxstyle='circle,pad=0.2', facecolor=color, alpha=0.7))

# 添加图例（仅一次）
if img_anns:
    ax1.legend(loc='upper right', fontsize=10)

ax1.set_title("Edge Points Connection (Check Point Order)", fontsize=14)
ax1.axis('off')

# 右侧：边界框检测图（保持不变）
ax2.imshow(img)
for ann in img_anns:
    cat_id = ann["category_id"]
    color = category_colors.get(cat_id, (0.5, 0.5, 0.5, 0.6))
    bbox = ann["bbox"]
    x, y, w, h = bbox
    
    # 绘制边界框
    rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                            edgecolor=color[:3], facecolor=color, alpha=0.3)
    ax2.add_patch(rect)
    
    # 添加类别标签
    label = f"Class {cat_id}"
    ax2.text(x, y - 5, label, color='white', fontsize=9, 
            bbox=dict(facecolor=color[:3], alpha=0.7, pad=2))

ax2.set_title("Object Detection", fontsize=14)
ax2.axis('off')

# 保存并关闭
plt.tight_layout()
output_path = os.path.join(output_dir, f"{image_id}_comparison.png")
plt.savefig(output_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved: {output_path}")

print("\nProcessing completed!")