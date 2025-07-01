import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from PIL import Image
from collections import defaultdict

# 配置路径
json_path = "instances/test/7_result.json"
image_dir = "instances/test/images"
output_dir = "/root/autodl-tmp/json_img_6"
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
}

# 处理每个图像
for image_id, img_anns in annotations_by_image.items():
    img_path = os.path.join(image_dir, f"{image_id}.png")
    if not os.path.exists(img_path):
        continue
    
    # 加载图像
    img = np.array(Image.open(img_path))
    height, width = img.shape[:2]
    
    # ===== 1. 单独保存实例分割图 =====
    fig1 = plt.figure(figsize=(12, 8))
    ax1 = fig1.add_axes([0, 0, 1, 1])  # 铺满整个画布
    ax1.imshow(img)
    ax1.axis('off')
    
    seg_patches = []
    for ann in img_anns:
        cat_id = ann["category_id"]
        color = category_colors.get(cat_id, (0.5, 0.5, 0.5, 0.5))
        
        if "segmentation" in ann and ann["segmentation"]:
            seg = ann["segmentation"][0]
            points = np.array(seg).reshape(-1, 2)
            polygon = patches.Polygon(points, closed=True, 
                                     facecolor=color, edgecolor='w', linewidth=1)
            seg_patches.append(polygon)
    
    if seg_patches:
        seg_collection = PatchCollection(seg_patches, match_original=True)
        ax1.add_collection(seg_collection)
    
    # 保存分割图（禁用边界裁剪）
    seg_output = os.path.join(output_dir, f"{image_id}_seg.png")
    plt.savefig(seg_output, dpi=150, pad_inches=0)
    plt.close(fig1)
    
    # ===== 2. 单独保存边界框图 =====
    fig2 = plt.figure(figsize=(12, 8))
    ax2 = fig2.add_axes([0, 0, 1, 1])  # 铺满整个画布
    ax2.imshow(img)
    ax2.axis('off')
    
    for ann in img_anns:
        cat_id = ann["category_id"]
        color = category_colors.get(cat_id, (0.5, 0.5, 0.5, 0.6))
        
        if "bbox" in ann:
            x, y, w, h = ann["bbox"]
            # 修正超界边界框
            x, y = max(0, x), max(0, y)
            w, h = min(w, width - x), min(h, height - y)
            
            rect = patches.Rectangle((x, y), w, h, linewidth=1.5,
                                    edgecolor=color[:3], facecolor=color, alpha=0.3)
            ax2.add_patch(rect)
    
    # 保存检测图（禁用边界裁剪）
    bbox_output = os.path.join(output_dir, f"{image_id}_bbox.png")
    plt.savefig(bbox_output, dpi=150, pad_inches=0)
    plt.close(fig2)

print(f"处理完成！结果保存至: {output_dir}")