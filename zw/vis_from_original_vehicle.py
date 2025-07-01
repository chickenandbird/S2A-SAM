import os
import json
import cv2
import numpy as np
import random
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
from tqdm import tqdm

# ===== 配置参数 =====
ann_file = 'instances/train/train_annotations.json'  # COCO格式的标注文件路径
img_dir = 'instances/train/images/'           # 图片文件夹路径
output_dir = 'output_visualizations/'  # 可视化结果保存路径
target_class = 'storage_tank'                  # 目标类别（支持父类）
num_images = 30                             # 生成图片数量

# ===== 主程序 =====
def visualize_storage_tank_instances():
    os.makedirs(output_dir, exist_ok=True)
    coco = COCO(ann_file)
    
    # 获取storage_tank及其所有子类别的ID列表
    cat_ids = coco.getCatIds(catNms=[target_class])
    if not cat_ids:
        raise ValueError(f"找不到 '{target_class}' 类别或子类！")
    
    # 获取包含目标类别的图片ID（去重）
    img_ids = list(set([ann['image_id'] for ann in coco.loadAnns(coco.getAnnIds(catIds=cat_ids))]))
    if len(img_ids) < num_images:
        print(f"警告: 仅找到 {len(img_ids)} 张含 '{target_class}' 的图片")
        selected_img_ids = img_ids
    else:
        selected_img_ids = img_ids[:num_images]
    
    # 遍历选中的图片
    for img_id in tqdm(selected_img_ids, desc="生成可视化"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 获取当前图片的所有标注实例
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        
        plt.figure(figsize=(15, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Image ID: {img_id} | storage_tank Instances")
        
        # 为每个实例生成随机颜色
        color_palette = {ann['id']: [random.random() for _ in range(3)] for ann in annotations}
        
        for ann in annotations:
            # 跳过非storage_tank类别的实例
            if ann['category_id'] not in cat_ids:
                continue
                
            color = color_palette[ann['id']]
            cat_name = coco.loadCats(ann['category_id'])[0]['name']
            
            # 绘制分割掩膜
            if 'segmentation' in ann:
                mask = coco.annToMask(ann)
                plt.imshow(
                    np.ma.masked_where(mask == 0, mask),
                    alpha=0.4,  # 降低透明度避免遮挡
                    cmap=plt.cm.viridis if ann['iscrowd'] else plt.cm.jet,
                )
            
            # 绘制边界框和标签
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                rect = plt.Rectangle(
                    (x, y), w, h,
                    fill=False, edgecolor=color, linewidth=2, linestyle='-' if not ann['iscrowd'] else '--'
                )
                plt.gca().add_patch(rect)
                plt.text(
                    x, y - 10, 
                    f"{cat_name} (ID:{ann['id']})",  # 显示类别名和实例ID
                    fontsize=10, color='white', 
                    bbox=dict(facecolor=(color[0], color[1], color[2], 0.7), alpha=0.7)
                )
        
        # 保存结果（不显示）
        output_path = os.path.join(output_dir, f"{img_id}_storage_tank_instances.png")
        plt.savefig(output_path, bbox_inches='tight', dpi=120, pad_inches=0.1)
        plt.close()

if __name__ == "__main__":
    visualize_storage_tank_instances()
    print(f"成功生成30张可视化图片至: {output_dir}")