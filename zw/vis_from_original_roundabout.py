import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
from tqdm import tqdm

# ===== 配置参数（根据实际路径修改） =====
ann_file = 'instances/train/train_annotations.json'  # COCO格式的标注文件路径
img_dir = 'instances/train/images/'           # 图片文件夹路径
output_dir = 'output_visualizations/'  # 可视化结果保存路径
target_class = 'roundabout'                  # 目标类别（环岛）
num_images = 10                             # 需要可视化的图片数量

# ===== 主程序 =====
def visualize_roundabout_instances():
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化COCO API
    coco = COCO(ann_file)
    
    # 获取目标类别的ID
    cat_ids = coco.getCatIds(catNms=[target_class])
    if not cat_ids:
        raise ValueError(f"类别 '{target_class}' 在数据集中不存在！")
    
    # 获取包含目标类别的图片ID
    img_ids = coco.getImgIds(catIds=cat_ids)
    if len(img_ids) < num_images:
        print(f"警告: 仅找到 {len(img_ids)} 张含 '{target_class}' 的图片，将全部可视化")
        selected_img_ids = img_ids
    else:
        selected_img_ids = img_ids[:num_images]
    
    # 遍历选中的图片
    for img_id in tqdm(selected_img_ids, desc="处理图片"):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV默认BGR转RGB
        
        # 获取当前图片的所有标注
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        annotations = coco.loadAnns(ann_ids)
        
        # 创建可视化画布
        plt.figure(figsize=(15, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title(f"Image ID: {img_id} | {target_class.capitalize()} Instances")
        
        # 绘制实例分割掩膜和边界框
        for ann in annotations:
            # 绘制分割掩膜
            if 'segmentation' in ann:
                mask = coco.annToMask(ann)  # 将COCO分割标注转为二值掩膜
                color = np.random.rand(3)    # 随机颜色
                plt.imshow(np.ma.masked_where(mask == 0, mask), 
                          alpha=0.5, cmap='viridis')
            
            # 绘制边界框和类别标签
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                plt.gca().add_patch(plt.Rectangle(
                    (x, y), w, h, 
                    fill=False, edgecolor='red', linewidth=2
                ))
                plt.text(x, y - 5, 
                        f"{target_class}",
                        fontsize=12, color='white', 
                        bbox=dict(facecolor='red', alpha=0.7))
        
        # 保存结果（禁用显示）
        output_path = os.path.join(output_dir, f"{img_id}_{target_class}.png")
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()  # 关闭图形释放内存

if __name__ == "__main__":
    visualize_roundabout_instances()
    print(f"可视化完成！结果已保存至: {output_dir}")