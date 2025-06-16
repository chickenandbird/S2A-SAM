import json
import random
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

# 配置路径参数
image_dir = "instances/train/images"
annotation_path = "instances/train/train_rbox_annotations.json"
output_dir = "instances/train"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载原始标注文件
with open(annotation_path, 'r') as f:
    coco_data = json.load(f)

# 创建图像ID到标注的映射
image_annotations = defaultdict(list)
for ann in coco_data['annotations']:
    image_annotations[ann['image_id']].append(ann)

# 提取所有图像ID并分割
all_image_ids = list(image_annotations.keys())
train_ids, val_ids = train_test_split(
    all_image_ids, 
    test_size=0.2,  # 验证集比例20%
    random_state=42,  # 固定随机种子确保可复现
    shuffle=True
)

# 构建新数据集结构
def create_subset(image_ids, coco_data):
    # 筛选图像
    images_subset = [img for img in coco_data['images'] if img['id'] in image_ids]
    
    # 筛选标注
    annotations_subset = []
    for img_id in image_ids:
        annotations_subset.extend(image_annotations[img_id])
    
    return {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data.get("categories", []),
        "images": images_subset,
        "annotations": annotations_subset
    }

# 创建训练集和验证集
train_data = create_subset(train_ids, coco_data)
val_data = create_subset(val_ids, coco_data)

# 保存分割结果
with open(os.path.join(output_dir, "train.json"), 'w') as f:
    json.dump(train_data, f, indent=2)

with open(os.path.join(output_dir, "val.json"), 'w') as f:
    json.dump(val_data, f, indent=2)

print(f"分割完成！共处理 {len(all_image_ids)} 张图片")
print(f"训练集: {len(train_ids)} 张图片 | 验证集: {len(val_ids)} 张图片")
print(f"输出路径: {output_dir}")
