import os
import json
import argparse
from pathlib import Path

def convert_coco_to_dota(coco_json_path, output_label_dir, coco_image_dir, output_image_dir):
    """
    转换COCO标注文件到DOTA格式
    :param coco_json_path: COCO标注JSON文件路径
    :param output_label_dir: 输出DOTA标签目录
    :param coco_image_dir: 原始COCO图像目录
    :param output_image_dir: 输出软链接图像目录
    """
    # 创建输出目录
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_image_dir, exist_ok=True)
    
    # 读取COCO标注
    with open(coco_json_path) as f:
        coco_data = json.load(f)
    
    # 创建类别映射
    category_map = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 按图像ID组织标注
    image_ann_map = {}
    for img in coco_data['images']:
        image_ann_map[img['id']] = []
    
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id in image_ann_map:
            image_ann_map[image_id].append(ann)
    
    # 处理每个图像
    for img_id, annotations in image_ann_map.items():
        # 获取图像信息
        img_info = next(img for img in coco_data['images'] if img['id'] == img_id)
        img_name = Path(img_info['file_name']).stem
        img_ext = Path(img_info['file_name']).suffix
        
        # 1. 创建图像软链接
        src_img_path = os.path.join(coco_image_dir, f"{img_name}{img_ext}")
        dst_img_path = os.path.join(output_image_dir, f"{img_name}{img_ext}")
        
        if not os.path.exists(dst_img_path):
            os.symlink(
                os.path.abspath(src_img_path),
                os.path.abspath(dst_img_path)
            )
        
        # 2. 创建DOTA标签文件
        txt_path = os.path.join(output_label_dir, f"{img_name}.txt")
        with open(txt_path, 'w') as f:
            # 写入DOTA文件头
            f.write("imagesource:COCO\n")
            f.write(f"gsd:{img_info.get('gsd', '0.0')}\n")
            
            # 写入每个标注对象
            for ann in annotations:
                # 获取四点坐标 (x1,y1,x2,y2,x3,y3,x4,y4)
                bbox = ann['bbox']
                if len(bbox) != 8:
                    continue  # 跳过非四点标注
                
                # 格式化坐标字符串
                points_str = " ".join(f"{p:.2f}" for p in bbox)
                class_name = category_map.get(ann['category_id'], 'unknown')
                
                # 写入DOTA格式行: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficult
                f.write(f"{points_str} {class_name} 0\n")

def main():
    # 配置路径
    base_dir = "instances"
    output_base = "instances2"
    
    # 训练集转换
    convert_coco_to_dota(
        coco_json_path=os.path.join(base_dir, "train", "train.json"),
        output_label_dir=os.path.join(output_base, "train", "labelTxt"),
        coco_image_dir=os.path.join(base_dir, "train", "images"),
        output_image_dir=os.path.join(output_base, "train", "images")
    )
    
    # 验证集转换
    convert_coco_to_dota(
        coco_json_path=os.path.join(base_dir, "train", "val.json"),
        output_label_dir=os.path.join(output_base, "val", "labelTxt"),
        coco_image_dir=os.path.join(base_dir, "train", "images"),
        output_image_dir=os.path.join(output_base, "val", "images")
    )
    
    print("转换完成！数据集结构：")
    print(f"└── instances2/")
    print(f"    ├── train/")
    print(f"    │   ├── images/  # 指向原图像的软链接")
    print(f"    │   └── labelTxt/  # DOTA格式标签")
    print(f"    └── val/")
    print(f"        ├── images/  # 指向原图像的软链接")
    print(f"        └── labelTxt/  # DOTA格式标签")

if __name__ == "__main__":
    main()