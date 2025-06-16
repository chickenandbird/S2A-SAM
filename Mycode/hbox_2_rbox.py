import cv2
import numpy as np
from pycocotools.coco import COCO
import json

def mask_to_dota_points(mask: np.ndarray) -> list:
    """ 从二值掩码生成DOTA格式的四点标注 """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: 
        return None
    
    # 过滤小面积轮廓 & 取最大连通域
    contours = [c for c in contours if cv2.contourArea(c) > 10]
    if not contours:
        return None
    main_contour = max(contours, key=cv2.contourArea)
    # 合并多轮廓（如分散的物体部件）
    combined_contour = np.vstack(main_contour)
    rect = cv2.minAreaRect(combined_contour)  # 返回((cx,cy), (w,h), θ)
    
    # 获取旋转矩形的四个顶点坐标（OpenCV格式）
    points = cv2.boxPoints(rect)  # 返回4×2数组

    cx, cy = rect[0]
    points = np.array(sorted(points, key=lambda p: np.arctan2(p[1]-cy, p[0]-cx)))

    # 展平为8个数值的列表 [x1,y1,x2,y2,x3,y3,x4,y4]
    return points.flatten().tolist()  # 直接返回四点坐标

def convert_to_rotated_coco(anno_path: str, output_path: str):
    coco = COCO(anno_path)
    new_annos = {
        "images": [],
        "categories": coco.dataset["categories"],
        "annotations": []
    }

    # 遍历所有图像
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(img_id)[0]
        new_annos["images"].append(img_info)
        
        # 处理每张图的标注
        ann_ids = coco.getAnnIds(imgIds=img_id)
        for ann in coco.loadAnns(ann_ids):
            if ann['iscrowd'] == 1:  # 跳过拥挤区域
                continue
            mask = coco.annToMask(ann)  # 将COCO多边形转为二值掩码
            dota_points = mask_to_dota_points(mask)  # 获取DOTA格式的四点坐标
            
            if dota_points:
                new_ann = {
                    "id": ann["id"],
                    "image_id": img_id,
                    "category_id": ann["category_id"],
                    "segmentation": ann["segmentation"],  # 保留原分割信息
                    "bbox": dota_points,  # 关键修改：使用四点坐标
                    "area": ann["area"],
                    "iscrowd": ann["iscrowd"]
                }
                new_annos["annotations"].append(new_ann)
    
    # 保存新标注文件
    with open(output_path, 'w') as f:
        json.dump(new_annos, f)

convert_to_rotated_coco('instances/train/train_annotations.json','instances/train/train_rbox_annotations.json')

# import json
# from collections import defaultdict

# def analyze_json_structure(data, indent=0, path="<root>", structure=None):
#     """递归分析JSON数据结构"""
#     if structure is None:
#         structure = defaultdict(list)
    
#     indent_str = "│   " * (indent - 1) + "├── " if indent > 0 else ""
    
#     if isinstance(data, dict):
#         structure[path].append("{} (dict)")
#         print(f"{indent_str}{path}: dict")
#         for key, value in data.items():
#             new_path = f"{path}.{key}" if path != "<root>" else key
#             analyze_json_structure(value, indent + 1, new_path, structure)
    
#     elif isinstance(data, list):
#         structure[path].append("[] (list)")
#         print(f"{indent_str}{path}: list")
#         if data:
#             # 分析第一个元素作为列表结构代表
#             analyze_json_structure(data[0], indent + 1, f"{path}[0]", structure)
#         else:
#             print("│   " * indent + "└── (empty list)")
    
#     else:  # 基本数据类型
#         data_type = type(data).__name__
#         structure[path].append(data_type)
#         print(f"{indent_str}{data_type}")
    
#     return structure

# def visualize_json_structure(file_path):
#     """可视化JSON文件结构"""
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         print(f"\nJSON文件结构分析: {file_path}")
#         print("─" * 50)
#         structure = analyze_json_structure(data)
        
#         # 生成类型统计
#         type_count = defaultdict(int)
#         for types in structure.values():
#             for t in types:
#                 type_count[t] += 1
        
#         print("\n" + "─" * 50)
#         print("结构统计:")
#         for t, count in type_count.items():
#             print(f"• {t}: {count}处")
        
#         max_depth = max(len(path.split('.')) for path in structure)
#         print(f"最大嵌套深度: {max_depth}层")
    
#     except FileNotFoundError:
#         print(f"错误: 文件 {file_path} 不存在")
#     except json.JSONDecodeError as e:
#         print(f"JSON解析错误: {e}")

# # 使用示例
# if __name__ == "__main__":
#     json_file = "instances/train/train_annotations.json"  # 替换为你的JSON文件路径
#     visualize_json_structure(json_file)