import os
import argparse

label_dict = {
    'storage_tank':1,
    'vehicle':2,
    'aircraft':3,
    'ship':4,
    'bridge':5,
    'sports_facility':6,
    'roundabout':7,
    'harbor':8
}

def convert_to_dota_annfiles(result_dir, output_dir, conf_threshold=0.3):
    """
    将S2ANet结果文件转换为DOTA格式标注
    :param result_dir: S2ANet结果文件夹路径 (包含Task1_*.txt文件)
    :param output_dir: 输出的DOTA标注文件夹路径
    :param conf_threshold: 置信度筛选阈值
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储每张图片的标注内容 {img_name: [annotations]}
    img_annotations = {}
    
    # 遍历所有结果文件
    for filename in os.listdir(result_dir):
        if not filename.startswith("Task1_") or not filename.endswith(".txt"):
            continue
        
        # 从文件名提取类别名 (Task1_class.txt → class)
        class_name = filename[6:].split('.')[0]
        class_id = label_dict[class_name]
        
        with open(os.path.join(result_dir, filename), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:  # 确保有足够数据
                    continue
                
                # 解析数据：图片名，置信度，8个坐标值
                img_name = parts[0]
                confidence = float(parts[1])
                coords = list(map(float, parts[2:10]))
                
                # 置信度筛选
                if confidence < conf_threshold:
                    continue
                
                # 创建DOTA格式标注行
                annotation = " ".join([f"{coord:.2f}" for coord in coords])
                annotation += f" {class_id}\n"  # 末尾0表示非困难样本
                
                # 添加到对应图片的标注
                if img_name not in img_annotations:
                    img_annotations[img_name] = []
                img_annotations[img_name].append(annotation)
    
    # 写入每张图片的标注文件
    for img_name, annotations in img_annotations.items():
        output_path = os.path.join(output_dir, f"{img_name}.txt")
        with open(output_path, 'w') as f:
            f.writelines(annotations)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="result/DOTA_28epoch_15", 
                       help="S2ANet结果文件夹路径")
    parser.add_argument("--output-dir", default="instances/test/annfile_28epoch_15", 
                       help="DOTA标注输出路径")
    parser.add_argument("--conf-threshold", type=float, default=0.3, 
                       help="置信度筛选阈值")
    args = parser.parse_args()
    
    convert_to_dota_annfiles(args.result_dir, args.output_dir, args.conf_threshold)
    print(f"转换完成! 共处理{len(os.listdir(args.result_dir))}个类别文件")
    print(f"标注已保存至: {args.output_dir}")
    