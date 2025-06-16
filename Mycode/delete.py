import os
import glob

# 配置路径
label_dir = "instances2/train/labelTxt"

def process_annotation_file(file_path):
    """处理单个标注文件，删除元数据行"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 保留所有包含8个坐标值+类别名的有效数据行
    valid_lines = []
    for line in lines:
        parts = line.strip().split()
        # 检查是否为目标检测数据行：至少8个坐标值 + 类别名 + 0
        if len(parts) >= 9 and parts[-2] in class_names:
            valid_lines.append(line)
    
    # 覆盖原文件
    with open(file_path, 'w') as f:
        f.writelines(valid_lines)

# 定义有效的类别名称（根据您的需求）
class_names = {'storage_tank', 'vehicle', 'aircraft', 'ship', 
               'bridge', 'sports_facility', 'roundabout', 'harbor'}

# 批量处理所有标注文件
txt_files = glob.glob(os.path.join(label_dir, "*.txt"))
for file_path in txt_files:
    process_annotation_file(file_path)
    print(f"已处理: {os.path.basename(file_path)}")

print(f"\n处理完成！共处理{len(txt_files)}个标注文件")