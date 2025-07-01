import os

def remove_container_crane_annotations(anno_dir):
    """
    删除DOTA格式标注文件中所有container-crane类别的标注行
    :param anno_dir: 存放标注txt文件的目录路径
    """
    for filename in os.listdir(anno_dir):
        if not filename.endswith('.txt'):
            continue
            
        file_path = os.path.join(anno_dir, filename)
        new_lines = []  # 存储保留的行
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            # 跳过空行和注释行
            if not line.strip() or line.startswith('#'):
                new_lines.append(line)
                continue
                
            parts = line.strip().split()
            # 保留非标注行（字段数<9）和不是container-crane的标注行[6](@ref)
            if len(parts) < 9 or parts[8] != 'container-crane':
                new_lines.append(line)
        
        # 覆盖原文件（先备份更安全）
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
        print(f"Processed: {filename} - Removed {len(lines)-len(new_lines)} instances")

# 使用示例
anno_directory = "mnt/nas-new/home/zhanggefan/zw/A_datasets/qiyuan/ALL/train_all/annfiles15/A_datasets/qiyuan/ALL/train_all1/annfiles"  # 替换为标注文件夹路径
remove_container_crane_annotations(anno_directory)