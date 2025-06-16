import os
import shutil
import random

# 配置路径
src_image_dir = "instances2/val/images"          # 原图像目录
src_label_dir = "instances2/val/labelTxt"         # 原标签目录
dst_image_dir = "instances2/val2/images"          # 新图像目录
dst_label_dir = "instances2/val2/labelTxt"         # 新标签目录
num_samples = 200                      # 抽取样本数

# 创建目标目录
os.makedirs(dst_image_dir, exist_ok=True)
os.makedirs(dst_label_dir, exist_ok=True)

# 获取所有标签文件名（不含后缀）
all_labels = [f.split(".")[0] for f in os.listdir(src_label_dir) if f.endswith(".txt")]

# 随机抽取200个样本的基名
selected_basenames = random.sample(all_labels, num_samples)

# 复制选中的图像和标签
for basename in selected_basenames:
    # 复制标签文件
    src_label = os.path.join(src_label_dir, basename + ".txt")
    dst_label = os.path.join(dst_label_dir, basename + ".txt")
    shutil.copy(src_label, dst_label)
    
    # 复制图像文件（支持多种格式）
    for ext in [".jpg", ".png", ".jpeg"]:
        src_img = os.path.join(src_image_dir, basename + ext)
        if os.path.exists(src_img):
            shutil.copy(src_img, os.path.join(dst_image_dir, basename + ext))
            break

print(f"已完成：{num_samples}个样本保存至 val2")