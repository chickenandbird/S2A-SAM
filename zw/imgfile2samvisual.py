
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import math
import os
def obb2xyxy(rbboxes):
    w,h = rbboxes[1]

    a = rbboxes[2]/180*math.pi
    cosa = abs(math.cos(a))
    sina = abs(math.sin(a))
    hbbox_w = cosa * w + sina * h
    hbbox_h = sina * w + cosa * h
    dx,dy = rbboxes [0]
    dw = hbbox_w
    dh = hbbox_h
    x1 = dx - dw / 2
    y1 = dy - dh / 2
    x2 = dx + dw / 2
    y2 = dy + dh / 2
    return [x1, y1, x2, y2]

# 1. 定义DOTA标注转SAM提示点的函数
def dota_to_sam_prompts(dota_line):   
    """
    将DOTA标注行转换为SAM所需的点提示(正点+负点)
    
    参数:
        # 257.51 262.26 428.57 307.82 376.37 503.79 205.31 458.23 3
    返回:
        points: 提示点坐标数组 [[x,y]] (5个点)
        labels: 点标签数组 [1,0,0,0,0] (1正4负)
    """
    parts = dota_line.strip().split()
    coords = np.array(parts[:8], dtype=np.float32)

    x_coords = coords[0::2]  # 从0开始，每隔2个元素取一个
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)

    y_coords = coords[1::2]  # 从1开始，每隔2个元素取一个
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)

    hbox = np.array([min_x, min_y, max_x, max_y])
    points = np.array([[(min_x+max_x)/2, (min_y+max_y)/2], [min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]])
    labels = np.array([1, 0, 0, 0, 0])
    return points, labels, hbox

# 2. 新的可视化函数 - 将所有掩码叠加到一张图上
def visualize_all_masks(image, masks_list, alpha=0.5):
    """
    在原始图像上绘制所有分割掩码
    
    参数:
        image: 原始图像 (RGB格式)
        masks_list: SAM返回的掩码列表
        alpha: 掩码透明度 (0-1)
    
    返回:
        combined_image: 叠加了所有掩码的图像
    """
    # 创建一个彩色背景层
    color_layer = np.zeros_like(image, dtype=np.float32)
    
    # 为每个掩码分配随机颜色并叠加
    for mask in masks_list:
        # 生成随机RGB颜色
        color = np.random.random(3) * 255
        # 将掩码转换为三通道
        mask_3d = np.stack([mask, mask, mask], axis=-1)
        # 将颜色应用到掩码区域
        color_layer = np.where(mask_3d, color, color_layer)
    
    # 将原始图像与彩色掩码层混合
    combined_image = cv2.addWeighted(
        image.astype(np.float32), 1 - alpha, 
        color_layer.astype(np.float32), alpha, 
        0
    ).astype(np.uint8)
    
    return combined_image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    




# 3. 主处理流程
def main():
    use_point = False
    # ===== 配置参数 =====
    IMAGE_PATH_HEAD = f"/mnt/nas-new/home/zhanggefan/zw/A_datasets/qiyuan/COCO/test/images"  # 替换为您的图像路径
    DOTA_TXT_PATH_HEAD = f"/mnt/nas-new/home/zhanggefan/zw/h2rbox-mmrotate/result/s2anet_dota_test_final_dota/"  # 替换为DOTA标注文件路径
    MODEL_TYPE = "vit_b"  # SAM模型类型
    CHECKPOINT_PATH = "/mnt/nas-new/home/zhanggefan/zw/SAMUS/result/ckpt/sam_vit_b_01ec64.pth"  # 模型权重路径
    
    # ===== 加载模型 =====
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    predictor = SamPredictor(sam)
    count = 0
    for filename in os.listdir(IMAGE_PATH_HEAD):
        if count > 100:
            break
        count += 1
        print(count, '--', filename)
        dota_txt = DOTA_TXT_PATH_HEAD + filename.split('.')[0] + '.txt'
        # ===== 读取并预处理图像 =====
        image_bgr = cv2.imread(os.path.join(IMAGE_PATH_HEAD,filename))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)  # 设置图像
        
        # ===== 处理DOTA标注 =====
        all_points = []
        all_labels = []
        all_bboxes = []
        try:
            with open(dota_txt, 'r') as f:
                for line in f:
                    if line.strip():  # 跳过空行
                        points, labels, hbox = dota_to_sam_prompts(line)
                        all_points.append(points)
                        all_labels.append(labels)
                        all_bboxes.append(hbox)
        # 文件处理代码
        except FileNotFoundError:  # [2,8](@ref)
            continue

        # ===== 对每个目标进行分割 =====
        all_masks = []
        for points, labels, hbox in zip(all_points, all_labels, all_bboxes):
            # 转换坐标格式 (N,2) -> (1,N,2)
            point_coords = points
            point_labels = labels
            # SAM预测
            if use_point:
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=False  # 每个目标只输出一个最佳掩码
                )
            else:
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box = hbox,
                    multimask_output=False  # 每个目标只输出一个最佳掩码
                )
            # import pdb
            # pdb.set_trace()
            all_masks.append(masks[0]) 
        
        # ===== 可视化所有结果在一张图上 =====
        if all_masks:
            # 创建包含所有分割结果的图像
            combined_image = visualize_all_masks(image_rgb, all_masks, alpha=0.5)
            
            # 添加提示点到图像上
            for points in all_points:
                # 绘制正点（中心点）
                center = points[0].astype(int)
                cv2.drawMarker(
                    combined_image, 
                    tuple(center), 
                    color=(0, 255, 0),  # 绿色
                    markerType=cv2.MARKER_STAR,
                    markerSize=20,
                    thickness=2
                )
                
                # 绘制负点（四个角点）
                for corner in points[1:]:
                    corner = corner.astype(int)
                    cv2.drawMarker(
                        combined_image, 
                        tuple(corner), 
                        color=(0, 0, 255),  # 红色
                        markerType=cv2.MARKER_CROSS,
                        markerSize=15,
                        thickness=2
                    )
            
            # 保存并显示结果
            plt.figure(figsize=(15, 15))
            plt.imshow(combined_image)
            plt.axis('off')
            plt.title("All Segmentation Results with Prompt Points", fontsize=16)
            plt.savefig(f"/mnt/nas-new/home/zhanggefan/zw/h2rbox-mmrotate/result/vis/{filename.split('.')[0]}.jpg", bbox_inches='tight', pad_inches=0, dpi=300)
            # plt.show()
            plt.close() 
        
        # # ===== 可选: 保存所有掩码 =====
        # for i, mask in enumerate(all_masks):
        #     # 二值化掩码 (0/255)
        #     mask_uint8 = (mask * 255).astype(np.uint8)
        #     cv2.imwrite(f"mask_{i}.png", mask_uint8)

if __name__ == "__main__":
    main()


































































