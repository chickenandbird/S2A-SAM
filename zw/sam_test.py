
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import math
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
def dota_to_sam_prompts(dota_line, img_size=(1024, 1024), offset=10):
    """
    将DOTA标注行转换为SAM所需的点提示（正点+负点）
    
    参数:
        dota_line: DOTA格式字符串 "x1 y1 x2 y2 x3 y3 x4 y4 class difficulty"
        img_size: 图像尺寸 (width, height)
        offset: 负点外扩偏移量(像素)
    
    返回:
        points: 提示点坐标数组 [[x,y]] (5个点)
        labels: 点标签数组 [1,0,0,0,0] (1正4负)
    """
    parts = dota_line.strip().split()
    coords = np.array(parts[:8], dtype=np.float32).reshape(4, 2)  # 提取四边形坐标
    
    # 计算中心点(正点)
    center = np.mean(coords, axis=0)
    
    # 计算四角外扩点(负点)
    neg_points = []
    for corner in coords:
        # 从中心指向角点的单位向量
        direction = corner - center
        unit_vector = direction / np.linalg.norm(direction)
        # 沿方向外扩
        new_point = corner + unit_vector * offset
        # 约束在图像范围内
        new_point[0] = np.clip(new_point[0], 0, img_size[0]-1)
        new_point[1] = np.clip(new_point[1], 0, img_size[1]-1)
        neg_points.append(new_point)
    
    # 组合所有点: [中心点, 4个外扩点]
    all_points = np.vstack([center, neg_points])
    labels = np.array([1, 0, 0, 0, 0])  # 标签: 1正 + 4负
    
    box_points = cv2.minAreaRect(coords)

    hbox = np.array(obb2xyxy(box_points))
    return all_points, labels,hbox

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

# 3. 主处理流程
def main():
    num=4970
    use_point = False
    # ===== 配置参数 =====
    IMAGE_PATH = f"instances3/trainval/images/{num}__1024__0___0.png"  # 替换为您的图像路径
    DOTA_TXT_PATH = f"instances3/trainval/annfiles/{num}__1024__0___0.txt"  # 替换为DOTA标注文件路径
    MODEL_TYPE = "vit_h"  # SAM模型类型
    CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"  # 模型权重路径
    
    # ===== 加载模型 =====
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    predictor = SamPredictor(sam)
    
    # ===== 读取并预处理图像 =====
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)  # 设置图像
    
    # ===== 处理DOTA标注 =====
    all_points = []
    all_labels = []
    all_bboxes = []
    with open(DOTA_TXT_PATH, 'r') as f:
        for line in f:
            if line.strip():  # 跳过空行
                points, labels,hbox = dota_to_sam_prompts(line, img_size=(image_rgb.shape[1], image_rgb.shape[0]))
                all_points.append(points)
                all_labels.append(labels)
                all_bboxes.append(hbox)
    
    # ===== 对每个目标进行分割 =====
    all_masks = []
    for points, labels,hbox in zip(all_points, all_labels,all_bboxes):
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
        all_masks.append(masks[0])  # 取置信度最高的掩码
    
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
        plt.savefig("combined_segmentation_result.jpg", bbox_inches='tight', pad_inches=0, dpi=300)
        plt.show()
    
    # ===== 可选: 保存所有掩码 =====
    for i, mask in enumerate(all_masks):
        # 二值化掩码 (0/255)
        mask_uint8 = (mask * 255).astype(np.uint8)
        cv2.imwrite(f"mask_{i}.png", mask_uint8)

if __name__ == "__main__":
    main()


































































