import os
from segment_anything import sam_model_registry, SamPredictor
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
def detect_edges_in_quad(image_path, quad_points, low_threshold=10, high_threshold=50):
    """
    在DOTA格式四边形区域内执行边缘检测
    :param image_path: 输入图像路径
    :param quad_points: DOTA格式四边形点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    :param low_threshold: Canny低阈值
    :param high_threshold: Canny高阈值
    :return: 强边缘点坐标列表
    """
    # 1. 读取图像并转换格式
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 2. 创建四边形区域掩模
    mask = np.zeros_like(img)
    pts = np.array(quad_points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], (255, 255, 255))
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    
    # 3. 应用Canny边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # 4. 应用掩模获取四边形区域内边缘
    quad_edges = cv2.bitwise_and(edges, edges, mask=mask_gray)
    
    # 5. 查找强边缘点
    strong_edges = np.where(quad_edges > 0)
    edge_points = list(zip(strong_edges[1], strong_edges[0]))
    
    return edge_points, img, quad_edges

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
def visualize_results(filename, img, quad_points, quad_edges, edge_points, centroids):
    """
    可视化边缘检测结果和质心
    :param filename: 输出文件名
    :param img: 原始图像 (BGR格式)
    :param quad_points: 四边形点坐标列表 [每个元素为4个点的数组]
    :param quad_edges: 边缘检测结果图像 (单通道)
    :param edge_points: 边缘点坐标列表 [(x,y), ...]
    :param centroids: 质心坐标列表 [(cx, cy), ...]
    """
    plt.figure(figsize=(24, 12))
    
    # 1. 原始图像+四边形+顶点标记+质心
    plt.subplot(131)
    img_copy = img.copy()
    
    # 绘制所有四边形
    for i, qpoints in enumerate(quad_points):
        # 绘制四边形边框
        cv2.polylines(img_copy, [np.array(qpoints, dtype=np.int32)], True, (0, 255, 0), 2)
        
        # 标记四边形顶点
        for j, point in enumerate(qpoints):
            cv2.circle(img_copy, tuple(point), 5, (255, 0, 0), -1)
            cv2.putText(img_copy, f"{i+1}-{j+1}", tuple(point), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 标记所有质心
    for i, (cx, cy) in enumerate(centroids):
        cx, cy = int(cx), int(cy)
        # 绘制五角星质心标记
        cv2.drawMarker(img_copy, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_STAR, 
                      markerSize=10, thickness=2, line_type=cv2.LINE_AA)
        cv2.putText(img_copy, f"C{i+1}", (cx+15, cy-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title("Quadrilateral Regions & Centroids")
    
    # 2. 边缘检测结果
    plt.subplot(132)
    zero_mask = np.zeros_like(quad_edges[0])
    for quad_edge in quad_edges:
        zero_mask = cv2.bitwise_or(zero_mask,quad_edge)

    plt.imshow(zero_mask, cmap='gray')
    plt.title("Edge Detection in Quad")
    
    # 3. 原始图像+边缘点+质心
    plt.subplot(133)
    img_copy2 = img.copy()
    
    # 绘制所有四边形
    for qpoints in quad_points:
        cv2.polylines(img_copy2, [np.array(qpoints, dtype=np.int32)], True, (0, 255, 0), 2)
    
    
    # 标记所有质心
    for i, (cx, cy) in enumerate(centroids):
        cx, cy = int(cx), int(cy)
        cv2.drawMarker(img_copy2, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_CROSS, 
                      markerSize=15, thickness=2, line_type=cv2.LINE_AA)
        cv2.putText(img_copy2, f"C{i+1}", (cx+15, cy-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    plt.imshow(cv2.cvtColor(img_copy2, cv2.COLOR_BGR2RGB))
    plt.title(f"{len(edge_points)} Edge Points & Centroids")
    
    plt.tight_layout()
    plt.savefig(f'canny/{filename}.png', bbox_inches='tight', dpi=150)
    plt.close()  # 关闭图形释放内存

def calculate_polygon_area(points):
    """
    计算多边形的有向面积
    :param points: 顶点坐标列表，格式为 [(x1, y1), (x2, y2), ..., (xn, yn)]
    :return: 多边形的有向面积
    """
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += (x1 * y2 - x2 * y1)
    return area / 2.0

def sort_points_clockwise(points):
    """
    将点排序为顺时针顺序
    :param points: 顶点坐标列表，格式为 [(x1, y1), (x2, y2), ..., (xn, yn)]
    :return: 顺时针排序后的点
    """
    area = calculate_polygon_area(points)
    if area > 0:
        # 如果面积为正，则顶点是逆时针顺序，需要反转
        points.reverse()
    return points

def generate_negative_points(bins_last, bins_first, num_points=2):
    """
    从两个点列列表中生成指定数量的负点（中点）
    
    参数:
        bins_last: 点列列表1 (如abnormal_bins[-1])
        bins_first: 点列列表2 (如abnormal_bins[0])
        num_points: 需生成的负点数量，默认为2
        
    返回:
        negative_points: 负点坐标列表，每个元素为(x, y)
    """
    negative_points = []
    
    for _ in range(num_points):
        # 1. 随机选择两个点：分别从两个列表中随机抽取一个点
        point1 = random.choice(bins_last)  # 从abnormal_bins[-1]随机选
        point2 = random.choice(bins_first)  # 从abnormal_bins[0]随机选
        
        # 2. 计算中点坐标
        mid_x = (point1[0] + point2[0]) / 2
        mid_y = (point1[1] + point2[1]) / 2
        negative_points.append((mid_x, mid_y))
    
    return negative_points

def r2hpl(r_list, height, width,class_id,edges,mask):   # list   -6.07 103.44 121.07 102.87 121.44 184.98 -5.70 185.55 
    pts = np.array(r_list).reshape(4,2).astype(np.int32)
    cv2.fillPoly(mask, [pts], (255,))

    quad_edges = cv2.bitwise_and(edges, edges, mask=mask)
    strong_edges = np.where(quad_edges>0)
    edge_points = list(zip(strong_edges[1], strong_edges[0]))

    min_x = 10000
    max_x = 0
    min_y = 10000
    max_y = 0
    for i in range(len(r_list)):
        if i%2==0:  # x坐标
            min_x = min(r_list[i], min_x)
            max_x = max(r_list[i], max_x)  # >0
        if i%2==1:  # y坐标
            min_y = min(r_list[i], min_y)
            max_y = max(r_list[i], max_y)  # >0
    # 保险起见
    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(width-1, max_x)
    max_y = min(height-1, max_y)

    negative_points = []
    # if min_x!=0 and min_y!=0 and max_x!=width-1 and max_y!=height-1 and int(class_id)==7:
    #     center = [(min_x+max_x)/2.0,(min_y+max_y)/2.0]
    #     cx, cy = center
    #     if edge_points!=[]:
    #         distances = [[math.sqrt((x - cx)**2 + (y - cy)**2),x,y] for x, y in edge_points]
        
    #         # 2. 确定最大距离并分成8个环形区域
    #         max_distance = max([distance[0] for distance in distances])
    #         bin_width = max_distance / 8
    #         bin_ranges = [(i * bin_width, (i + 1) * bin_width) for i in range(8)]
            
    #         # 3. 统计各环形区域的点数量
    #         bins = [0] * 8

    #         bin_points=[[],[],[],[],[],[],[],[]]

    #         for d in distances:
    #             for i, (low, high) in enumerate(bin_ranges):
    #                 if low <= d[0] < high:
    #                     bins[i] += 1
    #                     bin_points[i].append([d[1],d[2]])
    #                     break
            
    #         # 4. 检测异常区域
    #         avg_count = sum(bins) / len(bins)
    #         abnormal_bins = [i for i, count in enumerate(bins) if count > avg_count]
            
    #         if len(abnormal_bins)>=3:
    #             negative_points = generate_negative_points(bins_last=bin_points[-1], bins_first=bin_points[abnormal_bins[0]])

    if len(edge_points)!=0 and int(class_id)!=7:
        centroid = list(np.mean(np.array(edge_points),axis=0))  
        result_p = np.array([centroid, [min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]])
        result_l = np.array([1,0,0,0,0])    
    elif len(edge_points)==0 and int(class_id)!=7:
        centroid = []
        result_p = np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]])
        result_l = np.array([0,0,0,0])    
    elif len(edge_points)!=0 and int(class_id)==7:
        if negative_points==[]:
            centroid = list(np.mean(np.array(edge_points),axis=0)) 
            result_p = np.array([centroid, [min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]])
            result_l = np.array([1,0,0,0,0])    
        else:
            centroid = list(np.mean(np.array(edge_points),axis=0)) 
            result_p = np.array([centroid, [min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y],negative_points[0],negative_points[1]])
            result_l = np.array([1,0,0,0,0,0,0])                       
    else:
        centroid = []
        result_p = np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]])
        result_l = np.array([0,0,0,0])         
    result_h = np.array([min_x, min_y, max_x, max_y])
    # result_p = np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]])
    # result_l = np.array([0,0,0,0])
    det_bbox = [min_x, min_y, max_x-min_x, max_y-min_y]
    return result_h, result_p, result_l, det_bbox,quad_edges,edge_points,centroid#分别是旋转框内的全部边缘的mask和对应边缘的坐标

def vis_contours(masks, contours):

    # 假设 masks[0] 是一个 (542, 542) 的 NumPy 数组，值为 0 或 1
    mask = masks[0]

    # 第一幅图：根据 mask 绘制黑白图像
    black_and_white_image = mask.astype(np.uint8) * 255  # 将 1 转换为 255（白色），0 保持为 0（黑色）

    # 第二幅图：绘制轮廓点
    # 创建一个与 mask 大小相同的画布
    canvas = np.zeros_like(mask, dtype=np.uint8)

    # 为每个轮廓分配一个随机颜色
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i, contour in enumerate(contours):
        color = colors[i % len(colors)]  # 循环使用颜色
        # 将轮廓点绘制到画布上
        for point in contour:
            x, y = point[0]
            cv2.circle(canvas, (x, y), 2, color, -1)  # 绘制点，半径为 2，填充颜色

    # 显示两幅图像
    plt.figure(figsize=(12, 6))

    # 显示第一幅图
    plt.subplot(1, 2, 1)
    plt.title("Black and White Image")
    plt.imshow(black_and_white_image, cmap='gray')
    plt.axis('off')

    # 显示第二幅图
    plt.subplot(1, 2, 2)
    plt.title("Contours Image")
    plt.imshow(canvas, cmap='gray')
    plt.axis('off')

    output_path = "/mnt/nas-new/home/zhanggefan/zw/h2rbox-mmrotate/result/vis_contours.png"  # 指定保存路径
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)

def vis_segmentation_point(masks, contours_list):
    # 假设 masks[0] 是一个 (542, 542) 的 NumPy 数组，值为 0 或 1
    mask = masks[0]

    # 第一幅图：根据 mask 绘制黑白图像
    black_and_white_image = mask.astype(np.uint8) * 255  # 将 1 转换为 255（白色），0 保持为 0（黑色）

    # 假设 contours 是一个列表，每个元素是一个 [x1, y1, x2, y2, ...] 格式的列表
    # 创建一个与 mask 大小相同的画布
    canvas = np.zeros_like(mask, dtype=np.uint8)

    # 为每个轮廓分配一个随机颜色
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    # 遍历每个轮廓
    for i, contour in enumerate(contours_list):
        color = colors[i % len(colors)]  # 循环使用颜色
        # 将轮廓点绘制到画布上
        for j in range(0, len(contour), 2):
            x, y = contour[j], contour[j + 1]
            cv2.circle(canvas, (int(x), int(y)), 2, color, -1)  # 绘制点，半径为 2，填充颜色

    # 创建一个画布来保存两幅图像
    plt.figure(figsize=(12, 6))

    # 显示第一幅图
    plt.subplot(1, 2, 1)
    plt.title("Black and White Image")
    plt.imshow(black_and_white_image, cmap='gray')
    plt.axis('off')

    # 显示第二幅图
    plt.subplot(1, 2, 2)
    plt.title("Contours Image")
    plt.imshow(canvas, cmap='gray')
    plt.axis('off')

    # 保存整个画布为一张图像
    output_path = "/mnt/nas-new/home/zhanggefan/zw/h2rbox-mmrotate/result/vis_segmentation_point.png"  # 指定保存路径
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    print(f"合并后的图像已保存到 {output_path}")

def save_submit_file(sam_path, img_path, folder_path, submit_path):
    # 打开文件并写入 JSON 数据
    with open(submit_path, "w", encoding="utf-8") as file:
        file.write("[")

    sam = sam_model_registry["vit_h"](checkpoint=sam_path)
    sam.to(device='cuda')
    predictor = SamPredictor(sam)

    result = []
    num_ann = 1
    # 遍历每一张图片的预测结果
    num_count = 1
    # for filename in ['101.txt','1014.txt','1027.txt','1058.txt','1059.txt','1068.txt','1123.txt','1151.txt','1157.txt','10011.txt','10109.txt','10116.txt','10220.txt','10225.txt','10292.txt','10348.txt']:
    # for filename in ['142.txt','1014.txt']:
    # for filename in ['10348.txt']:
    for filename in os.listdir(folder_path):   # '1.txt'
    # for filename in ['1170.txt','1151.txt','1133.txt','1086.txt','1014.txt','1013.txt','148.txt','143.txt','142.txt','1290.txt','1143.txt','1221.txt','1210.txt','1301.txt','1320.txt','1334.txt','1372.txt','10276.txt','10358.txt','10493.txt','10660.txt']:
        if filename.endswith('.txt'):
            # 打开并读取文件
            print(f"{num_count}--{filename}")
            num_count += 1
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:   # 读取1.txt，即图片下面的所有检测结果 
                    # 提取文件名部分（去掉后缀）
                    img_id = filename.split('.')[0]   # '1'
                    image_path = os.path.join(img_path, img_id+'.png')
                    image = cv2.imread(image_path)
                    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    edges = cv2.Canny(gray_image,50,150)

                    predictor.set_image(image)
                    quad_edges,edge_points,centroids2,quad_points = [],[],[],[]
                    # 每一行都是这个图片的标注
                    for line in f:  # 60.53 234.26 121.40 212.64 127.80 230.63 66.92 252.25 7
                        result_single_image_single_ann = {}
                        # 去掉行首行尾的空白字符，并按空格分割成列表
                        elements = line.strip().split()
                        if len(elements) < 9:  # 确保每行至少有9个元素
                            continue

                        class_id = int(elements[8])  
                        det_rbox = [float(element) for element in elements[:8]]
                        mask = np.zeros_like(gray_image)
                        height, width, channels = image.shape
                        result_h, result_p, result_l, det_bbox,quad_edge,edge_point,centroid = r2hpl(det_rbox, height, width,class_id,edges,mask)
                        quad_points.append(list(np.array(det_rbox).reshape(4,2).astype(np.int32)))
                        quad_edges.append(quad_edge)
                        if len(edge_point)!=0:
                            edge_points.append(edge_point)
                        if len(centroid)!=0:
                            centroids2.append(centroid)
                        # 融合所有的结果，作为array输入
                        masks, a, b = predictor.predict(
                            point_coords=result_p,
                            point_labels=result_l,
                            box=result_h,
                            multimask_output=False,
                        )

                        result_single_image_single_ann["image_id"] = int(img_id)
                        result_single_image_single_ann["score"] = float(a[0])
                        result_single_image_single_ann["category_id"] = class_id
                        result_single_image_single_ann["bbox"] = [float(item) for item in det_bbox]
                        result_single_image_single_ann["id"] = int(num_ann)
                        num_ann = num_ann + 1

                        # 挑选连通域部分
                        # 将 mask 转换为 uint8 类型，因为 OpenCV 的函数需要 uint8 类型的输入
                        mask_uint8 = (masks[0] * 255).astype(np.uint8)

                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                        # 找到最大的连通域（忽略背景，背景的标签是 0）
                        if stats.shape[0] > 1:  # 确保存在前景连通域（stats[0]是背景）
                            areas = stats[1:, cv2.CC_STAT_AREA]
                            if areas.size > 0:  # 检查数组非空
                                max_label = 1 + np.argmax(areas)
                            else:
                                max_label = 0  # 无连通域时返回背景或默认值
                        else:
                            max_label = 0  # 无连通域时返回背景或默认值 # 
                        # 提取最大连通域的掩码，值为 1 的部分表示最大连通域，其余部分为 0
                        max_connected_component = (labels == max_label).astype(np.uint8)
                        masks[0] = max_connected_component


                        # 使用 OpenCV 的 findContours 函数找到边缘
                        contours, _ = cv2.findContours(masks[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        # 提取边缘点的坐标
                        segmentation_point = []
                        for contour in contours:
                            for point in contour:
                                x, y = point[0]
                                segmentation_point.append(x)
                                segmentation_point.append(y)

                        segmentation_point = [float(item) for item in segmentation_point]
                        points_2d = [(segmentation_point[i], segmentation_point[i + 1]) for i in range(0, len(segmentation_point), 2)]
                        sorted_points_2d = sort_points_clockwise(points_2d)
                        segmentation_point = [coord for point in sorted_points_2d for coord in point]
                        
                        result_single_image_single_ann['segmentation'] = [segmentation_point]
                        with open(submit_path, "a") as file:  # 以追加模式打开文件
                            if num_ann > 2:  # 如果不是第一个元素，先写入逗号
                                file.write(",\n")
                            json.dump(result_single_image_single_ann, file)  # 写入字典
                    
                        result.append(result_single_image_single_ann)
                    predictor.reset_image() 
                    if num_count<=500:
                        visualize_results(filename,image,quad_points,quad_edges,edge_points,centroids2)
        else:
            print(f"文件不存在: {file_path}") 

    # 写入右括号
    with open(submit_path, "a") as file:
        file.write("\n]")
    print(f"数据已成功写入到 {submit_path}")

    # # 打开文件并写入 JSON 数据
    # with open(submit_path, "w", encoding="utf-8") as file:
    #     json.dump(result, file, indent=4, ensure_ascii=False)
    # print(f"数据已成功写入到 {submit_path}")


if __name__ == "__main__":
    sam_path = 'sam_vit_h_4b8939.pth'
    img_path = 'instances/test/images'
    # 定义文件夹路径
    folder_path = 'instances/test/annfile_28epoch_15'  # 替换为你的文件夹路径
    submit_path = "instances/test/7_result.json"
    save_submit_file(sam_path, img_path, folder_path, submit_path)








# # 以下ly
# # contours是元组
# vis_contours(masks, contours)
# # 闭合图形
# segmentation_point = []
# for contour in contours:
#     epsilon = 0.01 * cv2.arcLength(contour, True)  # 简化精度（值越小越精确）
#     approx = cv2.approxPolyDP(contour, epsilon, True)  # 多边形简化
#     points = approx.reshape(-1, 2)  # 获取点并展平为 (n, 2) 的形状
#     if len(points) > 2:  # 有效多边形需至少3个点
#         # if not np.array_equal(points[0], points[-1]):  # 确保闭合
#         #     points = np.vstack((points, points[0]))
#         segmentation_point.append(points.flatten().tolist())  # 展平为 [x1, y1, x2, y2, ...] 格式
# vis_segmentation_point(masks, segmentation_point)