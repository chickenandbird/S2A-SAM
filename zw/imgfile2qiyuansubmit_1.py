import os
from segment_anything import sam_model_registry, SamPredictor
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def r2hpl(r_list, height, width):   # list   -6.07 103.44 121.07 102.87 121.44 184.98 -5.70 185.55 
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

    result_h = np.array([min_x, min_y, max_x, max_y])
    # if min_x ==0 or min_y == 0 or max_x == width-1 or max_y == height-1:
    #     result_p = np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]])
    #     result_l = np.array([0,0,0,0])
    # else:
    result_p = np.array([[(min_x+max_x)/2, (min_y+max_y)/2], [min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]])
    result_l = np.array([1,0,0,0,0])

    # result_p = np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]])
    # result_l = np.array([0,0,0,0])
    det_bbox = [min_x, min_y, max_x-min_x, max_y-min_y]
    return result_h, result_p, result_l, det_bbox

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
    for filename in ['101.txt','1027.txt','1058.txt','1059.txt','1068.txt','1123.txt','1151.txt','1157.txt','10011.txt','10109.txt','10116.txt','10220.txt','10225.txt','10292.txt']:
    # for filename in os.listdir(folder_path):   # '1.txt'
        if filename.endswith('.txt'):
            # 打开并读取文件
            print(f"{num_count}--{filename}")
            num_count += 1
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:   # 读取1.txt，即图片下面的所有检测结果 
                    # 提取文件名部分（去掉后缀）
                    img_id = filename.split('.')[0]   # '1'
                    image_path = os.path.join(img_path, img_id+'.png')
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    predictor.set_image(image)
                    
                    # 每一行都是这个图片的标注
                    for line in f:  # 60.53 234.26 121.40 212.64 127.80 230.63 66.92 252.25 7
                        result_single_image_single_ann = {}
                        # 去掉行首行尾的空白字符，并按空格分割成列表
                        elements = line.strip().split()
                        if len(elements) < 9:  # 确保每行至少有9个元素
                            continue

                        class_id = int(elements[8])  
                        det_rbox = [float(element) for element in elements[:8]]
                        
                        height, width, channels = image.shape
                        result_h, result_p, result_l, det_bbox = r2hpl(det_rbox, height, width)
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
                        # 使用 cv2.connectedComponentsWithStats 找到所有连通域及其统计信息
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                        # 找到最大的连通域（忽略背景，背景的标签是 0）
                        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 从非背景标签中找到面积最大的连通域
                        # 提取最大连通域的掩码，值为 1 的部分表示最大连通域，其余部分为 0
                        max_connected_component = (labels == max_label).astype(np.uint8)
                        masks[0] = max_connected_component


                        # 使用 OpenCV 的 findContours 函数找到边缘
                        contours, _ = cv2.findContours(masks[0].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                        # import pdb
                        # pdb.set_trace()
                        
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
                        
                        # vis_segmentation_point(masks, [segmentation_point])
                        # import pdb
                        # pdb.set_trace()

                        result_single_image_single_ann['segmentation'] = [segmentation_point]
                        with open(submit_path, "a") as file:  # 以追加模式打开文件
                            if num_ann > 2:  # 如果不是第一个元素，先写入逗号
                                file.write(",\n")
                            json.dump(result_single_image_single_ann, file)  # 写入字典

                        result.append(result_single_image_single_ann)
            except:
                continue
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
    folder_path = 'instances/test/annfile_2'  # 替换为你的文件夹路径
    submit_path = "instances/test/1_result.json"
    save_submit_file(sam_path, img_path, folder_path, submit_path)