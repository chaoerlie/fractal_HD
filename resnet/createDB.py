import cv2
import numpy as np
import os
import math
from skimage import measure


# 计算Box-counting维数的函数
def box_count(image, box_size):
    # Convert to binary image if necessary
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to make it binary (0 or 255)
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Image dimensions
    height, width = binary_image.shape

    # Initialize box count
    count = 0

    # Slide a box of size (box_size, box_size) across the image
    for y in range(0, height, box_size):
        for x in range(0, width, box_size):
            # Extract a box from the image
            box = binary_image[y:y + box_size, x:x + box_size]
            if np.any(box):  # If any pixel in the box is 1, count this box
                count += 1

    return count


# 计算 Hausdorff 维数的函数
def hausdorff_dimension(image, max_box_size=64, step=2):
    sizes = []
    counts = []

    # Iterate through different box sizes
    for box_size in range(1, max_box_size + 1, step):
        count = box_count(image, box_size)
        sizes.append(box_size)
        counts.append(count)

    # Fit a line in log-log space
    log_sizes = np.log(sizes)
    log_counts = np.log(counts)

    # Perform linear regression (fit a line)
    slope, intercept = np.polyfit(log_sizes, log_counts, 1)

    # Hausdorff dimension is the negative of the slope
    hd = -slope
    return hd


if __name__ == '__main__':
    # 处理2000张图片
    source_folder = 'dataset'  # 修改为B文件夹路径
    hd_values = []

    # 遍历每一张图片并计算HD
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            # 读取图片
            img_path = os.path.join(source_folder, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # 计算HD
            hd_value = hausdorff_dimension(image)
            hd_values.append(hd_value)
            print(f"HD for {filename}: {hd_value}")

    # 将HD值保存为训练数据集标签
    # 例如将结果保存为txt文件
    with open('hd_values.txt', 'w') as f:
        for hd_value in hd_values:
            f.write(f"{hd_value}\n")
