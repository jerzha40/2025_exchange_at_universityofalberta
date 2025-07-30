import math
from PIL import Image
import numpy as np

N = int(input("enter the photo size"))


def convert_image(image_path, output_path, N, save_matrix=False, matrix_save_path=None):
    """
    将输入图片转换为 2^N×2^N 像素的黑白（灰度）图片
    :param image_path: 输入图片的路径（如 "input.png"）
    :param output_path: 转换后图片的保存路径（如 "output.jpg"）
    :param N: 用户指定的整数，用于确定输出尺寸为 2^N×2^N 像素
    """
    # 打开图片并转换为灰度（黑白）
    with Image.open(image_path) as img:
        gray_img = img.convert("L")  # 转换为单通道灰度图，实现黑白效果

        # 基于用户给定的 N 计算目标尺寸
        target_size = 2**N  # 输出图片的边长为 2 的 N 次方

        # 等比例缩放图片，确保缩放后至少有一边达到目标尺寸
        width, height = gray_img.size
        if width > height:
            new_width = int(width * (target_size / height))
            new_height = target_size
        else:
            new_width = target_size
            new_height = int(height * (target_size / width))
        # 使用 LANCZOS 插值算法缩放，保持图像清晰度
        resized_img = gray_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        cropped_img = resized_img.crop((left, top, right, bottom))  # 执行裁剪

        cropped_img.save(output_path)
        full_matrix = np.array(cropped_img)
        np.save("matrix", full_matrix)
    # 输出转换完成信息及结果尺寸
    print(f"转换完成，输出尺寸：{target_size}×{target_size} 像素（基于 N={N} 计算）")
    print(full_matrix)


# 示例调用（使用时替换为实际路径和 N 值）
convert_image("fig/Noisy-Image.png", "output_image.png", N)
