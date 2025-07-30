import numpy as np
from PIL import Image


def npy_to_binary_image(npy_file_path, output_image_path, show_image=False):

    try:
        data = np.load(npy_file_path)

        # 转换为uint8类型矩阵.以便复原
        matrix = np.array(data, dtype=np.uint8)

        # 转换为黑白图像
        binary_image = Image.fromarray(matrix, mode="L")

        # 保存图像
        binary_image.save(output_image_path)
        print(f"图像已成功保存至: {output_image_path}")

        # 显示图像（如果需要）
        if show_image:
            binary_image.show()

        # 输出验证信息，也可以不显示
        print(f"矩阵形状：{matrix.shape}")
        print(f"图像尺寸：{binary_image.size}")

        return binary_image  # 返回图像对象，方便后续处理

    except FileNotFoundError:
        print(f"错误：找不到文件 {npy_file_path}")
    except Exception as e:
        print(f"处理过程中发生错误：{str(e)}")

if __name__=="__main__":
    npy_to_binary_image("matrix.npy","2n.png")
