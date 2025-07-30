import matplotlib.pyplot as plt
import numpy as np

img = plt.imread("Noisy-Image.png")  # 支持 PNG、JPG 等
print(img.shape)  # 如 (H, W, 3) 或 (H, W, 4)
