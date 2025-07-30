import numpy as np

"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
logging:
    - logging
"""
import logging
from logging.handlers import RotatingFileHandler

# ── 设置专属 logger ────────────────────────────────
logger = logging.getLogger("activeLOG")  # 给它命名，避免 root logger 混淆
logger.setLevel(logging.INFO)
logger.propagate = False  # 🚫 不传播给 root logger，避免 print 出现在 terminal
# 确保 log 路径存在
handler = RotatingFileHandler(
    filename=str("Haar2D.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=10,
    encoding="utf-8",
)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
handler.setFormatter(formatter)
# 清理旧 handler 避免重复添加
if logger.hasHandlers():
    logger.handlers.clear()
logger.addHandler(handler)
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
logger.info(f"START")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
N = 6
f = np.load("matrix.npy")
logger.info(f"f={f}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
ψ = np.concatenate(
    [
        np.sqrt(2 ** (-N)) * np.ones(2 ** (N - 1)),
        -np.sqrt(2 ** (-N)) * np.ones(2 ** (N - 1)),
    ]
)
logger.info(f"ψ={ψ}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
φ = np.concatenate(
    [
        np.sqrt(2 ** (-N)) * np.ones(2 ** (N - 1)),
        np.sqrt(2 ** (-N)) * np.ones(2 ** (N - 1)),
    ]
)
logger.info(f"φ={φ}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
logger.info(f"⟨ψ, ψ⟩={np.sum(np.outer(ψ,ψ)**2)}")
"""check ⟨ψ, ψ⟩=1
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
f_hat = np.zeros([2**N, 2**N])
f_hat[0][0] = np.sum(f * np.outer(φ, φ))
logger.info(f"f_hat[0][0]={f_hat[0][0]}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


def DEC2DY(a, b, l, r, j, k, m, n, f_hat, f):
    if r - l == 1:
        return
    c = 0
    for i in range(a, b):
        for s in range(l, r):
            c += (
                f[i][s]
                * np.sqrt(2**j)
                * ψ[2**j * i - 2**N * k]
                * np.sqrt(2**m)
                * ψ[2**m * s - 2**N * n]
            )
    f_hat[2**j + k][2**m + n] = c
    DEC2DY(a, b, l, (l + r) // 2, j, k, m + 1, n * 2 + 0, f_hat, f)
    DEC2DY(a, b, (l + r) // 2, r, j, k, m + 1, n * 2 + 1, f_hat, f)
    pass


def DEC2DX(a, b, j, k, f_hat, f):
    if b - a == 1:
        return
    DEC2DY(a, b, 0, 2**N, j, k, 0, 0, f_hat, f)
    DEC2DX(a, (a + b) // 2, j + 1, k * 2 + 0, f_hat, f)
    DEC2DX((a + b) // 2, b, j + 1, k * 2 + 1, f_hat, f)


def DEC2DEdge(a, b, j, k, f_hat, f):
    if b - a == 1:
        return
    ca = 0
    cc = 0
    for i in range(a, b):
        for s in range(0, 2**N):
            ca += f[i][s] * np.sqrt(2**j) * ψ[2**j * i - 2**N * k] * φ[s]
            cc += f[s][i] * np.sqrt(2**j) * φ[s] * ψ[2**j * i - 2**N * k]
    f_hat[2**j + k][0] = ca
    f_hat[0][2**j + k] = cc
    DEC2DEdge(a, (a + b) // 2, j + 1, k * 2 + 0, f_hat, f)
    DEC2DEdge((a + b) // 2, b, j + 1, k * 2 + 1, f_hat, f)


logger.info(f"f_hat={f_hat}")
DEC2DEdge(0, 2**N, 0, 0, f_hat, f)
DEC2DX(0, 2**N, 0, 0, f_hat, f)
logger.info(f"f_hat={f_hat}")
"""[a,b) c_{j,k}
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
f_rec = np.outer(φ, φ) * f_hat[0][0]
logger.info(f"f_rec_0={f_rec}")


def REC2DY(a, b, l, r, j, k, m, n, f_hat, f):
    if r - l == 1:
        return
    c = f_hat[2**j + k][2**m + n]
    for i in range(a, b):
        for s in range(l, r):
            f[i][s] += (
                c
                * np.sqrt(2**j)
                * ψ[2**j * i - 2**N * k]
                * np.sqrt(2**m)
                * ψ[2**m * s - 2**N * n]
            )
    REC2DY(a, b, l, (l + r) // 2, j, k, m + 1, n * 2 + 0, f_hat, f)
    REC2DY(a, b, (l + r) // 2, r, j, k, m + 1, n * 2 + 1, f_hat, f)
    pass


def REC2DX(a, b, j, k, f_hat, f):
    if b - a == 1:
        return
    REC2DY(a, b, 0, 2**N, j, k, 0, 0, f_hat, f)
    REC2DX(a, (a + b) // 2, j + 1, k * 2 + 0, f_hat, f)
    REC2DX((a + b) // 2, b, j + 1, k * 2 + 1, f_hat, f)


def REC2DEdge(a, b, j, k, f_hat, f):
    if b - a == 1:
        return
    ca = f_hat[2**j + k][0]
    cc = f_hat[0][2**j + k]
    for i in range(a, b):
        for s in range(0, 2**N):
            f[i][s] += ca * np.sqrt(2**j) * ψ[2**j * i - 2**N * k] * φ[s]
            f[s][i] += cc * np.sqrt(2**j) * φ[s] * ψ[2**j * i - 2**N * k]
    REC2DEdge(a, (a + b) // 2, j + 1, k * 2 + 0, f_hat, f)
    REC2DEdge((a + b) // 2, b, j + 1, k * 2 + 1, f_hat, f)


REC2DEdge(0, 2**N, 0, 0, f_hat, f_rec)
REC2DX(0, 2**N, 0, 0, f_hat, f_rec)
logger.info(f"f_rec={f_rec}")
logger.info(f"f={f}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
for i in range(2**N):
    for j in range(2**N):
        if np.abs(f_hat[i][j]) < 0.1:
            f_hat[i][j] = 0
print(f_hat)
f_rec_comp = np.outer(φ, φ) * f_hat[0][0]
REC2DEdge(0, 2**N, 0, 0, f_hat, f_rec_comp)
REC2DX(0, 2**N, 0, 0, f_hat, f_rec_comp)
logger.info(f"f_rec_comp={f_rec_comp}")
logger.info(f"f={f}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
from npytogrey import npy_to_binary_image

np.save("f_rec_comp", f_rec_comp)
npy_to_binary_image("f_rec_comp.npy", "denoise.png")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
