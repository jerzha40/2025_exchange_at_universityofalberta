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
    filename=str("Haar1DFAST.log"),
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
logger.info(f"\n")
logger.info(f"START")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
N = 10
f = np.load("matrix.npy")
logger.info(f"f={f}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
ψ = np.array([1 / 2, np.sqrt(3) / 2])
logger.info(f"ψ={ψ}")
φ = np.array([np.sqrt(3) / 2, -1 / 2])
logger.info(f"φ={φ}")
"""ψ & φ
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
ψψ = np.outer(ψ, ψ)
logger.info(f"ψψ={ψψ}")
φφ = np.outer(φ, φ)
logger.info(f"φφ={φφ}")
φψ = np.outer(φ, ψ)
logger.info(f"φψ={φψ}")
ψφ = np.outer(ψ, φ)
logger.info(f"ψφ={ψφ}")
"""ψ & φ
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
logger.info(f"⟨ψ, ψ⟩={np.sum(ψ**2)}")
logger.info(f"⟨φ, φ⟩={np.sum(φ**2)}")
logger.info(f"⟨ψψ, ψψ⟩={np.sum(ψψ**2)}")
logger.info(f"⟨φφ, φφ⟩={np.sum(φφ**2)}")
logger.info(f"⟨φψ, φψ⟩={np.sum(φψ**2)}")
logger.info(f"⟨ψφ, ψφ⟩={np.sum(ψφ**2)}")
"""check ⟨ψ, ψ⟩=1
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
f_hat = np.zeros([2**N, 2**N])
logger.info(f"f_hat={f_hat}")
# f_hat[0] = np.sum(f * φ)
# logger.info(f"f_hat[0]={f_hat[0]}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


def HaarDWTDEC2D(f: np.ndarray):
    f_hat = np.zeros([2**N, 2**N])
    for i in range(N):
        f = f.reshape(2 ** (N - i - 1), 2, 2 ** (N - i - 1), 2)
        # logger.info(f"f={f}")
        f_hat[2 ** (N - i - 1) : 2 ** (N - i), 2 ** (N - i - 1) : 2 ** (N - i)] = (
            np.einsum("ikjl,kl->ij", f, ψψ)
        )
        f_hat[0 : 2 ** (N - i - 1), 2 ** (N - i - 1) : 2 ** (N - i)] = np.einsum(
            "ikjl,kl->ij", f, φψ
        )
        f_hat[2 ** (N - i - 1) : 2 ** (N - i), 0 : 2 ** (N - i - 1)] = np.einsum(
            "ikjl,kl->ij", f, ψφ
        )
        f = np.einsum("ikjl,kl->ij", f, φφ)
        # logger.info(f"f={f}")
        # logger.info(f"f_hat={f_hat}")
    f_hat[0][0] = f[0][0]
    return f_hat


"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
IPUT_f = f
logger.info(f"IPUT_f={IPUT_f}")
OPUT_f = HaarDWTDEC2D(IPUT_f)
logger.info(f"OPUT_f={OPUT_f}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


def HaarDWTREC2D(f_hat):
    f = f_hat[0][0]
    for i in range(N):
        f = np.kron(f, φφ)
        # logger.info(f"f={f}")
        f += np.kron(
            f_hat[2 ** (0 + i) : 2 ** (1 + i), 2 ** (0 + i) : 2 ** (1 + i)], ψψ
        )
        f += np.kron(f_hat[0 : 2 ** (0 + i), 2 ** (0 + i) : 2 ** (1 + i)], φψ)
        f += np.kron(f_hat[2 ** (0 + i) : 2 ** (1 + i), 0 : 2 ** (0 + i)], ψφ)
        # logger.info(f"f={f}")
    return f


"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
print("s;dfkaj")
RCON_f = HaarDWTREC2D(OPUT_f)
logger.info(f"RCON_f={RCON_f}")
logger.info(f"IPUT_f={IPUT_f}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
# threshold = np.percentile(OPUT_f, 0)
# OPUT_f_filtered = np.where(OPUT_f > threshold, OPUT_f, 0)
# RCON_f_filtered = HaarDWTREC2D(OPUT_f_filtered)
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """
filtered = np.zeros_like(OPUT_f)
filtered = OPUT_f
filtered[0:256, 0:256] = np.zeros_like(OPUT_f[0:256, 0:256])
RCON_f_filtered = HaarDWTREC2D(filtered)
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
from npytogrey import npy_to_binary_image

np.save("denoise", RCON_f_filtered)
npy_to_binary_image("denoise.npy", "denoise.png")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
