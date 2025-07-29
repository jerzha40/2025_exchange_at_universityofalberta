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
    filename=str("Haar1D.log"),
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
N = 4
f = np.random.random(2**N)
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
logger.info(f"⟨ψ, ψ⟩={np.sum(ψ**2)}")
"""check ⟨ψ, ψ⟩=1
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
f_hat = np.zeros(2**N)
f_hat[0] = np.sum(f * φ)
logger.info(f"f_hat[0]={f_hat[0]}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


def DEC(a, b, j, k):
    if b - a == 1:
        return
    c = 0
    for i in range(a, b):
        c += f[i] * np.sqrt(2**j) * ψ[2**j * i - 2**N * k]
        f_hat[2**j + k] = c
    DEC(a, (a + b) // 2, j + 1, k * 2)
    DEC((a + b) // 2, b, j + 1, k * 2 + 1)
    pass


DEC(0, 2**N, 0, 0)
logger.info(f"f_hat={f_hat}")
"""[a,b) c_{j,k}
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
f_rec = φ * f_hat[0]
logger.info(f"f_rec_0={f_rec}")


def REC(a, b, j, k, f):
    if b - a == 1:
        return
    c = f_hat[2**j + k]
    for i in range(a, b):
        f[i] += c * np.sqrt(2**j) * ψ[2**j * i - 2**N * k]
    REC(a, (a + b) // 2, j + 1, k * 2, f)
    REC((a + b) // 2, b, j + 1, k * 2 + 1, f)
    pass


REC(0, 2**N, 0, 0, f_rec)
logger.info(f"f_rec={f_rec}")
logger.info(f"f={f}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
for i in range(2**N):
    if np.abs(f_hat[i]) < 0.1:
        print(i)
        f_hat[i] = 0
print(f_hat)
f_rec_comp = φ * f_hat[0]
REC(0, 2**N, 0, 0, f_rec_comp)
logger.info(f"f_rec_comp={f_rec_comp}")
logger.info(f"f={f}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
