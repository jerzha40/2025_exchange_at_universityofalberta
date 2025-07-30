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
N = 27
f = np.random.random(2**N)
logger.info(f"f={f}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
ψ = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])
logger.info(f"ψ={ψ}")
φ = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
logger.info(f"φ={φ}")
"""ψ & φ
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
logger.info(f"⟨ψ, ψ⟩={np.sum(ψ**2)}")
"""check ⟨ψ, ψ⟩=1
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
f_hat = np.zeros(2**N)
logger.info(f"f_hat={f_hat}")
# f_hat[0] = np.sum(f * φ)
# logger.info(f"f_hat[0]={f_hat[0]}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
# f = f.reshape(-1, 2)
# logger.info(f"f={f}")
# f_hat[2 ** (N - 1) :] = f @ ψ
# f = f @ φ
# logger.info(f"f={f}")
# logger.info(f"f_hat={f_hat}")
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """
# f = f.reshape(-1, 2)
# logger.info(f"f={f}")
# f_hat[2 ** (N - 2) : 2 ** (N - 1)] = f @ ψ
# f = f @ φ
# logger.info(f"f={f}")
# logger.info(f"f_hat={f_hat}")
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """
# f = f.reshape(-1, 2)
# logger.info(f"f={f}")
# f_hat[2 ** (N - 3) : 2 ** (N - 2)] = f @ ψ
# f = f @ φ
# logger.info(f"f={f}")
# logger.info(f"f_hat={f_hat}")
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """
# f = f.reshape(-1, 2)
# logger.info(f"f={f}")
# f_hat[2 ** (N - 4) : 2 ** (N - 3)] = f @ ψ
# f = f @ φ
# logger.info(f"f={f}")
# logger.info(f"f_hat={f_hat}")
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """
# f_hat[0] = f
# logger.info(f"f_hat={f_hat}")
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """


def HaarDWT(f):
    f_hat = np.zeros(2**N)
    for i in range(N):
        f = f.reshape(-1, 2)
        logger.info(f"f={f}")
        f_hat[2 ** (N - i - 1) : 2 ** (N - i)] = f @ ψ
        f = f @ φ
        logger.info(f"f={f}")
        logger.info(f"f_hat={f_hat}")
    f_hat[0] = f[0]
    return f_hat


"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
IPUT_f = np.random.random(2**N)
logger.info(f"IPUT_f={IPUT_f}")
OPUT_f = HaarDWT(IPUT_f)
logger.info(f"OPUT_f={OPUT_f}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


def HaarDWTREC(f_hat):
    f = f_hat[0]
    for i in range(N):
        f = np.kron(f, φ)
        logger.info(f"f={f}")
        f += np.kron(f_hat[2 ** (0 + i) : 2 ** (1 + i)], ψ)
        logger.info(f"f={f}")
    return f


"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
RCON_f = HaarDWTREC(OPUT_f)
logger.info(f"RCON_f={RCON_f}")
logger.info(f"IPUT_f={IPUT_f}")
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
