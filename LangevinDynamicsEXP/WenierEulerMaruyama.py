from torch import randn, sqrt, linspace, tensor, zeros, vmap, where, rand, exp, float64
import matplotlib.pyplot as plt
from PathManager import PathManager
import pandas as pd

"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
pm = PathManager()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


def p0(x):
    return (-1 / (1 + exp(-100 * (x - 0.9)))) + (1 / (1 + exp(-100 * (x - 0.1))))


def sample_p0(N):
    samples = []
    attempts = 0
    max_attempts = 100 * N  # 防止死循环
    while len(samples) < N and attempts < max_attempts:
        x = rand(1)  # 从 Uniform(0,1) 中采样
        u = rand(1)
        if u < p0(x):
            samples.append(x.item())
        attempts += 1
    if len(samples) < N:
        raise RuntimeError("采样未成功收敛，请增加 max_attempts 或检查 p0(x)")
    return tensor(samples)


"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
T = (0, 0.1)
X = (0, 1)
ε = 1
NT = 100
NX = 200
N = 100000
dt = tensor((T[1] - T[0]) / NT, dtype=float64)
x_n = zeros(NT, N)
x_n[0] = sample_p0(N)
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
x = linspace(X[0], X[1], NX, dtype=float64)
t = linspace(T[0], T[1], NT, dtype=float64)
print("asgg", x[:7], t[:7], sep="\n")
print("askj", x.shape, t.shape, sep="\n")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


def fold(x):
    return where(x > 1, 2 - x, where(x < 0, -x, x))


"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
for n in range(NT - 1):
    ξ = randn(N)  # 产生 dim 维标准正态随机数
    diffusion = ε * sqrt(dt) * ξ
    x_n[n + 1] = x_n[n] + diffusion
    x_n[n + 1] = vmap(fold)(x_n[n + 1])
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
plt.plot(t, x_n)
# plt.show()
plt.savefig(pm.get_data_path("rdwalk", ".png"))
plt.close()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
plt.hist(x_n[-1, :].numpy(), bins=50, range=(0, 1), density=True, alpha=0.7)
plt.xlabel("Position")
plt.ylabel("Probability Density")
plt.title("Histogram of particles at final time step")
plt.grid(True)
plt.savefig(pm.get_data_path("distribution", ".png"))
plt.close()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
plt.hist(x_n[0, :].numpy(), bins=50, range=(0, 1), density=True, alpha=0.7)
plt.xlabel("Position")
plt.ylabel("Probability Density")
plt.title("Histogram of particles at final time step")
plt.grid(True)
plt.savefig(pm.get_data_path("distribution0", ".png"))
plt.close()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
fig, ax = plt.subplots()
pd.Series(x_n[0, :]).plot(kind="kde", ax=ax, linewidth=2, label="KDE")
plt.savefig(pm.get_data_path("distribution0smooth", ".png"))
plt.close()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
fig, ax = plt.subplots()
ax.set_ylim(0, 1.3)
ax.set_xlim(0, 1)
pd.Series(x_n[-1, :]).plot(kind="kde", ax=ax, linewidth=2, label="KDE")
plt.savefig(pm.get_data_path("distributionsmooth", ".png"))
plt.close()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
fig, ax = plt.subplots()
ax.set_ylim(0, 1.3)
ax.set_xlim(0, 1)
plt.hist(x_n[0, :].numpy(), bins=50, range=(0, 1), density=True, alpha=0.7)
plt.hist(x_n[-1, :].numpy(), bins=50, range=(0, 1), density=True, alpha=0.7)
pd.Series(x_n[-1, :]).plot(kind="kde", ax=ax, linewidth=2, label="KDE")
pd.Series(x_n[0, :]).plot(kind="kde", ax=ax, linewidth=2, label="KDE")
plt.savefig(pm.get_data_path("distributioncombine", ".png"))
plt.close()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
