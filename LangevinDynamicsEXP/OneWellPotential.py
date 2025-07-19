import matplotlib.pyplot as plt
from torch import load, save, linspace, meshgrid, float64, stack, vmap
from torch import pi, exp, sqrt, tensor, nn, zeros_like
from torch.optim import Adam
from torch.func import jacrev
from PathManager import PathManager

T = (0, 1)
X = (-10, 10)
ε = 1
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


def b(x):
    return x**4


def p0(x, x0=5, sigma=5):
    mu = x0
    return exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * sqrt(tensor(2 * pi)))


"""(del_x)p=ε^2/2(del_x)^2p-(del_x)(bp)
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
pm = PathManager()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
p = load(pm.get_data_path("p", ".ntdt"), weights_only=False)
"""p, p(x,t), probabilistic distribution over time
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
x = linspace(X[0], X[1], 200, dtype=float64)
t = linspace(T[0], T[1], 10, dtype=float64)
print(x[100:105], t[:10], sep="\n")
print(x.shape, t.shape, sep="\n")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
mx, mt = meshgrid(x, t, indexing="ij")  # shape = [20, 20, 10]
print(mx[:5, :5], mt[:5, :5], sep="\n")
print(mx.shape, mt.shape, sep="\n")
assert mx.shape == mt.shape
"""generate meshgrid (x,t)
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
grids = stack([mx, mt], dim=2)
print(grids.shape)
print(grids[:5, :5, :5])
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
grids_shape = grids.shape
points = grids.reshape(-1, 2)
print(points.shape)
print(points[:5, :5])
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


def f_p(x, t):
    U = stack((x, t))
    return p(U)


def bp(x, t):
    return f_p(x, t) * b(x)


"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
d0p0 = vmap(p0)(x)
d0b = vmap(b)(points[:, 0])
d0p = vmap(f_p)(points[:, 0], points[:, 1])
print(d0p.shape)
print(d0p[:5, :5])
print(d0b.shape)
print(d0p0.shape)
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
grids_d0p = d0p.reshape((200, 10))  # 自动还原
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
plt.plot(x, d0p0)
plt.savefig(pm.get_data_path("d0p0", ".png"))
plt.plot(points[:, 0], d0b)
plt.savefig(pm.get_data_path("d0b", ".png"))
print(x.shape, grids_d0p.shape)
plt.plot(x, grids_d0p.cpu().detach().numpy()[:, 0])
plt.savefig(pm.get_data_path("d0p", ".png"))
plt.close()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
d1p_dt1 = vmap(jacrev(f_p, argnums=1))(points[:, 0], points[:, 1])
d1p_dx1 = vmap(jacrev(f_p, argnums=0))(points[:, 0], points[:, 1])
d2p_dx2 = vmap(jacrev(jacrev(f_p, argnums=0), argnums=0))(points[:, 0], points[:, 1])
d1bp_dx1 = vmap(jacrev(bp, argnums=0))(points[:, 0], points[:, 1])
print(d1p_dt1.shape)
print(d1p_dx1.shape)
print(d2p_dx2.shape)
print(d1bp_dx1.shape)
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


def Equation(points):
    d1p_dt1 = vmap(jacrev(f_p, argnums=1))(points[:, 0], points[:, 1])
    d2p_dx2 = vmap(jacrev(jacrev(f_p, argnums=0), argnums=0))(
        points[:, 0], points[:, 1]
    )
    d1bp_dx1 = vmap(jacrev(bp, argnums=0))(points[:, 0], points[:, 1])
    return d1p_dt1 - (ε**2 / 2) * d2p_dx2 + d1bp_dx1


equ = Equation(points=points)
print(equ.shape)
print(equ[:5])
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
optimizer = Adam(p.parameters(), lr=5e-5)
mse = nn.MSELoss()
for i in range(10000):
    optimizer.zero_grad()
    equ = Equation(points=points)
    grids_d0p = d0p.reshape((200, 10))  # 自动还原
    d0p_ = vmap(f_p)(mx[:, 0], mt[:, 0])
    loss = +10 * mse(d0p_.view(-1), d0p0.view(-1)) + 0.1 * mse(equ, zeros_like(equ))
    loss.backward()
    optimizer.step()
    print(f"{i:5d} {loss.tolist():.2e}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
d0p0 = vmap(p0)(x)
d0b = vmap(b)(points[:, 0])
d0p = vmap(f_p)(points[:, 0], points[:, 1])
print(d0p.shape)
print(d0p[:5, :5])
print(d0b.shape)
print(d0p0.shape)
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
grids_d0p = d0p.reshape((200, 10))  # 自动还原
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
plt.plot(x, d0p0)
plt.savefig(pm.get_data_path("d0p0", ".png"))
plt.close()
plt.plot(points[:, 0], d0b)
plt.savefig(pm.get_data_path("d0b", ".png"))
plt.close()
print(x.shape, grids_d0p.shape)
plt.plot(x, grids_d0p.cpu().detach().numpy()[:, 0])
plt.savefig(pm.get_data_path("d0p_0", ".png"))
plt.close()
plt.plot(x, grids_d0p.cpu().detach().numpy()[:, 9])
plt.savefig(pm.get_data_path("d0p_9", ".png"))
plt.close()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
save(p, pm.get_data_path("p", ".ntdt"))
"""save
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
