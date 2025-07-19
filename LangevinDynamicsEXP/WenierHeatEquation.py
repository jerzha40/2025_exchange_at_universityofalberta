import matplotlib.pyplot as plt
from torch import load, save, linspace, meshgrid, float64, stack, vmap
from torch import exp, nn, zeros_like, max, abs
from torch.optim import Adam
from torch.func import jacrev
from PathManager import PathManager
from FCN import FCN


T = (0, 0.1)
X = (0, 1)
ε = 1
NT = 100
NX = 200
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


def b(x):
    return x**4


def p0(x):
    return (-1 / (1 + exp(-100 * (x - 0.9)))) + (1 / (1 + exp(-100 * (x - 0.1))))


"""(del_x)p=ε^2/2(del_x)^2p-(del_x)(bp)
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
pm = PathManager()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
p = load(pm.get_data_path("p", ".ntdt"), weights_only=False)
# p = FCN(2, 1, 16, 4)
p.to("cuda")
"""p, p(x,t), probabilistic distribution over time
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
x = linspace(X[0], X[1], NX, dtype=float64)
t = linspace(T[0], T[1], NT, dtype=float64)
print("asgg", x[:7], t[:7], sep="\n")
print("askj", x.shape, t.shape, sep="\n")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
mx, mt = meshgrid(x, t, indexing="ij")  # shape = [20, 20, 10]
mx = mx.to("cuda")
mt = mt.to("cuda")
print("ssdf", mx[:7, :7], mt[:7, :7], sep="\n")
print("kasg", mx.shape, mt.shape, sep="\n")
assert mx.shape == mt.shape
"""generate meshgrid (x,t)
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
dxt0 = stack([mx, mt], dim=2)
dxt0 = dxt0.to("cuda")
print("hgsl", dxt0[:7, :7, 0], dxt0[:7, :7, 1], sep="\n")
print("sghd", dxt0.shape, sep="\n")
d0p = vmap(vmap(p))(dxt0)
print("sagk", d0p[:7, :7, :], sep="\n")
print("gljg", d0p.shape, sep="\n")
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(
    mx.detach().cpu().view(NX, NT),
    mt.detach().cpu().view(NX, NT),
    d0p.detach().cpu().view(NX, NT),
    cmap="viridis",
)
plt.title("init_d0p")
plt.savefig(pm.get_data_path("init_d0p", ".png"))
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
d0pt = vmap(vmap(p0))(mx)
ax.plot_surface(
    mx.detach().cpu().view(NX, NT),
    mt.detach().cpu().view(NX, NT),
    d0pt.detach().cpu().view(NX, NT),
    cmap="viridis",
)
plt.title("init_d0p0t")
plt.savefig(pm.get_data_path("init_d0p0t", ".png"))
plt.close()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(
    mx.detach().cpu().view(NX, NT),
    mt.detach().cpu().view(NX, NT),
    d0pt.detach().cpu().view(NX, NT),
    cmap="viridis",
)
plt.savefig(pm.get_data_path("init_d0pt", ".png"))
plt.close()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
optimizer = Adam(p.parameters(), lr=5e-5)
mse = nn.MSELoss()
for i in range(100):
    optimizer.zero_grad()
    d0p0 = vmap(p0)(mx[:, 0])
    d0p = vmap(p)(dxt0[:, 0, :]).view(NX)
    assert d0p0.shape == d0p.shape
    loss = mse(d0p, d0p0)
    loss.backward()
    optimizer.step()
    print(f"{i:5d} {loss.tolist():5.2e}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
d1p_dxt1 = vmap(vmap(jacrev(p)))(dxt0)
d2p_dxt2 = vmap(vmap(jacrev(jacrev(p))))(dxt0)
print("slak", d1p_dxt1.shape, d2p_dxt2.shape, sep="\n")
d2p_dx2 = d2p_dxt2[:, :, 0, 0, 0]
d1p_dt1 = d1p_dxt1[:, :, 0, 1]
print("aglk", d2p_dx2.shape, d1p_dt1.shape, sep="\n")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
# optimizer = Adam(p.parameters(), lr=5e-5)
# mse = nn.MSELoss()
for i in range(5000):
    optimizer.zero_grad()
    d0p0 = vmap(p0)(mx[:, 0])
    d0p = vmap(p)(dxt0[:, 0, :]).view(NX)
    assert d0p0.shape == d0p.shape
    d1p_dxt1 = vmap(vmap(jacrev(p)))(dxt0)
    d2p_dxt2 = vmap(vmap(jacrev(jacrev(p))))(dxt0)
    d2p_dx2 = d2p_dxt2[:, :, 0, 0, 0]
    d1p_dt1 = d1p_dxt1[:, :, 0, 1]
    d1p_dx1_0 = d1p_dxt1[1, :, 0, 0]
    d1p_dx1_1 = d1p_dxt1[-1, :, 0, 0]
    assert d2p_dx2.shape == d1p_dt1.shape
    loss = (
        10 * mse(d0p, d0p0)
        + 0.1 * mse((ε**2 / 2) * d2p_dx2, d1p_dt1)
        + 0.1 * mse(d1p_dx1_0, zeros_like(d1p_dx1_0))
        + 0.1 * mse(d1p_dx1_1, zeros_like(d1p_dx1_1))
    )
    # loss = (
    #     10 * max(abs(d0p - d0p0))
    #     + 0.1 * max(abs(d2p_dx2 - d1p_dt1))
    #     + 0.1 * max(abs(d1p_dx1_0))
    #     + 0.1 * max(abs(d1p_dx1_1))
    # )
    loss.backward()
    optimizer.step()
    print(f"{i:5d} {loss.tolist():5.2e}")
    if i % 100 == 0:
        save(p, pm.get_data_path("p", ".ntdt"))
        d0p = vmap(vmap(p))(dxt0)
        d0pt = vmap(vmap(p0))(mx)
        print("lsag", d0p[:7, :7, :], sep="\n")
        print("sgln", d0p.shape, sep="\n")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            mx.detach().cpu().view(NX, NT),
            mt.detach().cpu().view(NX, NT),
            d0p.detach().cpu().view(NX, NT),
            cmap="viridis",
        )
        ax.plot_surface(
            mx.detach().cpu().view(NX, NT),
            mt.detach().cpu().view(NX, NT),
            d0pt.detach().cpu().view(NX, NT),
        )
        plt.title("d0p0t")
        plt.savefig(pm.get_data_path("d0p0t", ".png"))
        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            mx.detach().cpu().view(NX, NT),
            mt.detach().cpu().view(NX, NT),
            d0p.detach().cpu().view(NX, NT),
            cmap="viridis",
        )
        plt.title("d0p")
        plt.savefig(pm.get_data_path("d0p", ".png"))
        plt.close()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
d0p = vmap(vmap(p))(dxt0)
d0pt = vmap(vmap(p0))(mx)
print("lsag", d0p[:7, :7, :], sep="\n")
print("sgln", d0p.shape, sep="\n")
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(
    mx.detach().cpu().view(NX, NT),
    mt.detach().cpu().view(NX, NT),
    d0p.detach().cpu().view(NX, NT),
    cmap="viridis",
)
ax.plot_surface(
    mx.detach().cpu().view(NX, NT),
    mt.detach().cpu().view(NX, NT),
    d0pt.detach().cpu().view(NX, NT),
)
plt.title("init_d0p0t")
plt.show()
plt.savefig(pm.get_data_path("init_d0p0t", ".png"))
plt.close()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
# grids = stack([mx, mt], dim=2)
# print(grids.shape)
# print(grids[:5, :5, :5])
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """
# grids_shape = grids.shape
# points = grids.reshape(-1, 2)
# print(points.shape)
# print(points[:5, :5])
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """


# def f_p(x, t):
#     U = stack((x, t))
#     return p(U)


# def bp(x, t):
#     return f_p(x, t) * b(x)


# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """
# d0p0 = vmap(p0)(x)
# d0b = vmap(b)(points[:, 0])
# d0p = vmap(f_p)(points[:, 0], points[:, 1])
# print(d0p.shape)
# print(d0p[:5, :5])
# print(d0b.shape)
# print(d0p0.shape)
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """
# grids_d0p = d0p.reshape((200, 10))  # 自动还原
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """
# plt.plot(x, d0p0)
# plt.savefig(pm.get_data_path("d0p0", ".png"))
# plt.plot(points[:, 0], d0b)
# plt.savefig(pm.get_data_path("d0b", ".png"))
# print(x.shape, grids_d0p.shape)
# plt.plot(x, grids_d0p.cpu().detach().numpy().numpy()[:, 0])
# plt.savefig(pm.get_data_path("d0p", ".png"))
# plt.close()
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """
# d1p_dt1 = vmap(jacrev(f_p, argnums=1))(points[:, 0], points[:, 1])
# d1p_dx1 = vmap(jacrev(f_p, argnums=0))(points[:, 0], points[:, 1])
# d2p_dx2 = vmap(jacrev(jacrev(f_p, argnums=0), argnums=0))(points[:, 0], points[:, 1])
# d1bp_dx1 = vmap(jacrev(bp, argnums=0))(points[:, 0], points[:, 1])
# print(d1p_dt1.shape)
# print(d1p_dx1.shape)
# print(d2p_dx2.shape)
# print(d1bp_dx1.shape)
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """


# def Equation(points):
#     d1p_dt1 = vmap(jacrev(f_p, argnums=1))(points[:, 0], points[:, 1])
#     d2p_dx2 = vmap(jacrev(jacrev(f_p, argnums=0), argnums=0))(
#         points[:, 0], points[:, 1]
#     )
#     d1bp_dx1 = vmap(jacrev(bp, argnums=0))(points[:, 0], points[:, 1])
#     return d1p_dt1 - (ε**2 / 2) * d2p_dx2 + d1bp_dx1


# equ = Equation(points=points)
# print(equ.shape)
# print(equ[:5])
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """
# optimizer = Adam(p.parameters(), lr=5e-5)
# mse = nn.MSELoss()
# for i in range(10000):
#     optimizer.zero_grad()
#     equ = Equation(points=points)
#     grids_d0p = d0p.reshape((200, 10))  # 自动还原
#     d0p_ = vmap(f_p)(mx[:, 0], mt[:, 0])
#     loss = +10 * mse(d0p_.view(-1), d0p0.view(-1)) + 0.1 * mse(equ, zeros_like(equ))
#     loss.backward()
#     optimizer.step()
#     print(f"{i:5d} {loss.tolist():.2e}")
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """
# d0p0 = vmap(p0)(x)
# d0b = vmap(b)(points[:, 0])
# d0p = vmap(f_p)(points[:, 0], points[:, 1])
# print(d0p.shape)
# print(d0p[:5, :5])
# print(d0b.shape)
# print(d0p0.shape)
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """
# grids_d0p = d0p.reshape((200, 10))  # 自动还原
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """
# plt.plot(x, d0p0)
# plt.savefig(pm.get_data_path("d0p0", ".png"))
# plt.close()
# plt.plot(points[:, 0], d0b)
# plt.savefig(pm.get_data_path("d0b", ".png"))
# plt.close()
# print(x.shape, grids_d0p.shape)
# plt.plot(x, grids_d0p.cpu().detach().numpy()[:, 0])
# plt.savefig(pm.get_data_path("d0p_0", ".png"))
# plt.close()
# plt.plot(x, grids_d0p.cpu().detach().numpy()[:, 9])
# plt.savefig(pm.get_data_path("d0p_9", ".png"))
# plt.close()
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """
d0p0 = vmap(p0)(x)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, d0p0.detach().cpu() * 1.3)
ax.set_ylim(0, 1.3)
ax.set_xlim(0, 1)
plt.title("p(x,0)")
plt.savefig(pm.get_data_path("p(x,0)", ".png"))
plt.close()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
d0px1 = vmap(vmap(p))(dxt0)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, d0px1[:, -1, :].detach().cpu().view(x.shape) * 1.3)
ax.set_ylim(0, 1.3)
ax.set_xlim(0, 1)
plt.title("p(x,1)")
plt.savefig(pm.get_data_path("p(x,1)", ".png"))
plt.close()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, d0p0.detach().cpu() * 1.3)
ax.plot(x, d0px1[:, -1, :].detach().cpu().view(x.shape) * 1.3)
ax.set_ylim(0, 1.3)
ax.set_xlim(0, 1)
plt.title("p(x,1+0)")
plt.savefig(pm.get_data_path("p(x,1+0)", ".png"))
plt.close()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
save(p, pm.get_data_path("p", ".ntdt"))
"""save
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
