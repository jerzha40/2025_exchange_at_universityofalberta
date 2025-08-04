import matplotlib.pyplot as plt
from torch import load, save, linspace, meshgrid, float64, stack, vmap
from torch import exp, nn, zeros_like, max, abs, Tensor, cuda
from PathManager import PathManager
from FCN import FCN
from torch.func import jacrev
from torch.optim import Adam

"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
# print(cuda.is_available())
# if cuda.is_available():
#     cuda.init()  # 或者来一句 dummy = empty(0, device='cuda')
#     cuda.set_device(0)  # 可选：显式指定 GPU
# """
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# """
pm = PathManager()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
Δ = 1.0
T = (0, Δ)
X = (0, 1)
NT = 100
NX = 200
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


def f(x):
    return x


def TrueSolution(x, t):
    return x * exp(t)


"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
δ: FCN = load(pm.get_data_path("δ", ".ntdt"), weights_only=False)
# δ: FCN = FCN(2, 1, 16, 4)
δ.to("cuda")
"""δ, δ(x,t)=Φ(x,t)-x
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
d1δ_dxt1 = vmap(vmap(jacrev(δ)))
d2δ_dxt2 = vmap(vmap(jacrev(jacrev(δ))))
d1f_dx1 = vmap(vmap(jacrev(f)))
d0f = vmap(vmap(f))
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
mx, mt = meshgrid(x, t, indexing="ij")  # shape = [20, 20, 10]
mx = mx.to("cuda")
mt = mt.to("cuda")
x = x.to("cuda")
print("ssdf", mx[:7, :7], mt[:7, :7], sep="\n")
print("kasg", mx.shape, mt.shape, sep="\n")
assert mx.shape == mt.shape
"""generate meshgrid (x,t)
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


def Graph(Z: Tensor, name: str):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        mx.detach().cpu().view(NX, NT),
        mt.detach().cpu().view(NX, NT),
        Z.detach().cpu().view(NX, NT),
        cmap="viridis",
    )
    plt.title(name)
    plt.savefig(pm.get_data_path(name, ".png"))


"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
xt = stack([mx, mt], dim=2)
xt = xt.to("cuda")
print("hgsl", xt[:7, :7, 0], xt[:7, :7, 1], sep="\n")
print("sghd", xt.shape, sep="\n")
δ_xt: Tensor = vmap(vmap(δ))(xt)
print("sagk", δ_xt[:7, :7, :], sep="\n")
print("gljg", δ_xt.shape, sep="\n")
Graph(δ_xt, "init_δ_xt")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
optimizer = Adam(δ.parameters(), lr=5e-5)
mse = nn.MSELoss()
for i in range(2000):
    optimizer.zero_grad()

    δ_xt: Tensor = δ(xt)
    δ_x: Tensor = δ_xt[:, 0, 0]
    d1δ_dxt1_xt: Tensor = d1δ_dxt1(xt)
    d2δ_dxt2_xt: Tensor = d2δ_dxt2(xt)
    # print("asdf", d1δ_dxt1_xt.shape, d2δ_dxt2_xt.shape, δ_xt.shape, sep="\n")
    d1δ_dt_xt = d1δ_dxt1_xt[:, :, 0, 1]
    d1δ_dx_xt = d1δ_dxt1_xt[:, :, 0, 0]
    d2δ_dtdx_xt = d2δ_dxt2_xt[:, :, 0, 0, 1]
    # print("sldk", d1δ_dt_xt.shape, d2δ_dtdx_xt.shape, d1δ_dx_xt.shape, sep="\n")
    f_δ_xt: Tensor = d0f(δ_xt)[:, :, 0]
    f_prime_δ_xt: Tensor = d1f_dx1(δ_xt)[:, :, 0, 0]
    # print("sgjk", d1δ_dt_xt.shape, f_δ_xt.shape, sep="\n")
    assert d1δ_dt_xt.shape == f_δ_xt.shape
    # print("slga", x.shape, δ_x.shape, sep="\n")
    # print("sgdj", f_prime_δ_xt.shape, d1δ_dx_xt.shape, d2δ_dtdx_xt.shape, sep="\n")

    loss = (
        mse(d1δ_dt_xt, f_δ_xt)
        + mse(x, δ_x)
        + mse(d2δ_dtdx_xt, f_prime_δ_xt * d1δ_dx_xt)
    )
    loss.backward()

    optimizer.step()
    print(f"{i:5d} {loss.tolist():5.2e}")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
δ_xt: Tensor = vmap(vmap(δ))(xt)[:, :, 0]
Graph(δ_xt, "after_δ_xt")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
TrueSolution_xt: Tensor = vmap(vmap(TrueSolution))(mx, mt)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(
    mx.detach().cpu().view(NX, NT),
    mt.detach().cpu().view(NX, NT),
    δ_xt.detach().cpu().view(NX, NT),
    cmap="viridis",
)
ax.plot_surface(
    mx.detach().cpu().view(NX, NT),
    mt.detach().cpu().view(NX, NT),
    TrueSolution_xt.detach().cpu().view(NX, NT),
    cmap="Wistia_r",
)
plt.title("compare")
plt.savefig(pm.get_data_path("compare", ".png"))
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
print("ajsk", δ_xt.shape, TrueSolution_xt.shape)
Graph(δ_xt - TrueSolution_xt, "difference")
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
save(δ, pm.get_data_path("δ", ".ntdt"))
"""save
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
