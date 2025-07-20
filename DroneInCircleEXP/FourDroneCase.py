import matplotlib.pyplot as plt
from numpy import random, zeros, array, linalg
from PathManager import PathManager

"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
pm = PathManager()
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
NUM = 4
DIM = 2
d = random.uniform(0, 1, size=(NUM, DIM))
v = zeros(shape=(NUM, DIM))
a = zeros(shape=(NUM, DIM))  # random.uniform(0, 1, size=(4, 2))
R = 1.0
dt = 0.001
N = 1000
t = array([1, 2, 3, 0])
μ = 0.01
"""4 initial d in [0,1] * [0,1]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
print("Initial positions (d):\n", d)
print("Velocities (v):\n", v)
print("Accelerations (a):\n", a)
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


def draw_configuration(d, v, a, name):
    plt.figure(figsize=(6, 6))
    plt.xlim(-1, 2)
    plt.ylim(-1, 2)
    plt.gca().set_aspect("equal")
    """
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    """
    plt.scatter(d[:, 0], d[:, 1], color="blue", label="Position")
    for i in range(len(d)):
        plt.text(d[i, 0] + 0.02, d[i, 1] + 0.02, str(i), fontsize=10, color="black")
    """
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    """
    for i in range(4):
        plt.arrow(
            d[i, 0],
            d[i, 1],
            v[i, 0] * 0.1,
            v[i, 1] * 0.1,
            head_width=0.01,
            color="green",
            length_includes_head=True,
        )
    """velocity
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    """
    for i in range(4):
        plt.arrow(
            d[i, 0],
            d[i, 1],
            a[i, 0] * 0.5,
            a[i, 1] * 0.5,
            head_width=0.01,
            color="red",
            length_includes_head=True,
        )
    """acceleration
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    """
    plt.legend(["Position", "Velocity", "Acceleration"])
    plt.title("Initial Position, Velocity (green), Acceleration (red)")
    # plt.grid(True)
    plt.savefig(name)
    plt.close()
    # plt.show()
    """
    ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    """


"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
draw_configuration(d, v, a, pm.get_data_path("dd", ".png"))
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
d_n = d
v_n = v
a_n = a
d_n1 = None
v_n1 = None
# for L in range(N):
L = 0
while True:
    for i in range(NUM):
        a_n[i] = zeros(shape=(2))
        for k in range(NUM):
            if k == i:
                continue
            if linalg.norm(d_n[i] - d_n[k]) <= R:
                a_n[i] += μ * (d_n[i] - d_n[k]) / (linalg.norm(d_n[i] - d_n[k]) ** 3)
        a_n[i] += ((d_n[t[i]] - d_n[i]) / linalg.norm(d_n[t[i]] - d_n[i])) - v_n[i]
    d_n1 = d_n + v_n * dt
    v_n1 = v_n + a_n * dt
    d_n = d_n1
    v_n = v_n1
    # print(d_n)
    # print(v_n)
    # print(a_n)
    draw_configuration(d_n, v_n, a_n, pm.get_data_path(f"dd_{L}", ".png"))
    L += 1
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
