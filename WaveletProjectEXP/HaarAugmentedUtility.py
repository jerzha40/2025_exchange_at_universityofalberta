from torch import zeros, kron
from torch import Tensor

"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


def HaarDecomp(N: int, f: Tensor, ψ: Tensor, φ: Tensor):
    assert len(f) == 2**N
    f_hat: Tensor = zeros(2**N)
    for i in range(N):
        f = f.reshape(-1, 2)
        print(f"slkh\nf={f}")
        f_hat[2 ** (N - i - 1) : 2 ** (N - i)] = f @ ψ
        f = f @ φ
        print(f"ghld\nf={f}")
        print(f"f_hat={f_hat}")
    f_hat[0] = f[0]
    return f_hat


def HaarRecons(N: int, f_hat: Tensor, ψ: Tensor, φ: Tensor):
    f = f_hat[0]
    for i in range(N):
        f = kron(f, φ)
        # print(f"slkd\nf={f}")
        f += kron(f_hat[2 ** (0 + i) : 2 ** (1 + i)], ψ)
        # print(f"hsgl\nf={f}")
    return f


"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


def HaarDecompAug(N: int, f: Tensor, Ψ: Tensor, Φ: Tensor):
    assert len(f) == 2**N
    f_hat: Tensor = zeros(2**N)
    for i in range(N):
        f = f.reshape(-1, 2)
        print(f"slkh\nf={f}")
        f_hat[2 ** (N - i - 1) : 2 ** (N - i)] = f @ Ψ[i, :]
        f = f @ Φ[i, :]
        print(f"ghld\nf={f}")
        print(f"f_hat={f_hat}")
    f_hat[0] = f[0]
    return f_hat


def HaarReconsAug(N: int, f_hat: Tensor, Ψ: Tensor, Φ: Tensor):
    f = f_hat[0]
    for i in range(N):
        f = kron(f, Φ[N - i - 1, :])
        # print(f"slkd\nf={f}")
        f += kron(f_hat[2 ** (0 + i) : 2 ** (1 + i)], Ψ[N - i - 1, :])
        # print(f"hsgl\nf={f}")
    return f


"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
