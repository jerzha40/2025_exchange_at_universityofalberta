"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
$祷言HaarAugmentedNetwork.py:
美好的祝福
素晴らしいの祝福を
"""

import torch
from torch import Tensor, nn
from torch import rand, zeros, zeros_like
from torch.nn.functional import normalize

"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


class HAN(nn.Module):
    "define a HaarAugmentedNetwork"

    def __init__(self, N: int):
        super(HAN, self).__init__()
        self.N: int = N
        self.P: Tensor = nn.Parameter(rand([N, 2]), requires_grad=True)

    def forward(self, f: Tensor):
        Φ = normalize(self.P)
        Ψ = zeros_like(Φ)
        Ψ[:, 0] = Φ[:, 1]
        Ψ[:, 1] = -Φ[:, 0]
        assert len(f) == 2**self.N
        f_hat: Tensor = zeros(2**self.N)
        for i in range(self.N):
            f = f.reshape(-1, 2)
            # print(f"slkh\nf={f}")
            f_hat[2 ** (self.N - i - 1) : 2 ** (self.N - i)] = f @ Ψ[i, :]
            f = f @ Φ[i, :]
            # print(f"ghld\nf={f}")
            # print(f"f_hat={f_hat}")
        f_hat[0] = f[0]
        return f_hat


"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
