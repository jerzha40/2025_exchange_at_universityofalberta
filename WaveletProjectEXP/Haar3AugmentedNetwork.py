"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
$祷言HaarAugmentedNetwork.py:
美好的祝福
素晴らしいの祝福を
"""

import torch
from torch import Tensor, nn, linalg
from torch import rand, zeros, zeros_like
from torch.nn.functional import normalize

"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""


class H3AN(nn.Module):
    "define a Haar3AugmentedNetwork"

    def __init__(self, N: int):
        super(H3AN, self).__init__()
        self.N: int = N
        self.Φ_P: Tensor = nn.Parameter(rand([N, 3, 3]), requires_grad=True)

    def forward(self, f: Tensor):
        Q_row, _ = linalg.qr(self.Φ_P.transpose(-2, -1), mode="reduced")
        Φ: Tensor = Q_row.transpose(-2, -1)
        assert len(f) == 3**self.N
        f_hat: Tensor = zeros(3**self.N)
        for i in range(self.N):
            f = f.reshape(-1, 3)
            # print(f"slkh\nf={f}")
            for j in range(1, 3):
                f_hat[3 ** (self.N - i - 1) * j : 3 ** (self.N - i - 1) * (j + 1)] = (
                    f @ Φ[i, j, :]
                )
            f = f @ Φ[i, 0, :]
            # print(f"ghld\nf={f}")
            # print(f"f_hat={f_hat}")
        f_hat[0] = f[0]
        return f_hat


"""
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
"""
