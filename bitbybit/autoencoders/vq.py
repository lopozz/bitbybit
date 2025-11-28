import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int = 1024, codebook_dim: int = 2):
        super().__init__()

        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, x):

        # euclidean distance (x - e)**2 = x**2 -2xe + e**2
        dist = (
            x.pow(2).sum(1, keepdim=True)                           # [B, 1]
            - 2 * x @ self.codebook.weight.T                        # [B, C] @ [C, Cs] -> [B, Cs]
            + self.codebook.weight.pow(2).sum(1, keepdim=True).T    # [1, Cs]
        )
        ids = torch.argmin(dist, dim=-1)                            # [B]
        z_q = self.codebook(ids)                                    # [B, C]
        z_q = x + (z_q - x).detach()  # noop in forward pass, straight-through gradient estimator in backward pass

        return z_q, ids