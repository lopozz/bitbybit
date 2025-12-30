"""
Neural Discrete Representation Learning
Van Den Oord, Aaron, and Oriol Vinyals - 2017

Learning useful representations without supervision remains a key challenge in
machine learning. In this paper, we propose a simple yet powerful generative
model that learns such discrete representations. Our model, the Vector Quantised
Variational AutoEncoder (VQ-VAE), differs from VAEs in two key ways: the
encoder network outputs discrete, rather than continuous, codes; and the prior
is learnt rather than static. In order to learn a discrete latent representation, we
incorporate ideas from vector quantisation (VQ). Using the VQ method allows the
model to circumvent issues of “posterior collapse” -— where the latents are ignored
when they are paired with a powerful autoregressive decoder -— typically observed
in the VAE framework. Pairing these representations with an autoregressive prior,
the model can generate high quality images, videos, and speech as well as doing
high quality speaker conversion and unsupervised learning of phonemes, providing
further evidence of the utility of the learnt representations.

https://arxiv.org/pdf/1711.00937
"""

import torch
import torch.nn as nn

from typing import List

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int = 1024, codebook_dim: int = 2):
        super().__init__()

        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, x):
        # euclidean distance (x - e)**2 = x**2 -2xe + e**2
        dist = (
            x.pow(2).sum(1, keepdim=True)  # [B, 1]
            - 2 * x @ self.codebook.weight.T  # [B, C] @ [C, Cs] -> [B, Cs]
            + self.codebook.weight.pow(2).sum(1, keepdim=True).T  # [1, Cs]
        )
        ids = torch.argmin(dist, dim=-1)  # [B]
        e_k = self.codebook(ids)  # [B, C]

        # straight-through estimator (see https://arxiv.org/pdf/1308.3432)
        e_k = x + (e_k - x).detach()

        return e_k, ids

class LinearEncoder(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        for in_d, out_d in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_d, out_d))
            self.bns.append(nn.BatchNorm1d(out_d))

    def forward(self, x):
        for i, (layer, bn) in enumerate(zip(self.layers, self.bns)):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = bn(x)
                x = torch.relu(x)
        return x


class LinearDecoder(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        for in_d, out_d in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_d, out_d))
            self.bns.append(nn.BatchNorm1d(out_d))

    def forward(self, z):
        x = z
        for i, (layer, bn) in enumerate(zip(self.layers, self.bns)):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = bn(x)
                x = torch.relu(x)
            else:
                x = torch.sigmoid(x)
        return x


class LinearVQVAE(nn.Module):
    """
    Example dims: [784, 1000, 500, 250, 30]
    """

    def __init__(self, dims: List[int], codebook_size: int):
        super().__init__()
        self.enc = LinearEncoder(dims)

        self.vq = VectorQuantizer(codebook_size, dims[-1])

        dims = list(reversed(dims))  # mirror
        self.dec = LinearDecoder(dims)

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        z = self.encode(x)
        e_k, ids = self.vq(z)
        out = self.decode(e_k)
        return out, z, e_k, ids


class Encoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=256, n_layers=6, verbose=False):
        super().__init__()
        self.verbose = verbose
        convs = []
        for i in range(n_layers):
            convs.append(
                nn.Conv1d(
                    in_channels if i == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        # x: [B, C, T]
        if self.verbose:
            print(f"Input: {x.shape}")
            print("Encoder")
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = torch.relu(x)

            if self.verbose:
                print(f"After layer {i + 1}: {x.shape}")

        return x


class Decoder(nn.Module):
    def __init__(self, out_channels=1, hidden_dim=256, n_layers=6, verbose=False):
        super().__init__()
        self.verbose = verbose
        deconvs = []
        for i in range(n_layers):
            deconvs.append(
                nn.ConvTranspose1d(
                    hidden_dim,
                    out_channels if i == n_layers - 1 else hidden_dim,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=0,
                )
            )
        self.deconvs = nn.ModuleList(deconvs)

    def forward(self, x):
        if self.verbose:
            print("Decoder")
        for i, deconv in enumerate(self.deconvs):
            x = deconv(x)
            if i < len(self.deconvs) - 1:
                x = torch.relu(x)
                if self.verbose:
                    print(f"After layer {i + 1}: {x.shape}")
        if self.verbose:
            print(f"Output: {x.shape}")
        return x


class VQVAE(nn.Module):
    def __init__(self, hidden_dim=2, n_layers=6, codebook_size=512, verbose=False):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.enc = Encoder(
            in_channels=1, hidden_dim=hidden_dim, n_layers=n_layers, verbose=verbose
        )
        self.vq = VectorQuantizer(codebook_size=codebook_size, codebook_dim=hidden_dim)
        self.dec = Decoder(
            out_channels=1, hidden_dim=hidden_dim, n_layers=n_layers, verbose=verbose
        )

    def forward(self, x):
        batch_size, _, _ = x.shape  # [B, D, T]

        z_e = self.enc(x)  # [B, D, T_e]
        z_e = z_e.permute(0, 2, 1).reshape(
            -1, self.hidden_dim
        )  # [B, T_e, D] -> [B*T_e, D]
        e_k, ids = self.vq(z_e)
        e_k = e_k.view(batch_size, -1, self.hidden_dim).permute(0, 2, 1)  # [B, D, T_e]
        y = self.dec(e_k)  # [B, D, T]

        return y, z_e, ids
