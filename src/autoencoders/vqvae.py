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
    

class LinearVectorQuantizer(nn.Module):
    """
    For vector latents: x is [B, D]
    Returns:
      e_k: [B, D] (quantized, straight-through)
      ids: [B] (code index per item)
    """
    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, x: torch.Tensor):

        # distances: [B, K]
        dist = (
            x.pow(2).sum(dim=1, keepdim=True)        # [B, 1]
            - 2.0 * (x @ self.codebook.weight.t())                      # [B, K]
            + self.codebook.weight.pow(2).sum(dim=1, keepdim=True).t()  # [1, K]
        )

        ids = dist.argmin(dim=1)   # [B]
        e_k = self.codebook(ids)   # [B, D]

        # straight-through estimator
        e_k_st = x + (e_k - x).detach()

        return e_k_st, ids

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

        self.vq = LinearVectorQuantizer(codebook_size, dims[-1])

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



class ConvVectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook = nn.Embedding(codebook_size, codebook_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z_e):  # [B, C, H, W]
        B, C, H, W = z_e.shape
        z_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [N, C]

        w = self.codebook.weight  # [K, C]
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)     # [N, 1]
            - 2 * z_flat @ w.t()                   # [N, K]
            + w.pow(2).sum(1).unsqueeze(0)         # [1, K]
        )
        ids = torch.argmin(dist, dim=1)            # [N]
        e_k = self.codebook(ids)                   # [N, C] (grads -> codebook)
        e_k_st = z_flat + (e_k - z_flat).detach()  # straight-through (grads -> encoder)

        e_k = e_k.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        e_k_st = e_k_st.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        ids = ids.view(B, H, W)

        return e_k, e_k_st, ids
    


class ConvEncoder(nn.Module):
    def __init__(self, dims: List[int], n_res: int = 2):
        super().__init__()
        assert len(dims) >= 2, "dims must be like [in_dim, ..., latent_dim]"
        self.dims = dims

        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(
                nn.Conv2d(
                    in_channels=in_d,
                    out_channels=out_d,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # x: [B, C, H, W]

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # no nonlinearity on code by default
                x = torch.relu(x)

        return x  # z


class ConvDecoder(nn.Module):
    def __init__(self, dims: List[int], n_res: int = 2):
        super().__init__()
        assert len(dims) >= 2

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        for in_d, out_d in zip(dims[:-1], dims[1:]):
            self.layers.append(
                nn.ConvTranspose2d(
                    in_d,
                    out_d,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                )
            )

    def forward(self, x):
        for i, conv in enumerate(self.layers):
            x = conv(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
            else:
                x = torch.sigmoid(x)
        return x


class ConvVQVAE(nn.Module):
    """
    Example dims: [1, 8, 16, 4]
    """

    def __init__(self, dims: List[int], codebook_size: int):
        super().__init__()
        self.enc = ConvEncoder(dims)

        self.vq = ConvVectorQuantizer(codebook_size, dims[-1])

        dims = list(reversed(dims))  # mirror
        self.dec = ConvDecoder(dims)

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        z = self.encode(x)
        e_k, e_k_st, ids = self.vq(z)
        out = self.decode(e_k_st)
        return out, z, e_k, ids




class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        h = torch.relu(x)
        h = self.conv3(h)
        h = torch.relu(h)
        h = self.conv1(h)
        return x + h


class Encoder(nn.Module):
    def __init__(self, in_ch=3, hidden=256, n_res=2, z_ch=3):
        super().__init__()

        self.c1 = nn.Conv2d(in_ch, hidden, kernel_size=4, stride=2, padding=1)
        self.c2 = nn.Conv2d(hidden, hidden, kernel_size=4, stride=2, padding=1)
        self.res = nn.Sequential(*[ResBlock(hidden) for _ in range(n_res)])
        self.to_z = nn.Conv2d(hidden, z_ch, kernel_size=1)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = self.res(x)
        z_e = self.to_z(x)           # [B, 1, 32, 32]
        return z_e



class Decoder(nn.Module):
    def __init__(self, out_ch=3, hidden=256, z_ch=1):
        super().__init__()
        # lift 1 channel latent to hidden feature maps
        self.from_z = nn.Conv2d(z_ch, hidden, kernel_size=1)

        self.t1 = nn.ConvTranspose2d(hidden, hidden, kernel_size=4, stride=2, padding=1)
        self.t2 = nn.ConvTranspose2d(hidden, out_ch, kernel_size=4, stride=2, padding=1)

    def forward(self, z_q):
        x = torch.relu(self.from_z(z_q))
        x = torch.relu(self.t1(x))
        x = torch.sigmoid(self.t2(x))  # assume inputs in [0,1]
        return x


class VQVAE(nn.Module):

    def __init__(self, codebook_size=512, hidden=256, n_res=2, z_ch=1):
        super().__init__()
        self.enc = Encoder(in_ch=3, hidden=hidden, n_res=n_res, z_ch=z_ch)
        self.vq = ConvVectorQuantizer(codebook_size=codebook_size, codebook_dim=z_ch)
        self.dec = Decoder(out_ch=3, hidden=hidden, z_ch=z_ch)

    def forward(self, x):
        z_e = self.enc(x)                 # [B, 1, 32, 32]
        e_k, e_k_st, ids = self.vq(z_e)   # both are [B, 1, 32, 32]
        out = self.dec(e_k_st)            # [B, 3, 128, 128]
        return out, z_e, e_k, ids
