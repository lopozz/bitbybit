"""
Reducing the Dimensionality of Data with Neural Networks
G. E. Hinton* and R. R. Salakhutdinov - 2006

High-dimensional data can be converted to low-dimensional codes by training a multilayer neural
network with a small central layer to reconstruct high-dimensional input vectors. Gradient descent
can be used for fine-tuning the weights in such ‘‘autoencoder’’ networks, but this works well only if
the initial weights are close to a good solution. We describe an effective way of initializing the
weights that allows deep autoencoder networks to learn low-dimensional codes that work much
better than principal components analysis as a tool to reduce the dimensionality of data.

https://www.cs.toronto.edu/~hinton/absps/science.pdf
"""

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import List, Callable





class LinearEncoder(nn.Module):
    """
    Example dims: [784, 1000, 500, 250, 30]
    Produces z of size 30.
    """
    def __init__(self, dims: List[int], act: Callable[[Tensor], Tensor]):
        super().__init__()
        assert len(dims) >= 2, "dims must be like [in_dim, ..., latent_dim]"
        self.dims = dims
        self.act = act

        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
        self.layers = nn.ModuleList(layers)


    def forward(self, x):
        # x: [B, in_dim]

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # no nonlinearity on code by default
                x = self.act(x)
        return x  # z


class LinearDecoder(nn.Module):
    """
    Example dims: [30, 250, 500, 1000, 784]
    Reconstructs to size 784.
    """
    def __init__(self, dims: List[int], act: Callable[[Tensor], Tensor]):
        super().__init__()
        assert len(dims) >= 2, "dims must be like [latent_dim, ..., out_dim]"
        self.dims = dims
        self.act = act

        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
        self.layers = nn.ModuleList(layers)


    def forward(self, z):

        x = z
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.act(x, self.activation)
            else:
                x = F.relu(x) # by default

        return x


class LinearAE(nn.Module):
    """
    Provide encoder dims only; decoder is created as the mirror.
    Example:
      ae = AE(enc_dims=[784, 1000, 500, 250, 30], out_activation="sigmoid")
    """
    def __init__(self, dims: List[int], act: Callable[[Tensor], Tensor]):
        super().__init__()
        self.enc = LinearEncoder(dims, act=act)
        dims = list(reversed(dims))  # mirror
        self.dec = LinearDecoder(dims, act=act)

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out, z


class ConvEncoder(nn.Module):
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
            x = F.relu(x)

            if self.verbose:
                print(f"After layer {i + 1}: {x.shape}")

        return x


class ConvDecoder(nn.Module):
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
                x = F.relu(x)
                if self.verbose:
                    print(f"After layer {i + 1}: {x.shape}")
        if self.verbose:
            print(f"Output: {x.shape}")
        return x


class ConvAE(nn.Module):
    def __init__(self, hidden_dim=512, n_layers=6, verbose=False):
        super().__init__()
        self.enc = ConvEncoder(
            in_channels=1, hidden_dim=hidden_dim, n_layers=n_layers, verbose=verbose
        )
        self.dec = ConvDecoder(out_channels=1, hidden_dim=hidden_dim, n_layers=n_layers)

    def forward(self, x):
        z = self.enc(x)  # [B, C, T_e]
        y = self.dec(z)  # [B, C, T]
        return y