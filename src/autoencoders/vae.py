"""
Auto-Encoding Variational Bayes
Kingma, Diederik P. and Max Welling. - 2013

How can we perform efficient inference and learning in directed probabilistic
models, in the presence of continuous latent variables with intractable posterior
distributions, and large datasets? We introduce a stochastic variational inference
and learning algorithm that scales to large datasets and, under some mild 
differentiability conditions, even works in the intractable case. Our contributions 
are two-fold. First, we show that a reparameterization of the variational lower bound
yields a lower bound estimator that can be straightforwardly optimized using standard 
stochastic gradient methods. Second, we show that for i.i.d. datasets with
continuous latent variables per datapoint, posterior inference can be made especially 
efficient by fitting an approximate inference model (also called a recognition model) 
to the intractable posterior using the proposed lower bound estimator. Theoretical 
advantages are reflected in experimental results.

https://arxiv.org/pdf/1312.6114
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

def VAELoss(x, y, mean, log_var, recon_weight=1, kl_weight=1):

    # compute the MSE For Every Pixel [B, H*W]
    pixel_mse = ((x-y)**2)

    # sum  up pixel loss per image and average across batch
    recon_loss = pixel_mse.sum(axis=-1).mean()

    # compute KL per image and sum across flattened latent
    kl = (1 + log_var - mean**2 - torch.exp(log_var)) #.flatten(1)
    kl_per_image = - 0.5 * torch.sum(kl, dim=-1)

    # average KL across the batch ###
    kl_loss = torch.mean(kl_per_image)
    
    return recon_weight*recon_loss, kl_weight*kl_loss


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


class LinearVAE(nn.Module):
    """
    Example dims: [784, 1000, 500, 250, 30]
    """

    def __init__(self, dims: List[int]):
        super().__init__()
        self.enc = LinearEncoder(dims[:-1])

        self.fn_mu =  nn.Linear(dims[-2], dims[-1])
        self.fn_logvar = nn.Linear(dims[-2], dims[-1])

        dims = list(reversed(dims))  # mirror
        self.dec = LinearDecoder(dims)

        

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        x = self.encode(x)

        # sample with reparamaterization trick
        mu = self.fn_mu(x)
        logvar = self.fn_logvar(x)
        sigma = torch.exp(0.5*logvar)
        noise = torch.randn_like(sigma, device=sigma.device)
        z = mu + sigma*noise

        out = self.decode(z)
        return out, z, mu, logvar