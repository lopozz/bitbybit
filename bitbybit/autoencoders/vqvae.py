"""
Neural Discrete Representation Learning

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

import tqdm
import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from bitbybit.autoencoders.vq import VectorQuantizer


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=256):
        super().__init__()

        # 2x strided conv layers: 4x4, stride 2, 256 channels
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)

        # Residual blocks: ReLU → 3x3 conv → ReLU → 1x1 conv
        self.res1_conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.res1_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1)
        self.res2_conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.res2_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1)

    def forward(self, x):
        # conv block
        x = self.conv1(x)
        x = self.conv2(x)

        # residual block 1
        out = F.relu(x)
        out = self.res1_conv3(out)
        out = F.relu(out)
        out = self.res1_conv1(out)
        x = x + out

        # residual block 2
        out = F.relu(x)
        out = self.res2_conv3(out)
        out = F.relu(out)
        out = self.res2_conv1(out)
        x = x + out

        return x

class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_dim=256):
        super().__init__()

        # Residual blocks: mirror of the encoder’s ResNet-style blocks
        self.res1_conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.res1_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1)

        self.res2_conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.res2_conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, stride=1)

        # 2x upsampling with transposed convs: 4x4, stride 2
        self.deconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # residual block 1
        out = F.relu(x)
        out = self.res1_conv3(out)
        out = F.relu(out)
        out = self.res1_conv1(out)
        x = x + out

        # residual block 2
        out = F.relu(x)
        out = self.res2_conv3(out)
        out = F.relu(out)
        out = self.res2_conv1(out)
        x = x + out

        # upsampling back to image resolution
        x = F.relu(x)
        x = self.deconv1(x)
        x = self.deconv2(x)


        return x

class VQVAE(nn.Module):
    def __init__(self, hidden_dim=2, codebook_size=512):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.enc = Encoder(in_channels=1, hidden_dim=hidden_dim)
        self.vq = VectorQuantizer(codebook_size=codebook_size, codebook_dim=hidden_dim)
        self.dec = Decoder(out_channels=1, hidden_dim=hidden_dim)

    def forward(self, x):
        batch_size, C, _, _ = x.shape
        z_e = self.enc(x)
        _, _, H_e, W_e = z_e.shape
        z_e = z_e.permute(0, 2, 3, 1)      # [B, H, W, D]
        z_e = z_e.view(-1, self.hidden_dim)
        z_q, ids = self.vq(z_e)
        z_q = z_q.view(batch_size, H_e, W_e, self.hidden_dim)
        z_q = z_q.permute(0, 3, 1, 2)
        y = self.dec(z_q); print(y.shape)

        return y, z_e, z_q, ids
    
    def forward(self, x):
        batch_size, _, _, _ = x.shape

        z_e = self.enc(x)                        # [B, D, H_e, W_e]
        batch_size, self.hidden_dim, H_e, W_e = z_e.shape

        # flatten for VQ
        z_flat = z_e.permute(0, 2, 3, 1).view(-1, self.hidden_dim)  # [B*H_e*W_e, D]
        z_q_flat, ids = self.vq(z_flat)                            # [B*H_e*W_e, D]

        # back to feature map
        z_q = z_q_flat.view(batch_size, H_e, W_e, self.hidden_dim).permute(0, 3, 1, 2)  # [B, D, H_e, W_e]

        y = self.dec(z_q)                                         # [B, 1, H, W]

        return y, z_e, z_q, ids


def VAELoss(x, x_hat, mean, log_var, kl_weight=1, reconstruction_weight=1):
    ### Compute the MSE For Every Pixel [B, C, H, W] ###
    pixel_mse = (x - x_hat) ** 2

    ### Flatten Each Image in Batch to Vector [B, C*H*W] ###
    pixel_mse = pixel_mse.flatten(1)

    ### Sum  Up Pixel Loss Per Image and Average Across Batch ###
    reconstruction_loss = pixel_mse.sum(axis=-1).mean()

    ### Compute KL Per Image and Sum Across Flattened Latent###
    kl = (1 + log_var - mean**2 - torch.exp(log_var)).flatten(1)
    kl_per_image = -0.5 * torch.sum(kl, dim=-1)

    ### Average KL Across the Batch ###
    kl_loss = torch.mean(kl_per_image)

    return reconstruction_weight * reconstruction_loss + kl_weight * kl_loss


def train(
    model,
    kl_weight,
    train_set,
    test_set,
    batch_size,
    training_iterations,
    evaluation_iterations,
    model_type="VQVAE",
):
    data_variance = torch.var(train_set.data / 255.0)

    if model_type != "VAE":
        kl_weight = None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    trainloader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=8
    )
    testloader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=8
    )

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    train_loss = []
    evaluation_loss = []

    encoded_data_per_eval = []
    quantized_encoded_data_per_eval = [] if model_type == "VQVAE" else None
    train_losses = []
    evaluation_losses = []

    pbar = tqdm(range(training_iterations))

    train = True

    step_counter = 0
    while train:
        for images, labels in trainloader:
            images = images.to(device)

            if model_type == "VQVAE":
                encoded, quantized_encoded, decoded, codebook_loss, commitment_loss = (
                    model(images)
                )
                reconstruction_loss = (
                    torch.mean((images - decoded) ** 2) / data_variance
                )
                loss = reconstruction_loss + codebook_loss + 0.25 * commitment_loss
            elif model_type == "VAE":
                encoded, decoded, mu, logvar = model(images)
                loss = VAELoss(images, decoded, mu, logvar, kl_weight)
            elif model_type == "AE":
                encoded, decoded = model(images)
                loss = torch.mean((images - decoded) ** 2)

            train_loss.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step_counter % evaluation_iterations == 0:
                model.eval()

                encoded_evaluations = []
                if model_type == "VQVAE":
                    quantized_encoded_evaluations = []

                for images, labels in testloader:
                    images = images.to(device)

                    if model_type == "VQVAE":
                        (
                            encoded,
                            quantized_encoded,
                            decoded,
                            codebook_loss,
                            commitment_loss,
                        ) = model(images)
                        reconstruction_loss = (
                            torch.mean((images - decoded) ** 2) / data_variance
                        )
                        loss = (
                            reconstruction_loss + codebook_loss + 0.25 * commitment_loss
                        )
                    elif model_type == "VAE":
                        encoded, decoded, mu, logvar = model(images)
                        loss = VAELoss(images, decoded, mu, logvar, kl_weight)
                    elif model_type == "AE":
                        encoded, decoded = model(images)
                        loss = torch.mean((images - decoded) ** 2)

                    evaluation_loss.append(loss.item())

                    encoded, labels = encoded.cpu(), labels.reshape(-1, 1)

                    encoded_evaluations.append(
                        torch.cat((encoded.flatten(1), labels), axis=-1)
                    )

                    if model_type == "VQVAE":
                        quantized_encoded = quantized_encoded.cpu()
                        quantized_encoded_evaluations.append(
                            torch.cat((quantized_encoded.flatten(1), labels), axis=-1)
                        )

                encoded_data_per_eval.append(
                    torch.concatenate(encoded_evaluations).detach()
                )
                if model_type == "VQVAE":
                    quantized_encoded_data_per_eval.append(
                        torch.concatenate(quantized_encoded_evaluations).detach()
                    )

                train_loss = np.mean(train_loss)
                evaluation_loss = np.mean(evaluation_loss)

                train_losses.append(train_loss)
                evaluation_losses.append(evaluation_loss)

                train_loss = []
                evaluation_loss = []

                model.train()

            step_counter += 1
            pbar.update(1)

            if step_counter >= training_iterations:
                print("Completed Training!")
                train = False
                break

    encoded_data_per_eval = [np.array(i) for i in encoded_data_per_eval]

    if model_type == "VQVAE":
        quantized_encoded_data_per_eval = [
            np.array(i) for i in quantized_encoded_data_per_eval
        ]

    print("Final Training Loss", train_losses[-1])
    print("Final Evaluation Loss", evaluation_losses[-1])

    return (
        model,
        train_losses,
        evaluation_losses,
        encoded_data_per_eval,
        quantized_encoded_data_per_eval,
    )
