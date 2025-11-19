"""
Neural Discrete Representation Learning (https://arxiv.org/pdf/1711.00937)

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
"""

import tqdm
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader


class VectorQuantizer(nn.Module):
    # def __init__(self, codebook_size=1024, latent_dim=2):
    def __init__(self, codebook_size: int = 1024, codebook_dim: int = 2):
        super().__init__()

        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        # self.embedding = nn.Embedding(codebook_size, latent_dim)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        ### Distance btwn every Latent and Code: (L-C)**2 = (L**2 - 2LC + C**2 ) ###

        # L2: [B, L] -> [B, 1]
        # l2 = torch.sum(x**2, dim=1, keepdim=True)

        # # C2: [C, L] -> [C]
        # c2 = torch.sum(self.embedding.weight**2, dim=1).unsqueeze(0)

        # # CL: [B,L]@[L,C] -> [B, C]
        # cl = x@self.embedding.weight.t()

        # # [B, 1] - 2 * [B, C] + [C] -> [B, C]
        # distances = l2 - 2*cl + c2

        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ self.embedding.weight.t()
            + self.embedding.weight.pow(2).sum(1, keepdim=True).t()
        )

        closest = torch.argmin(dist, dim=-1)  # [B, 1]

        quantized_latents_idx = torch.zeros(
            batch_size, self.codebook_size, device=x.device
        )  # [B, C]

        quantized_latents_idx[torch.arange(batch_size), closest] = 1

        # [B, C] @ [C, L] -> [B, L]
        quantized_latents = quantized_latents_idx @ self.embedding.weight

        return quantized_latents


class LinearVectorQuantizedVAE(nn.Module):
    def __init__(self, latent_dim=2, codebook_size=512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        #########################################################
        ###  The New Layers Added in from Original VAE Model  ###
        self.vq = VectorQuantizer(codebook_size, latent_dim)

        #########################################################

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 32 * 32),
            nn.Sigmoid(),
        )

    def forward_enc(self, x):
        x = self.encoder(x)

        return x

    def quantize(self, z):
        #############################################
        ## Quantize the Latent Space Representation #

        codes = self.vq(z)

        ### Compute VQ Loss ###
        codebook_loss = torch.mean((codes - z.detach()) ** 2)
        commitment_loss = torch.mean((codes.detach() - z) ** 2)

        ### Straight Through ###
        codes = z + (codes - z).detach()

        #############################################

        return codes, codebook_loss, commitment_loss

    def forward_dec(self, x):
        codes, codebook_loss, commitment_loss = self.quantize(x)
        decoded = self.decoder(codes)

        return codes, decoded, codebook_loss, commitment_loss

    def forward(self, x):
        batch, channels, height, width = x.shape

        ### Flatten Image to Vector ###
        x = x.flatten(1)

        ### Pass Through Encoder ###
        latents = self.forward_enc(x)

        ### Pass Sampled Data Through Decoder ###
        quantized_latents, decoded, codebook_loss, commitment_loss = self.forward_dec(
            latents
        )

        ### Put Decoded Image Back to Original Shape ###
        decoded = decoded.reshape(batch, channels, height, width)

        return latents, quantized_latents, decoded, codebook_loss, commitment_loss


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
