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
from torch.utils.data import DataLoader


class VectorQuantizer(nn.Module):
    # def __init__(self, codebook_size=1024, latent_dim=2):
    def __init__(self, codebook_size: int = 1024, codebook_dim: int = 2):
        super().__init__()

        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size

        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        # Compute euclidean distance with codebook: (x - e)**2
        dist = (
            x.pow(2).sum(1, keepdim=True)                           # [B, 1]
            - 2 * x @ self.codebook.weight.T                        # [B, C] @ [C, Cs] -> [B, Cs]
            + self.codebook.weight.pow(2).sum(1, keepdim=True).T    # [1, Cs]
        )

        # TODO: is this optimized/clean?
        # take inspiration from this https://github.com/hubertsiuzdak/snac/blob/main/snac/vq.py
        ids = torch.argmin(dist, dim=-1)  # [B, 1]

        # Create empty quantized latents embedding
        ids_mx = torch.zeros(
            batch_size, self.codebook_size, device=x.device
        )  # [B, Cs]

        # Place a 1 at the indexes for each sample for the codebook we want
        ids_mx[torch.arange(batch_size), ids] = 1

        z_q = ids_mx @ self.codebook.weight # [B, Cs] @ [Cs, C] -> [B, C]

        return z_q, ids


class LinearVectorQuantizedVAE(nn.Module):
    def __init__(self, latent_dim=2, codebook_size=512):
        super().__init__()

        # TODO: check in the paper what is the architectire for images
        # Ideally the implementtaion should work for both
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        self.vq = VectorQuantizer(codebook_size, latent_dim)

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

        codes = self.vq(z)

        codebook_loss = torch.mean((codes - z.detach()) ** 2)
        commitment_loss = torch.mean((codes.detach() - z) ** 2)

        # straight through
        codes = z + (codes - z).detach()

        return codes, codebook_loss, commitment_loss

    def forward_dec(self, x):
        codes, codebook_loss, commitment_loss = self.quantize(x)
        decoded = self.decoder(codes)

        return codes, decoded, codebook_loss, commitment_loss

    def forward(self, x):
        batch, channels, height, width = x.shape

        # Flatten image to vector
        x = x.flatten(1)

        latents = self.forward_enc(x)

        quantized_latents, decoded, codebook_loss, commitment_loss = self.forward_dec(
            latents
        )

        # Put decoded image back to original shape ###
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
