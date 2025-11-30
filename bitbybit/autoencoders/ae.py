import torch.nn as nn
import torch.nn.functional as F

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
        print('Encoder')
        for i, conv in enumerate(self.convs):
            x = conv(x)
            x = F.relu(x)

            if self.verbose:
                print(f"After layer {i+1}: {x.shape}")

        return x


class Decoder(nn.Module):

    def __init__(self, out_channels=1, hidden_dim=256, n_layers=6):
        super().__init__()
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
        print('Decoder')
        for i, deconv in enumerate(self.deconvs):
            x = deconv(x)
            if i < len(self.deconvs) - 1:
                x = F.relu(x)
                if self.verbose:
                    print(f"After layer {i+1}: {x.shape}")
        if self.verbose:
            print(f"Output: {x.shape}")
        return x


class AE(nn.Module):
    def __init__(self, hidden_dim=512, n_layers=6, verbose=False):
        super().__init__()
        self.enc = Encoder(
            in_channels=1,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            verbose=verbose
        )
        self.dec = Decoder(
            out_channels=1,
            hidden_dim=hidden_dim,
            n_layers=n_layers
        )

    def forward(self, x):
        z = self.enc(x) # [B, C, T_e]
        y = self.dec(z) # [B, C, T]
        return y
