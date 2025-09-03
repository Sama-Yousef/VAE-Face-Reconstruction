import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from model_classes.BetaVAE import BetaVAE



class DFCVAE(BetaVAE):

    def __init__(self, latent_size=100, beta=1):
        super(DFCVAE, self).__init__()

        self.latent_size = latent_size
        self.beta = beta

        # encoder
        self.e1 = self._conv(3, 32)
        self.e2 = self._conv(32, 64)
        self.e3 = self._conv(64, 128)
        self.e4 = self._conv(128, 256)
        self.fc_mu = nn.Linear(4096, latent_size)
        self.fc_var = nn.Linear(4096, latent_size)

        # decoder
        self.d1 = self._upconv(256, 128)
        self.d2 = self._upconv(128, 64)
        self.d3 = self._upconv(64, 32)
        self.d4 = self._upconv(32, 3)
        self.fc_z = nn.Linear(latent_size, 4096)

    def encode(self, x):
        x = F.leaky_relu(self.e1(x))
        x = F.leaky_relu(self.e2(x))
        x = F.leaky_relu(self.e3(x))
        x = F.leaky_relu(self.e4(x))
        x = x.view(-1, 4096)
        return self.fc_mu(x), self.fc_var(x)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(-1, 256, 4, 4)
        z = F.leaky_relu(self.d1(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d2(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d3(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d4(F.interpolate(z, scale_factor=2)))
        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )

    def _upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=1
            ),
            nn.BatchNorm2d(out_channels),
        )