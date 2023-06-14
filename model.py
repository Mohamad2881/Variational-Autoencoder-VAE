
from torch import nn
import torch

class VAE(nn.Module):
    def __init__(self):
        super().__init__()

        # using conv with strides of two instead of max pooling layers.
        # because during inference we will not have the encoder part, so we cannot
        # use max unpooling as they will require indices from the encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=(3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),

            nn.Flatten()
        )
        
        self.z_mean = nn.Linear(64*4*4, 10)
        self.z_log_var = nn.Linear(64*4*4, 10)

        self.decoder = nn.Sequential(
                nn.Linear(10, 64*4*4),
                nn.Unflatten(dim=1, unflattened_size=(64, 4, 4)),
                
                nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(2, 2), output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=(3, 3)),
                nn.ReLU(),

                nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, kernel_size=(3, 3)),
                nn.Sigmoid(),  # because input images will have values between 0 and 1. 
        )

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_log_var.size())#.to(z_log_var.get_device())
        if z_log_var.is_cuda:
            eps = eps.cuda()
        # Calculate sigma (std) from log variance
        std = torch.exp(z_log_var/2.)

        return z_mu + (eps * std)

    def forward(self, x):
        # Encoder
        encoded = self.encoder(x)
        # print(encoded.size())
        
        # Get mu and sigma from encoder
        z_mean, z_log_var = self.z_mean(encoded), self.z_log_var(encoded)   # (batch_size, latent_dim=10)
        
        # sample a latent vector z
        z = self.reparameterize(z_mean, z_log_var)  # (batch_size, latent_dim=10)

        # Decoder
        decoded = self.decoder(z)
        # print(x.size())

        return encoded, z_mean, z_log_var, decoded  # (batch_size, 64*4*4)


class VAE_celeb(nn.Module):
    def __init__(self, latent_dim=200):
        super().__init__()
        self.latent_dim = latent_dim
        # using conv with strides of two instead of max pooling layers.
        # because during inference we will not have the encoder part, so we cannot
        # use max unpooling as they will require indices from the encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.25),

            nn.Flatten()
        )

        self.z_mean = nn.Linear(128 * 13 * 13, latent_dim)
        self.z_log_var = nn.Linear(128 * 13 * 13, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 13 * 13),
            nn.Unflatten(dim=1, unflattened_size=(128, 13, 13)),

            nn.ConvTranspose2d(128, 128, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.25),

            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),

            nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(2, 2), output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.25),

            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.25),

            nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(2, 2), output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.25),

            nn.ConvTranspose2d(32, 3, kernel_size=(3, 3)),
            nn.Sigmoid(),  # because input images will have values between 0 and 1.
        )

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_log_var.size())  # .to(z_log_var.get_device())
        if z_log_var.is_cuda:
            eps = eps.cuda()
        # Calculate sigma (std) from log variance
        std = torch.exp(z_log_var / 2.)

        return z_mu + (eps * std)

    def encode(self, x):
        # Encoder
        encoded = self.encoder(x)
        # print(encoded.size())

        # # Get mu and sigma from encoder
        z_mean, z_log_var = self.z_mean(encoded), self.z_log_var(encoded)  # (batch_size, latent_dim)

        # # sample a latent vector z
        z = self.reparameterize(z_mean, z_log_var)  # (batch_size, latent_dim)

        return encoded, z, z_mean, z_log_var

    def forward(self, x):
        # Encoder
        encoded, z, z_mean, z_log_var = self.encode(x)

        # Decoder
        decoded = self.decoder(z)

        return encoded, z_mean, z_log_var, decoded