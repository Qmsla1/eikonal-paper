import torch
import torch.nn as nn

from manifolds.data_manifold import DataManifold


class MNISTManifold(DataManifold):
    """
    Represents an MNIST manifold using a pre-trained generator network.
    """

    def __init__(self, dev, latent_dim=100, pretrained_model='./mnist_generator.pth'):
        """
        Initializes an MNISTManifold object.

        Args:
            latent_dim (int): The dimension of the latent space.
        """
        self.dev = dev
        self.latent_dim = latent_dim
        self.pretrained_model = pretrained_model

        self._init_mnist_generator()

        super().__init__(self.mnist_function, input_dim=latent_dim, output_dim=(28, 28), type='mnist')

    def _init_mnist_generator(self):
        self.mnist_generator = MnistGenerator(self.latent_dim)
        try:
            self.mnist_generator.load_state_dict(torch.load(self.pretrained_model, map_location=self.dev))
            self.mnist_generator.to(self.dev)
            self.mnist_generator.eval()
        except:
            print(f"Warning: Failed to found mnist model pre trained weights at {self.pretrained_model}")
            return None

    def mnist_function(self, z):
        return self.mnist_generator(z.reshape(1, -1, 1, 1)).reshape(-1)


class MnistGenerator(nn.Module):
    def __init__(self, latent_dim):
        super(MnistGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 7, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
