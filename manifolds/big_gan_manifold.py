import torch
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample)

from manifolds.data_manifold import DataManifold


class BigGANManifold(DataManifold):
    """
    Represents a BigGAN manifold with base functionality.
    """

    def __init__(self, output_size, dev):
        # Load pre-trained BigGAN model (implementation details omitted)
        self.dev = dev
        self.output_size = output_size

        self._init_big_gan_model()
        self._init_class_vector()

        super().__init__(self.big_gan_function, input_dim=128, output_dim=(3, output_size, output_size), type='biggan')

    def _init_big_gan_model(self):
        self.model = BigGAN.from_pretrained(f"biggan-deep-{int(self.output_size)}").to(self.dev)

    def _init_class_vector(self):
        self.class_vector = torch.from_numpy(one_hot_from_names(['golden retriever'], batch_size=1)).to(self.dev)

    def big_gan_function(self, z):
        if z.dim() == 1:  # Check if z has only one dimension
            z = z.unsqueeze(0)
        return self.model(z, self.class_vector, truncation=1.0).reshape(-1)


class BigGANManifoldNoClass(BigGANManifold):
    """
    Represents a BigGAN manifold without class conditioning.
    """

    def __init__(self, output_size, dev):
        super().__init__(output_size, dev)

    def _init_class_vector(self):
        # Create a zero class vector (no class conditioning)
        self.class_vector = torch.zeros(1, 1000).to(self.dev)


class BigGANManifoldAllClass(BigGANManifold):
    """
    Represents a BigGAN manifold with averaging class conditioning.
    """

    def __init__(self, output_size, dev):
        super().__init__(output_size, dev)

    def _init_class_vector(self):
        # Create a zero class vector (no class conditioning)
        # Create a mean class vector by averaging all class embeddings
        class_vector = torch.full((1, 1000), 0)
        class_vector[0, 300] = 0.5
        class_vector[0, 301] = 0.5
        self.class_vector = class_vector.to(self.dev)
