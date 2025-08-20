import torch
from manifolds.data_manifold import DataManifold


class HelixManifold(DataManifold):
    """
    Represents a helix manifold.
    """

    def __init__(self, dev):
        self.dev = dev
        super().__init__(self.helix_function, input_dim=1, output_dim=3, type='helix')

    @staticmethod
    def helix_function(z):
        return torch.hstack([torch.cos(z), torch.sin(z), z])
