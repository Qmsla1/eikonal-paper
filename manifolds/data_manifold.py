from typing import Callable

import torch


class DataManifold:
    """
    Represents a data manifold with a mapping function, input and output dimensions,
    and a type identifier.
    """

    def __init__(self, func: Callable, input_dim: int, output_dim: object, type: str):
        """
        Initializes a DataManifold object.

        Args:
            func (torch.nn.Module): The function (typically a neural network) that maps
                                     from the latent space to the data space.
            input_dim (int): The dimension of the latent space (input dimension).
            output_dim (int): The dimension of the data space (output dimension).
            type (str): The type of the manifold (e.g., "helix", "sphere", "mnist", "biggan").
        """

        self.func = func
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.type = type

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        """
        Applies the manifold's mapping function to a latent vector.

        Args:
            z (torch.Tensor): A latent vector representing a point in the manifold.

        Returns:
            torch.Tensor: The corresponding data point mapped onto the manifold.
        """

        return self.func(z)


