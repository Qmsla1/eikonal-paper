import torch
import numpy as np
from manifolds.data_manifold import DataManifold


class LineManifold(DataManifold):
    """
    Represents a helix manifold.
    """

    def __init__(self, dev, y=1):
        self.dev = dev
        self._y = y
        super().__init__(self.line_function, input_dim=1, output_dim=2, type='line')

    def line_function(self, z):
        y = torch.full(z.shape, self._y, device=self.dev)
        return torch.hstack([z, y])

    def sample(self, n_points, delta=0.1):
        size = (int(n_points * 1.1), 2)
        data = torch.distributions.Uniform(-2, 2).sample(size)
        filter_map = (data[:, 1] > self._y + delta) | (data[:, 1] < self._y - delta)
        outside_manifold_data = data[filter_map, :]

        return outside_manifold_data

    def sample_test(self, n_points, delta=0.1):
        size = (int(n_points * 1.1), 2)
        data = torch.distributions.Uniform(-10, 10).sample(size)
        filter_map = (data[:, 1] > self._y + delta) | (data[:, 1] < self._y - delta)
        outside_manifold_data = data[filter_map, :]

        return outside_manifold_data

    def analytic_projection(self, x_train):
        data_line = x_train[:, :-1]
        y_to_stack = torch.full((data_line.shape[0], 1), self._y,device=self.dev)
        data_line = torch.hstack((data_line, y_to_stack)).to(self.dev)
        return data_line
