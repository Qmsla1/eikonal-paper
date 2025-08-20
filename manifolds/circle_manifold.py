import torch
import matplotlib.pyplot as plt
import numpy as np
from manifolds.data_manifold import DataManifold


class CircleManifold(DataManifold):
    """
    Represents a circle manifold.
    """

    def __init__(self, dev, output_dim=3):
        self.dev = dev
        super().__init__(self.circle_function, input_dim=1, output_dim=output_dim, type='circle')

    def circle_function(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if self.output_dim == 2:
            return torch.stack([torch.cos(z[:, 0]), torch.sin(z[:, 0])], dim=1)
        else:
            return torch.hstack([torch.cos(z[:, 0]).reshape(-1,1), torch.sin(z[:, 0]).reshape(-1,1), torch.zeros((z.shape[0], self.output_dim - 2), device=z.device)])

    def sample_n_points(self, n_points):
        z = torch.linspace(0, 2 * np.pi, n_points, device=self.dev)
        return self.circle_function(z.view(-1, 1))

    def sample(self, n_points, delta=0.1):
        size = (int(n_points * 1.1), 3)

        # Generate random data
        rand_data = torch.distributions.Uniform(-2, 2).sample(size).to(self.dev)
        desired_norm_vec = torch.distributions.Uniform(0, 2).sample((int(n_points * 1.1), 1)).to(self.dev)

        # Compute the norm and normalize the data
        norm_vec = torch.norm(rand_data, dim=1, keepdim=True)
        data = (rand_data * desired_norm_vec) / norm_vec

        # Apply the analytic projection
        data_xy0 = self.analytic_projection(data)

        # Filter elements outside the desired range
        xy_norm_vec = torch.norm(data - data_xy0, dim=1)
        filter_map = xy_norm_vec > delta
        outside_manifold_data = data[filter_map, :]

        return outside_manifold_data

    def sample_test(self, n_points, delta=0.1):
        size = (int(n_points * 1.1), 3)

        # Generate random data
        rand_data = torch.distributions.Uniform(-10, 10).sample(size).to(self.dev)
        desired_norm_vec = torch.distributions.Uniform(0, 2).sample((int(n_points * 1.1), 1)).to(self.dev)
        # Compute the norm and normalize the data
        norm_vec = torch.norm(rand_data, dim=1, keepdim=True)
        data = (rand_data * desired_norm_vec) / norm_vec
        # Apply the analytic projection
        data_xy0 = self.analytic_projection(data)
        # Filter elements outside the desired range
        xy_norm_vec = torch.norm(data - data_xy0, dim=1)
        filter_map = xy_norm_vec > delta
        outside_manifold_data = data[filter_map, :]
        return outside_manifold_data

    def analytic_projection(self, x_train):
        xy_norm_vec = torch.norm(x_train[:, :-1], dim=1, keepdim=True)
        data_xy = x_train[:, :-1] / xy_norm_vec
        zeros_vec = torch.zeros((data_xy.shape[0], 1),device=self.dev)
        data_xy0 = torch.cat((data_xy, zeros_vec), dim=1)
        return data_xy0

    def distance_function(self, x):
        norm_circle_plane = torch.norm(x[:, :2], dim=1)
        if x.shape[1] == 2:
            return torch.abs(norm_circle_plane - 1)
        else:
            norm_other_dims = torch.norm(x[:, 2:], dim=1)
            return torch.sqrt((norm_circle_plane - 1) ** 2 + norm_other_dims ** 2)

    def cartesian_to_spherical(self, pt):
        x = pt[:, 0]
        y = pt[:, 1]
        phi = torch.atan2(y, x)  # todo check if it's the x,y ?
        return phi
