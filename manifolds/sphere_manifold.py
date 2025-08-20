import torch
import matplotlib.pyplot as plt
import numpy as np
from manifolds.data_manifold import DataManifold


class SphereManifold(DataManifold):
    """
    Represents a sphere manifold.
    """

    def __init__(self, dev):
        self.dev = dev
        super().__init__(self.sphere_function, input_dim=2, output_dim=3, type='sphere')

    @staticmethod
    def sphere_function(z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        return torch.stack(
            [torch.sin(z[:, 0]) * torch.cos(z[:, 1]), torch.sin(z[:, 0]) * torch.sin(z[:, 1]), torch.cos(z[:, 0])],
            dim=1)

    def sample(self, n_points, delta=0.1):
        size = (int(n_points * 1.1), 3)

        # Generate random data
        rand_data = torch.distributions.Uniform(-2, 2).sample(size).to(self.dev)
        desired_norm_vec = torch.distributions.Uniform(0, 2).sample((int(n_points * 1.1), 1)).to(self.dev)

        # Compute the norm and normalize the data
        norm_vec = torch.norm(rand_data, dim=1, keepdim=True)
        data = (rand_data * desired_norm_vec) / norm_vec

        # Filter elements outside the desired range
        norm_vec = norm_vec.squeeze()  # Remove the extra dimension added by keepdim=True
        filter_map = (norm_vec > 1 + delta) | (norm_vec < 1 - delta)
        outside_manifold_data = data[filter_map, :]

        return outside_manifold_data

    def sample_test(self, n_points, delta=0.1):
        size = (int(n_points * 1.1), 3)

        # Generate random data
        rand_data = torch.distributions.Uniform(-10, 10).sample(size).to(self.dev)
        desired_norm_vec = torch.distributions.Uniform(0, 10).sample((int(n_points * 1.1), 1)).to(self.dev)

        # Compute the norm and normalize the data
        norm_vec = torch.norm(rand_data, dim=1, keepdim=True)
        data = (rand_data * desired_norm_vec) / norm_vec

        # Filter elements outside the desired range
        norm_vec = norm_vec.squeeze()  # Remove the extra dimension added by keepdim=True
        filter_map = (norm_vec > 1 + delta) | (norm_vec < 1 - delta)
        outside_manifold_data = data[filter_map, :]

        return outside_manifold_data

    def analytic_projection(self, x_train):
        norm_vec = torch.linalg.norm(x_train, axis=1)
        data = x_train / norm_vec.reshape(-1, 1)
        return data

    @classmethod
    def plot_sphere(cls):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        N = 100
        # Generate grid points for theta and phi
        theta = torch.linspace(0, np.pi, N)
        phi = torch.linspace(0, 2 * np.pi, N)
        theta, phi = torch.meshgrid(theta, phi)
        # Compute sphere coordinates
        sphere_points = cls.sphere_function(torch.stack([theta.reshape(-1), phi.reshape(-1)], dim=1))
        # Reshape for plotting
        x = sphere_points[:, 0].reshape(N, N)
        y = sphere_points[:, 1].reshape(N, N)
        z = sphere_points[:, 2].reshape(N, N)
        # Plot surface
        ax.plot_surface(x.numpy(), y.numpy(), z.numpy(), rstride=1, cstride=1, cmap='viridis')
        plt.savefig('sphere_sampled.png')

    def cartesian_to_spherical(self, pt):
        x = pt[:, 0]
        y = pt[:, 1]
        z = pt[:, 2]
        theta = torch.acos(z)
        phi = torch.atan2(y, x)
        return theta, phi
