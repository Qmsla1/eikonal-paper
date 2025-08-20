from manifolds.big_gan_manifold import BigGANManifoldAllClass, BigGANManifold
from manifolds.helix_manifold import HelixManifold
from manifolds.mnist_manifold import MNISTManifold
from manifolds.sphere_manifold import SphereManifold
from manifolds.circle_manifold import CircleManifold
from manifolds.line_manifold import LineManifold


class ManifoldFactory:
    @staticmethod
    def get_instance(args, dev):
        manifold_type = args.manifold_type
        if manifold_type == 'helix':
            manifold = HelixManifold(dev)
        elif manifold_type == 'biggan_allclass':
            manifold = BigGANManifoldAllClass(args.latent_dim, dev)
        elif manifold_type == 'biggan':
            manifold = BigGANManifold(args.latent_dim, dev)
        elif manifold_type == 'sphere':
            manifold = SphereManifold(dev)
        elif manifold_type == 'mnist':
            manifold = MNISTManifold(dev)
        elif manifold_type == 'line':
            manifold = LineManifold(dev)
        else:
            raise ValueError(f'Invalid manifold type: {manifold_type}')

        return manifold

    @staticmethod
    def get_instance_direct(dev, manifold_type, latent_dim=0, output_dim=0):
        if manifold_type == 'helix':
            manifold = HelixManifold(dev)
        elif manifold_type == 'biggan_allclass':
            manifold = BigGANManifoldAllClass(latent_dim, dev)
        elif manifold_type == 'biggan':
            manifold = BigGANManifold(latent_dim, dev)
        elif manifold_type == 'sphere':
            manifold = SphereManifold(dev)
        elif manifold_type == 'circle':
            manifold = CircleManifold(dev=dev, output_dim=output_dim)
        elif manifold_type == 'mnist':
            manifold = MNISTManifold(dev)
        elif manifold_type == 'line':
            manifold = LineManifold(dev)
        else:
            raise ValueError(f'Invalid manifold type: {manifold_type}')

        return manifold