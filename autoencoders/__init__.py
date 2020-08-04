from autoencoders.models.bigae import BigAE
from autoencoders.models.biggan import (
    Generator128 as BigGAN128,
    Generator256 as BigGAN256,
)
from autoencoders.distributions import DiracDistribution, DiagonalGaussianDistribution

def get_model(name):
    _models = {
        "bigae_animals": lambda: BigAE.from_pretrained("animals"),
        "bigae_animalfaces": lambda: BigAE.from_pretrained("animalfaces"),
        "biggan_128": lambda: BigGAN128.from_pretrained(),
        "biggan_256": lambda: BigGAN256.from_pretrained(),
    }
    return _models[name]()
