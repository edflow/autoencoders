from autoencoders.models.bigae import BigAE
from autoencoders.distributions import DiracDistribution, DiagonalGaussianDistribution

def get_model(name):
    _models = {
        "bigae_animals": lambda: BigAE.from_pretrained("animals"),
    }
    return _models[name]()
