import torch
import torch.nn as nn

from edflow import get_logger
from edflow.util import retrieve

from autoencoders.distributions import DiagonalGaussianDistribution
from autoencoders.ckpt_util import get_ckpt_path


class BasicFullyConnectedNet(nn.Module):
    def __init__(self, dim, depth, hidden_dim=256, use_tanh=False, use_bn=False, out_dim=None):
        super(BasicFullyConnectedNet, self).__init__()
        layers = []
        layers.append(nn.Linear(dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.LeakyReLU())
        for d in range(depth):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(hidden_dim, dim if out_dim is None else out_dim))
        if use_tanh:
            layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class FlatVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        n_down = retrieve(config, "Model/n_down")
        z_dim = retrieve(config, "Model/z_dim")
        in_channels = retrieve(config, "Model/in_channels")
        mid_channels = retrieve(config, "Model/mid_channels", default=in_channels)
        use_bn = retrieve(config, "Model/use_bn", default=False)
        self.be_deterministic = retrieve(config, "Model/be_deterministic", default=False)

        self.encoder = BasicFullyConnectedNet(dim=in_channels, depth=n_down,
                                              hidden_dim=mid_channels,
                                              out_dim=in_channels,
                                              use_bn=use_bn)
        self.mu_layer = BasicFullyConnectedNet(in_channels, depth=n_down,
                                               hidden_dim=mid_channels,
                                               out_dim=z_dim,
                                               use_bn=use_bn)
        self.logvar_layer = BasicFullyConnectedNet(in_channels, depth=n_down,
                                                   hidden_dim=mid_channels,
                                                   out_dim=z_dim,
                                                   use_bn=use_bn)
        self.decoder = BasicFullyConnectedNet(dim=z_dim, depth=n_down + 1,
                                              hidden_dim=mid_channels,
                                              out_dim=in_channels,
                                              use_bn=use_bn)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)[:, :, None, None]
        logvar = self.logvar_layer(h)[:, :, None, None]
        return DiagonalGaussianDistribution(torch.cat((mu, logvar), dim=1), deterministic=self.be_deterministic)

    def decode(self, x):
        if len(x.shape) == 4:
            x = x.squeeze(-1).squeeze(-1)
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = self.encode(x).sample()
        x = self.decoder(x)
        return x

    def get_last_layer(self):
        return self.decoder.main[-1].weight

    @classmethod
    def from_pretrained(cls, name):
        if name is not "dequant_biggan":
            raise NotImplementedError
        config_dir = {"dequant_biggan":
                        {"Model":
                            {"in_channels": 128,
                            "n_down": 2,
                            "mid_channels": 4096,
                            "z_dim": 128
                            }
                        }
                    }
        ckpt_dict = {"dequant_biggan": "dequant_vae"}
        model = cls(config_dir[name])
        ckpt = get_ckpt_path(ckpt_dict[name])
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")))
        model.eval()
        return model


if __name__ == "__main__":
    model = FlatVAE.from_pretrained("dequant_biggan")
    print("loaded model.")