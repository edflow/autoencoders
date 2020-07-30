# Autoencoders

A collection of autoencoder models in PyTorch.

## Installation

```
git clone https://github.com/edflow/autoencoders.git
cd autoencoders
pip install -e .
```

## Usage

```
import autoencoders

model = autoencoders.get_model("bigae_animals")
```

Currently available models are

- `bigae_animals`: Trained on `128x128x3` animal images from ImageNet.

Models implement `encode`, which returns a Distribution, and `decode` which
returns an image. A minimal working example is

```
import torch
import numpy as np
from PIL import Image
import autoencoders

model = autoencoders.get_model("bigae_animals")

x = Image.open("assets/fox.jpg")                      # example image
x = torch.tensor(np.array(x)/127.5-1.0)               # h,w,RGB in [-1,1]
x = x[None,...].transpose(3,2).transpose(2,1).float() # batch,RGB,h,w
p = model.encode(x)                                   # Distribution
z = p.sample()                                        # sampled latent code
xrec = model.decode(z)                                # batch,RGB,h,w
```
