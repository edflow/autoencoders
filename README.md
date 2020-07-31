# Autoencoders

A collection of autoencoder models in PyTorch.

## Quickstart

Install

```
git clone https://github.com/edflow/autoencoders.git
cd autoencoders
pip install -e .
```

and run a reconstruction demo:

```
edexplore -b configs/demo.yaml
```

To reconstruct your own images, just add them to the `assets` folder.

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

## Data

### ImageNet

You can take a look with

```
edexplore --dataset autoencoders.data.ImageNetTrain
edexplore --dataset autoencoders.data.ImageNetVal
```

Note that the datasets will be downloaded (through [Academic
Torrents](http://academictorrents.com/)) and prepared the first time they are
used. Since ImageNet is quite large, this requires a lot of disk space and
time. If you already have ImageNet on your disk, you can speed things up by
putting the data into `${XDG_CACHE}/autoencoders/data/ILSVRC2012_train/data/`
(which defaults to `~/.cache/autoencoders/data/ILSVRC2012_train/data/`). It
should have the following structure:

```
${XDG_CACHE}/autoencoders/data/ILSVRC2012_train/data/
├── n01440764
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   ├── ...
├── n01443537
│   ├── n01443537_10007.JPEG
│   ├── n01443537_10014.JPEG
│   ├── ...
├── ...
```

If you haven't extracted the data, you can also place
`ILSVRC2012_img_train.tar` into
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_train/`, which will then be
extracted into above structure without downloading it again.
