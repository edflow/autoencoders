# Autoencoders
![teaser](img/ae_teaser.png)

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
putting the data into `${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/`
(which defaults to `~/.cache/autoencoders/data/ILSVRC2012_{split}/data/`), where `{split}` is 
one of `train`/`validation`. It should have the following structure:

```
${XDG_CACHE}/autoencoders/data/ILSVRC2012_{split}/data/
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
`ILSVRC2012_img_train.tar`/`ILSVRC2012_img_val.tar` into
`${XDG_CACHE}/autoencoders/data/ILSVRC2012_train/`/`${XDG_CACHE}/autoencoders/data/ILSVRC2012_validation/`, 
which will then be extracted into above structure without downloading it again.

### AnimalFaces
This dataset was for example used in [FUNIT](https://nvlabs.github.io/FUNIT/). It contains all 149 carnivorous mammal animal 
classes from the ImageNet dataset. If this dataset is not available on your disk, the dataset will
automatically be build upon first use, following the cropping procedure as described and 
implemented [here](https://github.com/nvlabs/FUNIT/). Note that this requires that the __ImageNet__ dataset is already 
present as described above.

We provide two different splits of this dataset:
    
- The *"classic"* FUNIT split: Here, the train set contains images of
    119 animal classes, while the test set contains 30 *different* classes.
    - train split: __AnimalFacesTrain__
    - test split: __AnimalFacesTest__
    
- The *"shared"* split: __AnimalFacesShared__: Here, both the train and test split contain images of 
    *all* 149 classes.
    - train split: __AnimalFacesSharedTrain__
    - test split: __AnimalFacesSharedTest__      
    
### ImageNetAnimals
This dataset contains the same images as __AnimalFacesShared__, but *without* cropping.
- train split: __ImageNetAnimalsTrain__
- test split: __ImageNetAnimalsValidation__