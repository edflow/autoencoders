import torch
import numpy as np
from PIL import Image
from autoencoders.models.bigae import BigAE

def load_img(path):
    I = Image.open(path)
    x = torch.tensor(np.array(I)/127.5-1.0)
    x = x[None,...].transpose(3,2).transpose(2,1).float()
    return x

def tensor_to_img(x):
    x = x[0].transpose(0,1).transpose(1,2)
    x = x.detach().cpu().numpy()
    x = ((x+1.0)*127.5).astype(np.uint8)
    return Image.fromarray(x)


def test_bigae():
    m = BigAE.from_pretrained("animals")

    xin = load_img("assets/fox.jpg")
    p = m.encode(xin)
    xout = m.decode(p.mode())

    xout = tensor_to_img(xout)
    xtarget = Image.open("assets/fox_bigae.png")

    assert np.allclose(xout, xtarget)
