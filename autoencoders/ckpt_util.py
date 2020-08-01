import os, hashlib
import requests
from tqdm import tqdm

URL_MAP = {
    "bigae_animals": "https://heibox.uni-heidelberg.de/f/f0adb4d509ea4132b9ea/?dl=1",
    "bigae_animalfaces": "TODO",
    "biggan_128": "https://heibox.uni-heidelberg.de/f/56ed256209fd40968864/?dl=1",
}
CKPT_MAP = {
    "bigae_animals": "autoencoders/bigae/animals-1672855.ckpt",
    "bigae_animalfaces": "autoencoders/bigae/animalfaces-631606.ckpt",
    "biggan_128": "autoencoders/biggan/biggan-128.pth",
}
MD5_MAP = {
    "bigae_animals": "6213882571854935226a041b8dcaecdd",
    "bigae_animalfaces": "7f379d6ebcbc03a710ef0605806f0b51",
    "biggan_128": "a2148cf64807444113fac5eede060d28",
}

def download(url, local_path, chunk_size = 1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream = True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total = total_size, unit = "B", unit_scale = True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size = chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)

def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()

def get_ckpt_path(name, root=None, check=False):
    assert name in URL_MAP
    cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    root = root if root is not None else os.path.join(cachedir, "autoencoders")
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path)==MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5==MD5_MAP[name], md5
    return path
