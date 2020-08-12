import sys, os, tarfile, pickle, glob, shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange
import urllib
from PIL import Image

from edflow.util import retrieve
import edflow.datasets.utils as edu
from edflow.data.dataset import SubDataset

from autoencoders.ckpt_util import download
from autoencoders.data.util import ImagePaths


class AwA2Base(edu.DatasetMixin):
    NAME = "AwA2"
    def __init__(self, config=None):
        self.config = config or dict()
        self.logger = edu.get_logger(self)
        self._prepare()
        self._load()

    def _prepare(self):
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
        SRC = {
            "AwA2-data.zip": "http://cvml.ist.ac.at/AwA2/AwA2-data.zip",
        }
        if not edu.is_prepared(self.root):
            # prep
            self.logger.info("Preparing dataset {} in {}".format(self.NAME, self.root))
            os.makedirs(self.root, exist_ok=True)

            datadir = os.path.join(self.root, "Animals_with_Attributes2")
            if not os.path.exists(datadir):
                datapath = os.path.join(self.root, "AwA2-data.zip")
                if not os.path.exists(datapath):
                    download(SRC["AwA2-data.zip"], datapath)
                edu.unpack(datapath)

            # make filelist
            images = list()
            for path, subdirs, files in os.walk(os.path.join(datadir, "JPEGImages")):
                for name in files:
                    searchname = name.lower()
                    if (searchname.rfind('jpg') != -1
                            or searchname.rfind('png') != -1
                            or searchname.rfind('jpeg') != -1):
                        filename = os.path.relpath(
                            os.path.join(path, name),
                            start=self.root)
                        images.append(filename)

            prng = np.random.RandomState(1)
            test = set(prng.choice(len(images), 5000, replace=False))
            train_images = [images[i] for i in range(len(images)) if not i in test]
            test_images = [images[i] for i in range(len(images)) if i in test]

            with open(os.path.join(self.root, "train.txt"), "w") as f:
                f.write("\n".join(train_images)+"\n")

            with open(os.path.join(self.root, "test.txt"), "w") as f:
                f.write("\n".join(test_images)+"\n")

            with open(os.path.join(self.root,
                                   "Animals_with_Attributes2/classes.txt"),
                      "r") as f:
                classes = f.read().splitlines()
                classes = [cls.split()[-1] for cls in classes]
                classes = sorted(classes)

            with open(os.path.join(self.root, "classes.txt"), "w") as f:
                f.write("\n".join(classes)+"\n")

            edu.mark_prepared(self.root)
            

    def _load(self):
        with open(self.get_txt_filelist(), "r") as f:
            self.relpaths = f.read().splitlines()

        assert len(self.relpaths) == self.expected_length

        self.synsets = [p.split("/")[2] for p in self.relpaths]
        self.abspaths = [os.path.join(self.root, p) for p in self.relpaths]

        with open(os.path.join(self.root, "classes.txt"), "r") as f:
            unique_synsets = f.read().splitlines()
            assert len(unique_synsets) == 50

        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        self.class_labels = [class_dict[s] for s in self.synsets]

        self.human_labels = self.synsets

        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "human_label": np.array(self.human_labels),
        }
        self.data = ImagePaths(self.abspaths,
                               labels=labels,
                               size=retrieve(self.config, "size", default=0),
                               random_crop=self.random_crop)


    def get_txt_filelist(self):
        raise NotImplementedError()


class AwA2Train(AwA2Base):
    expected_length = 37322-5000
    random_crop = True
    def get_txt_filelist(self):
        return os.path.join(self.root, "train.txt")

class AwA2Test(AwA2Base):
    expected_length = 5000
    random_crop = False
    def get_txt_filelist(self):
        return os.path.join(self.root, "test.txt")

if __name__ == "__main__":
    from edflow.util import pp2mkdtable

    print("train")
    d = AwA2Train()
    print(len(d))
    e = d[0]
    print(pp2mkdtable(e))
