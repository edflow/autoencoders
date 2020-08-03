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


class ImageNetBase(edu.DatasetMixin):
    def __init__(self, config=None):
        self.config = config or dict()
        self.logger = edu.get_logger(self)
        self._prepare()
        self._prepare_synset_to_human()
        self._load()

    def _prepare(self):
        raise NotImplementedError()

    def _prepare_synset_to_human(self):
        SIZE = 2655750
        URL = "https://heibox.uni-heidelberg.de/f/9f28e956cd304264bb82/?dl=1"
        self.human_dict = os.path.join(self.root, "synset_human.txt")
        if (not os.path.exists(self.human_dict) or
                not os.path.getsize(self.human_dict)==SIZE):
            download(URL, self.human_dict)

    def _load(self):
        with open(self.txt_filelist, "r") as f:
            self.relpaths = f.read().splitlines()

        assert len(self.relpaths) == self.expected_length

        self.synsets = [p.split("/")[0] for p in self.relpaths]
        self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]

        unique_synsets = np.unique(self.synsets)
        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        self.class_labels = [class_dict[s] for s in self.synsets]

        with open(self.human_dict, "r") as f:
            human_dict = f.read().splitlines()
            human_dict = dict(line.split(maxsplit=1) for line in human_dict)

        self.human_labels = [human_dict[s] for s in self.synsets]

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


class ImageNetTrain(ImageNetBase):
    NAME = "ILSVRC2012_train"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "a306397ccf9c2ead27155983c254227c0fd938e2"
    FILES = [
        "ILSVRC2012_img_train.tar",
    ]
    SIZES = [
        147897477120,
    ]

    def _prepare(self):
        self.random_crop = retrieve(self.config, "ImageNetTrain/random_crop",
                                    default=True)
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        self.expected_length = 1281167
        if not edu.is_prepared(self.root):
            # prep
            self.logger.info("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    import academictorrents as at
                    atpath = at.get(self.AT_HASH, datastore=self.root)
                    assert atpath == path

                self.logger.info("Extracting {} to {}".format(path, datadir))
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                self.logger.info("Extracting sub-tars.")
                subpaths = sorted(glob.glob(os.path.join(datadir, "*.tar")))
                for subpath in tqdm(subpaths):
                    subdir = subpath[:-len(".tar")]
                    os.makedirs(subdir, exist_ok=True)
                    with tarfile.open(subpath, "r:") as tar:
                        tar.extractall(path=subdir)


            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            edu.mark_prepared(self.root)


class ImageNetValidation(ImageNetBase):
    NAME = "ILSVRC2012_validation"
    URL = "http://www.image-net.org/challenges/LSVRC/2012/"
    AT_HASH = "5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5"
    VS_URL = "https://heibox.uni-heidelberg.de/f/3e0f6e9c624e45f2bd73/?dl=1"
    FILES = [
        "ILSVRC2012_img_val.tar",
        "validation_synset.txt",
    ]
    SIZES = [
        6744924160,
        1950000,
    ]

    def _prepare(self):
        self.random_crop = retrieve(self.config, "ImageNetValidation/random_crop",
                                    default=False)
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)
        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        self.expected_length = 50000
        if not edu.is_prepared(self.root):
            # prep
            self.logger.info("Preparing dataset {} in {}".format(self.NAME, self.root))

            datadir = self.datadir
            if not os.path.exists(datadir):
                path = os.path.join(self.root, self.FILES[0])
                if not os.path.exists(path) or not os.path.getsize(path)==self.SIZES[0]:
                    import academictorrents as at
                    atpath = at.get(self.AT_HASH, datastore=self.root)
                    assert atpath == path

                self.logger.info("Extracting {} to {}".format(path, datadir))
                os.makedirs(datadir, exist_ok=True)
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                vspath = os.path.join(self.root, self.FILES[1])
                if not os.path.exists(vspath) or not os.path.getsize(vspath)==self.SIZES[1]:
                    download(self.VS_URL, vspath)

                with open(vspath, "r") as f:
                    synset_dict = f.read().splitlines()
                    synset_dict = dict(line.split() for line in synset_dict)

                self.logger.info("Reorganizing into synset folders")
                synsets = np.unique(list(synset_dict.values()))
                for s in synsets:
                    os.makedirs(os.path.join(datadir, s), exist_ok=True)
                for k, v in synset_dict.items():
                    src = os.path.join(datadir, k)
                    dst = os.path.join(datadir, v)
                    shutil.move(src, dst)


            filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
            filelist = [os.path.relpath(p, start=datadir) for p in filelist]
            filelist = sorted(filelist)
            filelist = "\n".join(filelist)+"\n"
            with open(self.txt_filelist, "w") as f:
                f.write(filelist)

            edu.mark_prepared(self.root)


class ImageNetAnimalsBase(edu.DatasetMixin):
    def __init__(self, config=None):
        self._prepare_animal_synsets()
        with open(self.animal_synsets, "r") as f:
            animal_synsets = f.read().splitlines()

        assert len(animal_synsets)==149
        animal_synsets = set(animal_synsets)

        data = self.BASE_DATASET(config)
        indices = [i for i in range(len(data)) if data.labels["synsets"][i] in animal_synsets]
        self.data = SubDataset(data, indices)

    def _prepare_animal_synsets(self):
        SIZE = 1490
        URL = "https://heibox.uni-heidelberg.de/f/c18cdf02ea0b4e758729/?dl=1"
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        self.root = os.path.join(cachedir, "autoencoders/data/ImageNetAnimals")
        self.animal_synsets = os.path.join(self.root, "animal_synsets.txt")
        if (not os.path.exists(self.animal_synsets) or
                not os.path.getsize(self.animal_synsets)==SIZE):
            download(URL, self.animal_synsets)


class ImageNetAnimalsTrain(ImageNetAnimalsBase):
    BASE_DATASET=ImageNetTrain


class ImageNetAnimalsValidation(ImageNetAnimalsBase):
    BASE_DATASET=ImageNetValidation


class AnimalFacesBase(ImageNetBase):
    NAME = "AnimalFaces"
    COOR_URL = "https://github.com/NVlabs/FUNIT/raw/master/datasets/animalface_coordinates.txt"
    TEST_URL = "https://github.com/NVlabs/FUNIT/raw/master/datasets/animals_list_test.txt"
    TRAIN_URL = "https://github.com/NVlabs/FUNIT/raw/master/datasets/animals_list_train.txt"
    SHARED_TEST_URL = "https://heibox.uni-heidelberg.de/f/f44e33a5155b4b2fab47/?dl=1"
    SHARED_TRAIN_URL = "https://heibox.uni-heidelberg.de/f/20cb4d546b304e5aba99/?dl=1"
    RESTRICTED_TEST_URL = "https://heibox.uni-heidelberg.de/f/91fc6c34d16141afa6e4/?dl=1"
    RESTRICTED_TRAIN_URL = "https://heibox.uni-heidelberg.de/f/a82aa5471f534026ad4f/?dl=1"

    def _prepare(self):
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
        self.root = os.path.join(cachedir, "autoencoders/data/AnimalFaces")
        self.logger.info("Using data located at {}".format(self.root))

        os.makedirs(self.root, exist_ok=True)
        self.datadir = os.path.join(self.root, "data")

        if not edu.is_prepared(self.root):
            self.logger.info("Preparing dataset {} in {}".format(self.NAME, self.root))

            if not os.path.exists(self.datadir):
                os.makedirs(self.datadir, exist_ok=True)
                imagenet = ImageNetTrain()

                coor_path = os.path.join(self.root, "animalface_coordinates.txt")
                if not os.path.exists(coor_path):
                    download(self.COOR_URL, coor_path)

                with open(coor_path, "r") as f:
                    animalface_coordinates = f.readlines()

                for line in tqdm(animalface_coordinates):
                    ls = line.strip().split(' ')
                    img_name = os.path.join(imagenet.datadir, ls[0])
                    img = Image.open(img_name)
                    img = img.convert('RGB')
                    x = int(ls[1])
                    y = int(ls[2])
                    w = int(ls[3])
                    h = int(ls[4])
                    crop = img.crop((x, y, w, h))

                    out_name = os.path.join(self.datadir,
                                            '%s_%d_%d_%d_%d.jpg' % (ls[0], x, y, w, h))
                    os.makedirs(os.path.dirname(out_name), exist_ok=True)
                    crop.save(out_name)

            train_path = os.path.join(self.root, "animals_list_train.txt")
            if not os.path.exists(train_path):
                download(self.TRAIN_URL, train_path)

            test_path = os.path.join(self.root, "animals_list_test.txt")
            if not os.path.exists(test_path):
                download(self.TEST_URL, test_path)

            shared_train_path = os.path.join(self.root, "shared_animalfaces_train.txt")
            if not os.path.exists(shared_train_path):
                download(self.SHARED_TRAIN_URL, shared_train_path)

            shared_test_path = os.path.join(self.root, "shared_animalfaces_test.txt")
            if not os.path.exists(shared_test_path):
                download(self.SHARED_TEST_URL, shared_test_path)

            restricted_train_path = os.path.join(self.root, "restricted_animalfaces_train.txt")
            if not os.path.exists(restricted_train_path):
                download(self.RESTRICTED_TRAIN_URL, restricted_train_path)

            restricted_test_path = os.path.join(self.root, "restricted_animalfaces_test.txt")
            if not os.path.exists(restricted_test_path):
                download(self.RESTRICTED_TEST_URL, restricted_test_path)

            edu.mark_prepared(self.root)


class AnimalFacesTrain(AnimalFacesBase):
    def _prepare(self):
        super()._prepare()
        self.random_crop = False
        self.txt_filelist = os.path.join(self.root, "animals_list_train.txt")
        self.expected_length = 93404


class AnimalFacesTest(AnimalFacesBase):
    def _prepare(self):
        super()._prepare()
        self.random_crop = False
        self.txt_filelist = os.path.join(self.root, "animals_list_test.txt")
        self.expected_length = 24080


class AnimalFacesSharedTrain(AnimalFacesBase):
    def _prepare(self):
        super()._prepare()
        self.random_crop = False
        self.txt_filelist = os.path.join(self.root, "shared_animalfaces_train.txt")
        self.expected_length = 105653


class AnimalFacesSharedTest(AnimalFacesBase):
    def _prepare(self):
        super()._prepare()
        self.random_crop = False
        self.txt_filelist = os.path.join(self.root, "shared_animalfaces_test.txt")
        self.expected_length = 11831


class AnimalFacesRestrictedTrain(AnimalFacesBase):
    def _prepare(self):
        super()._prepare()
        self.random_crop = False
        self.txt_filelist = os.path.join(self.root, "restricted_animalfaces_train.txt")
        self.expected_length = 8036


class AnimalFacesRestrictedTest(AnimalFacesBase):
    def _prepare(self):
        super()._prepare()
        self.random_crop = False
        self.txt_filelist = os.path.join(self.root, "restricted_animalfaces_test.txt")
        self.expected_length = 898


if __name__ == "__main__":
    from edflow.util import pp2mkdtable

    print("ImageNet")
    print("train")
    d = ImageNetTrain()
    print(len(d))
    e = d[0]
    print(pp2mkdtable(e))

    print("validation")
    d = ImageNetValidation()
    print(len(d))
    e = d[0]
    print(pp2mkdtable(e))

    print("AnimalFaces")
    print("train")
    d = AnimalFacesTrain()
    print(len(d))
    e = d[0]
    print(pp2mkdtable(e))

    print("validation")
    d = AnimalFacesTest()
    print(len(d))
    e = d[0]
    print(pp2mkdtable(e))
