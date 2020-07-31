import sys, os
import numpy as np
import albumentations
from PIL import Image
from edflow.iterators.batches import DatasetMixin
from edflow.util import retrieve


class ImagePaths(DatasetMixin):
    def __init__(self, paths, labels=None, size=None, random_crop=False):
        self.labels = labels if labels is not None else dict()
        self.size = size
        self.random_crop = random_crop

        assert not "file_path_" in self.labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def get_example(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class Folder(DatasetMixin):
    def __init__(self, config):
        folder = retrieve(config, "Folder/folder")
        size = retrieve(config, "Folder/size", default=0)
        random_crop = retrieve(config, "Folder/random_crop", default=False)

        relpaths = sorted(os.listdir(folder))
        abspaths = [os.path.join(folder, p) for p in relpaths]
        labels = {"relpaths": relpaths}

        self.data = ImagePaths(paths=abspaths,
                               labels=labels,
                               size=size,
                               random_crop=random_crop)
