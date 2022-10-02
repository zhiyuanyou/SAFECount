from __future__ import division

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import datasets.transforms as T


class BaseDataset(Dataset):
    """
    A dataset should implement
        1. __len__ to get size of the dataset, Required
        2. __getitem__ to get a single data, Required

    """

    def __init__(self):
        super(BaseDataset, self).__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class BaseTransform(object):
    """
    Resize image, density, and boxes
    """

    def __init__(
        self, size_rsz, hflip=False, vflip=False, rotate=False, gamma=False, gray=False
    ):
        self.size_rsz = size_rsz  # [h, w]
        self.hflip = hflip
        self.vflip = vflip
        self.rotate = rotate
        self.gamma = gamma
        self.gray = gray

    def __call__(self, image, density, boxes, points, size_orig):
        h_orig, w_orig = size_orig
        h_rsz, w_rsz = self.size_rsz
        h_scale, w_scale = h_rsz / h_orig, w_rsz / w_orig

        # resize image
        image = cv2.resize(image, (w_rsz, h_rsz))
        # gamma
        if self.gamma:
            gamma_range = self.gamma["range"]
            prob = self.gamma["prob"]
            gamma_fn = T.Gamma(gamma_range, prob)
            image = gamma_fn(image)
        image = Image.fromarray(image)

        # resize density
        cnt_orig = np.sum(density)
        density = cv2.resize(density, (w_rsz, h_rsz))
        if not cnt_orig == 0:
            cnt_rsz = np.sum(density)
            density = density * (cnt_orig / cnt_rsz)
        density = Image.fromarray(density)

        # resize boxes
        boxes_rsz = []
        for box in boxes:
            y_tl, x_tl, y_br, x_br = box
            y_tl = int(y_tl * h_scale)
            y_br = int(y_br * h_scale)
            x_tl = int(x_tl * w_scale)
            x_br = int(x_br * w_scale)
            boxes_rsz.append([y_tl, x_tl, y_br, x_br])

        # resize points
        points_rsz = []
        for point in points:
            x, y = point
            y = int(y * h_scale)
            x = int(x * w_scale)
            points_rsz.append([x, y])

        # hflip, vflip, rotate, gray
        if self.hflip:
            prob = self.hflip.get("prob", 0.5)
            transform_fn = T.RandomHFlip(self.size_rsz, prob)
            image, density, boxes_rsz, points_rsz = transform_fn(
                image, density, boxes_rsz, points_rsz
            )
        if self.vflip:
            prob = self.vflip.get("prob", 0.5)
            transform_fn = T.RandomVFlip(self.size_rsz, prob)
            image, density, boxes_rsz, points_rsz = transform_fn(
                image, density, boxes_rsz, points_rsz
            )
        if self.rotate:
            assert (
                boxes == [] and points == []
            ), "random rotate coud only be used in custom_exemplar_dataset"
            transform_fn = T.RandomRotation(self.rotate["degrees"])
            image, density = transform_fn(image, density)
        if self.gray:
            prob = self.gray.get("prob", 0.5)
            transform_fn = T.RandomGrayscale(prob)
            image = transform_fn(image)

        return image, density, boxes_rsz, points_rsz


class ExemplarTransform(object):
    """
    Resize image, and box
    """

    def __init__(self, size_rsz):
        self.size_rsz = size_rsz  # [h, w]

    def __call__(self, image, box, size_orig):
        h_orig, w_orig = size_orig
        h_rsz, w_rsz = self.size_rsz
        h_scale, w_scale = h_rsz / h_orig, w_rsz / w_orig

        # resize image
        image = cv2.resize(image, (w_rsz, h_rsz))
        image = Image.fromarray(image)

        # resize box
        y_tl, x_tl, y_br, x_br = box
        y_tl = int(y_tl * h_scale)
        y_br = int(y_br * h_scale)
        x_tl = int(x_tl * w_scale)
        x_br = int(x_br * w_scale)
        box_rsz = [y_tl, x_tl, y_br, x_br]

        return image, box_rsz
