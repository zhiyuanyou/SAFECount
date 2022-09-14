from __future__ import division

import json
import os

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from datasets.base_dataset import BaseDataset, BaseTransform, ExemplarTransform
from datasets.transforms import RandomColorJitter


def build_custom_exemplar_dataloader(cfg, training, distributed=True):
    rank = dist.get_rank()

    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])

    if training:
        hflip = cfg.get("hflip", False)
        vflip = cfg.get("vflip", False)
        rotate = cfg.get("rotate", False)
        gamma = cfg.get("gamma", False)
        gray = cfg.get("gray", False)
        transform_fn = BaseTransform(
            cfg["input_size"], hflip, vflip, rotate, gamma, gray
        )
    else:
        transform_fn = BaseTransform(cfg["input_size"], False, False, False, False)

    exemplar_fn = ExemplarTransform(cfg["input_size"])

    if training and cfg.get("colorjitter", None):
        colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])
    else:
        colorjitter_fn = None

    if rank == 0:
        print("building CustomDataset from: {}".format(cfg["meta_file"]))

    dataset = CustomDataset(
        cfg["img_dir"],
        cfg["density_dir"],
        cfg["meta_file"],
        cfg["exemplar"],
        transform_fn=transform_fn,
        exemplar_fn=exemplar_fn,
        normalize_fn=normalize_fn,
        colorjitter_fn=colorjitter_fn,
    )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )

    return data_loader


class CustomDataset(BaseDataset):
    def __init__(
        self,
        img_dir,
        density_dir,
        meta_file,
        cfg_exemplar,
        transform_fn,
        exemplar_fn,
        normalize_fn,
        colorjitter_fn=None,
    ):
        self.img_dir = img_dir
        self.density_dir = density_dir
        self.meta_file = meta_file
        self.cfg_exemplar = cfg_exemplar
        self.transform_fn = transform_fn
        self.exemplar_fn = exemplar_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn

        # construct metas
        with open(meta_file, "r+") as f_r:
            self.metas = []
            for line in f_r:
                meta = json.loads(line)
                self.metas.append(meta)

        # construct exemplar_metas
        exemplar_metas = []
        with open(cfg_exemplar["meta_file"], "r+") as f_r:
            for line in f_r:
                meta = json.loads(line)
                exemplar_metas.append(meta)
        if cfg_exemplar["num_exemplar"] > len(exemplar_metas):
            raise ValueError(
                "num_exemplar must smaller than the length of exemplar_metas!"
            )
        else:
            exemplar_metas = exemplar_metas[0 : cfg_exemplar["num_exemplar"]]

        # construct exemplar list
        exemplar_imgs = []
        exemplar_boxes = []
        for meta in exemplar_metas:
            image = cv2.imread(os.path.join(cfg_exemplar["img_dir"], meta["filename"]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            # resize to a fix size
            box = meta["box"]
            if self.exemplar_fn:
                image, box = self.exemplar_fn(image, box, (height, width))
            image = transforms.ToTensor()(image)
            box = torch.tensor(box, dtype=torch.float64)
            if cfg_exemplar["norm"] and self.normalize_fn:
                image = self.normalize_fn(image)
            exemplar_imgs.append(image.unsqueeze(0))
            exemplar_boxes.append(box.unsqueeze(0))
        self.exemplar_imgs = torch.cat(exemplar_imgs, dim=0)  # n x c x h x w
        self.exemplar_boxes = torch.cat(exemplar_boxes, dim=0)  # n x c x h x w

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        meta = self.metas[index]
        # read img
        img_name = meta["filename"]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        # read density
        density_name = meta["density"]
        density_path = os.path.join(self.density_dir, density_name)
        density = np.load(density_path)
        # transform
        if self.transform_fn:
            image, density, _, _ = self.transform_fn(
                image, density, [], [], (height, width)
            )
        if self.colorjitter_fn:
            image = self.colorjitter_fn(image)
        image = transforms.ToTensor()(image)
        density = transforms.ToTensor()(density)
        if self.normalize_fn:
            image = self.normalize_fn(image)
        return {
            "filename": img_name,
            "height": height,
            "width": width,
            "image": image,
            "density": density,
            "exemplar_imgs": self.exemplar_imgs,
            "exemplar_boxes": self.exemplar_boxes,
        }
