import torch.distributed as dist

from datasets.custom_dataset import build_custom_dataloader
from datasets.custom_exemplar_dataset import build_custom_exemplar_dataloader


def build(cfg, dataset_type, distributed):
    if dataset_type == "train":
        cfg.update(cfg.get("train", {}))
        training = True
    elif dataset_type == "val":
        cfg.update(cfg.get("val", {}))
        training = False
    elif dataset_type == "test":
        cfg.update(cfg.get("test", {}))
        training = False
    else:
        raise ValueError("dataset_type must among [train, val, test]!")

    dataset = cfg["type"]
    if dataset == "custom":
        data_loader = build_custom_dataloader(cfg, training, distributed)
    elif dataset == "custom_exemplar":
        data_loader = build_custom_exemplar_dataloader(cfg, training, distributed)
    else:
        raise NotImplementedError(f"{dataset} is not supported")

    return data_loader


def build_dataloader(cfg_dataset, distributed=True):
    rank = dist.get_rank()

    train_loader = None
    if cfg_dataset.get("train", None):
        train_loader = build(cfg_dataset, dataset_type="train", distributed=distributed)

    val_loader = None
    if cfg_dataset.get("val", None):
        val_loader = build(cfg_dataset, dataset_type="val", distributed=distributed)

    test_loader = None
    if cfg_dataset.get("test", None):
        test_loader = build(cfg_dataset, dataset_type="test", distributed=distributed)

    if rank == 0:
        print("build dataset done")

    return train_loader, val_loader, test_loader
