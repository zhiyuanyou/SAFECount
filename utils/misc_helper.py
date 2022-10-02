import logging
import os
import random
import shutil
from collections.abc import Mapping
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist


def basicConfig(*args, **kwargs):
    return


# To prevent duplicate logs, we mask this baseConfig setting
logging.basicConfig = basicConfig


def create_logger(name, log_file, level=logging.INFO):
    log = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s"
    )
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def get_current_time():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    return current_time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def save_checkpoint(state, is_best, config):
    folder = config.save_path

    torch.save(state, os.path.join(folder, "ckpt.pth.tar"))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, "ckpt.pth.tar"),
            os.path.join(folder, "ckpt_best.pth.tar"),
        )

    if config.saver.get(
        "always_save", True
    ):  # default: save checkpoint after validate()
        epoch = state["epoch"]
        shutil.copyfile(
            os.path.join(folder, "ckpt.pth.tar"),
            os.path.join(folder, f"ckpt_{epoch}.pth.tar"),
        )


def load_state(path, model, optimizer=None):

    rank = dist.get_rank()

    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        if rank == 0:
            print("=> loading checkpoint '{}'".format(path))

        checkpoint = torch.load(path, map_location=map_func)

        # fix size mismatch error
        ignore_keys = []
        for k, v in checkpoint["state_dict"].items():
            if k in model.state_dict().keys():
                v_dst = model.state_dict()[k]
                if v.shape != v_dst.shape:
                    ignore_keys.append(k)
                    if rank == 0:
                        print(
                            "caution: size-mismatch key: {} size: {} -> {}".format(
                                k, v.shape, v_dst.shape
                            )
                        )

        for k in ignore_keys:
            checkpoint["state_dict"].pop(k)

        model.load_state_dict(checkpoint["state_dict"], strict=False)

        if rank == 0:
            ckpt_keys = set(checkpoint["state_dict"].keys())
            own_keys = set(model.state_dict().keys())
            missing_keys = own_keys - ckpt_keys
            for k in missing_keys:
                print("caution: missing keys from checkpoint {}: {}".format(path, k))

        if optimizer is not None:
            best_metric = checkpoint["best_metric"]
            epoch = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            if rank == 0:
                print(
                    "=> also loaded optimizer from checkpoint '{}' (Epoch {})".format(
                        path, epoch
                    )
                )
            return best_metric, epoch
    else:
        if rank == 0:
            print("=> no checkpoint found at '{}'".format(path))


def set_random_seed(seed=233, reproduce=False):
    np.random.seed(seed)
    torch.manual_seed(seed**2)
    torch.cuda.manual_seed(seed**3)
    random.seed(seed**4)
    torch.cuda.manual_seed_all(seed**5)

    if reproduce:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def to_device(input, device="cuda", dtype=None):
    """Transfer data between devidces"""

    if "image" in input:
        input["image"] = input["image"].to(dtype=dtype)

    def transfer(x):
        if torch.is_tensor(x):
            return x.to(device=device)
        elif isinstance(x, list):
            return [transfer(_) for _ in x]
        elif isinstance(x, Mapping):
            return type(x)({k: transfer(v) for k, v in x.items()})
        else:
            return x

    return {k: transfer(v) for k, v in input.items()}
