import cv2
import torch
import torch.nn as nn

from .pytorch_ssim import SSIM


class _MSELoss(nn.Module):
    def __init__(self, outstride, weight, reduction="mean"):
        super().__init__()
        self.criterion_mse = nn.MSELoss(reduction=reduction)
        self.resize = nn.UpsamplingBilinear2d(scale_factor=1.0 / outstride)
        self.outstride = outstride
        self.weight = weight

    def forward(self, input):
        density_pred = input["density_pred"]
        density_gt = input["density"]
        if not self.outstride == 1:
            cnt_gt = density_gt.sum()
            density_gt = self.resize(density_gt)
            cnt_cur = density_gt.sum()
            density_gt = density_gt / cnt_cur * cnt_gt
        return self.criterion_mse(density_pred, density_gt)


class MaskMSELoss(nn.Module):
    def __init__(self, maskpath, input_size, weight, reduction="mean"):
        super().__init__()
        # mask
        mask = cv2.imread(maskpath, 0)
        mask = cv2.resize(mask, (input_size[1], input_size[0]), cv2.INTER_NEAREST)
        mask[mask != 0] = 1
        self.mask = torch.tensor(mask).cuda()
        self.criterion_mse = nn.MSELoss(reduction=reduction)
        self.weight = weight

    def forward(self, input):
        input["density_pred"] = input["density_pred"] * self.mask
        input["density"] = input["density"] * self.mask
        density_pred = input["density_pred"]
        density_gt = input["density"]
        return self.criterion_mse(density_pred, density_gt)


class SSIMLoss(nn.Module):
    def __init__(self, window_size, outstride, weight):
        super().__init__()
        self.criterion_ssim = SSIM(window_size)
        self.outstride = outstride
        self.weight = weight

    def forward(self, input):
        density_pred = input["density_pred"]
        density_gt = input["density"]
        if not self.outstride == 1:
            cnt_gt = density_gt.sum()
            density_gt = self.resize(density_gt)
            cnt_cur = density_gt.sum()
            density_gt = density_gt / cnt_cur * cnt_gt
        return 1 - self.criterion_ssim(density_pred, density_gt)


def build_criterion(config):
    loss_dict = {}
    for i in range(len(config)):
        cfg = config[i]
        loss_name = cfg["name"]
        loss_dict[loss_name] = globals()[cfg["type"]](**cfg["kwargs"])
    return loss_dict
