import glob
import os

import numpy as np
import torch


def dump(save_dir, outputs):
    filenames = outputs["filename"]
    batch_size = len(filenames)
    density = outputs["density"]  # b x 1 x h x w
    density_pred = outputs["density_pred"]  # b x 1 x h x w
    gt_cnt = torch.sum(density, dim=(2, 3)).cpu().numpy()  # b x 1
    pred_cnt = torch.sum(density_pred, dim=(2, 3)).cpu().numpy()  # b x 1
    for i in range(batch_size):
        file_dir, filename = os.path.split(filenames[i])
        filename, _ = os.path.splitext(filename)
        save_file = os.path.join(save_dir, filename + ".npz")
        np.savez(
            save_file, filename=filenames[i], gt_cnt=gt_cnt[i], pred_cnt=pred_cnt[i]
        )


def merge_together(save_dir):
    npz_file_list = glob.glob(os.path.join(save_dir, "*.npz"))
    filenames = []
    gt_cnts = []
    pred_cnts = []
    for npz_file in npz_file_list:
        npz = np.load(npz_file)
        filenames.append(str(npz["filename"]))
        gt_cnts.append(npz["gt_cnt"])
        pred_cnts.append(npz["pred_cnt"])
    gt_cnts = np.concatenate(np.asarray(gt_cnts), axis=0)
    pred_cnts = np.concatenate(np.asarray(pred_cnts), axis=0)
    return gt_cnts, pred_cnts


def performances(gt_cnts, pred_cnts):
    val_mae = np.mean(np.abs(pred_cnts - gt_cnts))
    val_rmse = np.sqrt(np.mean((pred_cnts - gt_cnts) ** 2))
    return val_mae, val_rmse
