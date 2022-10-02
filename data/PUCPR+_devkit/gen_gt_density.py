import json
import os

import cv2
import numpy as np


def gen_gaussian2d(shape, sigma=1):
    h, w = [_ // 2 for _ in shape]
    y, x = np.ogrid[-h : h + 1, -w : w + 1]
    gaussian = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gaussian[gaussian < np.finfo(gaussian.dtype).eps * gaussian.max()] = 0
    return gaussian


def draw_gaussian(density, center, radius, k=1, delte=6, overlap="add"):
    diameter = 2 * radius + 1
    gaussian = gen_gaussian2d((diameter, diameter), sigma=diameter / delte)
    x, y = center
    height, width = density.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    if overlap == "max":
        masked_density = density[y - top : y + bottom, x - left : x + right]
        masked_gaussian = gaussian[
            radius - top : radius + bottom, radius - left : radius + right
        ]
        np.maximum(masked_density, masked_gaussian * k, out=masked_density)
    elif overlap == "add":
        density[y - top : y + bottom, x - left : x + right] += gaussian[
            radius - top : radius + bottom, radius - left : radius + right
        ]
    else:
        raise NotImplementedError


def _min_dis_global(points):
    """
    points: m x 2, m x [x, y]
    """
    dis_min = float("inf")
    for point in points:
        point = point[None, :]  # 2 -> 1 x 2
        dis = np.sqrt(np.sum((points - point) ** 2, axis=1))  # m x 2 -> m
        dis = sorted(dis)[1]
        if dis_min > dis:
            dis_min = dis
    return dis_min


def points2density(points, radius_backup=None):
    """
    points: m x 2, m x [x, y]
    """
    num_points = points.shape[0]
    density = np.zeros(image_size, dtype=np.float32)  # [h, w]
    if num_points == 0:
        return np.zeros(image_size, dtype=np.float32)
    elif num_points == 1:
        radius = radius_backup
    else:
        radius = min(int(_min_dis_global(points)), radius_backup)
    for point in points:
        draw_gaussian(density, point, radius, overlap="max")
    return density


if __name__ == "__main__":
    root_dir = "./Images/"
    gt_dir = "./gt_density_map/"
    os.makedirs(gt_dir, exist_ok=True)

    # read all data
    metas = []
    anno_files = ["train.json", "test.json"]
    for anno_file in anno_files:
        with open(anno_file, "r+") as fr:
            for line in fr:
                meta = json.loads(line)
                metas.append(meta)

    # create gt density map
    for meta in metas:
        filename = meta["filename"]
        filepath = os.path.join(root_dir, filename)
        image = cv2.imread(filepath)
        image_size = image.shape[0:2]  # [h, w]
        boxes = meta["boxes"]
        cnt_gt = len(boxes)

        points = []
        for box in boxes:
            yl, xl, yr, xr = box
            point = [(xl + xr) // 2, (yl + yr) // 2]
            points.append(point)
        points = np.array(points)
        radius_backup = (xr - xl + yr - yl) // 2

        density = points2density(points, radius_backup)

        if not cnt_gt == 0:
            cnt_cur = density.sum()
            density = density / cnt_cur * cnt_gt

        filename_ = os.path.splitext(filename)[0]
        save_path = os.path.join(gt_dir, filename_ + ".npy")
        np.save(save_path, density)
        print(f"Success: generate gt density map for {filename}")
