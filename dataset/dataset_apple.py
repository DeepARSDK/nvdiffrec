import os
import glob

import torch
import numpy as np
import pandas as pd

from render import util

from .dataset import Dataset


def _load_mask(fn):
    img = torch.tensor(util.load_image(fn), dtype=torch.float32)
    if len(img.shape) == 2:
        img = img[..., None].repeat(1, 1, 3)
    return img


def _load_img(fn):
    img = util.load_image_raw(fn)
    if img.dtype != np.float32:  # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img


class DatasetApple(Dataset):
    def __init__(self, base_dir, FLAGS, examples=None):
        self.FLAGS = FLAGS
        self.base_dir = base_dir
        self.examples = examples

        # Enumerate all image files and get resolution
        all_img = [
            f
            for f in sorted(glob.glob(os.path.join(self.base_dir, "images", "*")))
            if f.lower().endswith("png") or f.lower().endswith("jpg") or f.lower().endswith("jpeg")
        ]
        self.resolution = _load_img(all_img[0]).shape[0:2]

        data_frame = pd.read_csv(os.path.join(self.base_dir, "camera_positions.csv"))
        file_names = data_frame["FileName"].values
        translations = data_frame[["TranslationX", "TranslationY", "TranslationZ"]].values
        rotations = data_frame[
            [
                "Rotation00", "Rotation01", "Rotation02", "Rotation10", "Rotation11", "Rotation12", "Rotation20", "Rotation21" "Rotation22",
            ]
        ].values
        print(file_names, translations, rotations)
