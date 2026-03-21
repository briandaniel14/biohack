"""PyTorch datasets for filament segmentation."""

import os
import pickle

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from biohack.preprocess import preprocess, preprocess_mask


class SyntheticDataset(Dataset):
    def __init__(self, root: str = "synthetic_data"):
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.files = sorted(os.listdir(self.img_dir))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img = tifffile.imread(os.path.join(self.img_dir, fname))
        mask = tifffile.imread(os.path.join(self.mask_dir, fname))

        img = preprocess(img)  # float32, z-scored
        mask = preprocess_mask(mask).astype(np.float32)

        # (1, H, W)
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)


class RealDataset(Dataset):
    def __init__(self, pkl_path: str = "data/annotated_data.pkl"):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = preprocess(item["image"]).astype(np.float32)
        mask = preprocess_mask(item["mask"]).astype(np.float32)
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)
