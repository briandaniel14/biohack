"""PyTorch datasets for filament segmentation."""

import os
import pickle

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

from biohack.preprocess import preprocess, preprocess_mask


class SyntheticDataset(Dataset):
    def __init__(self, root: str = "synthetic_data", cache: bool = True):
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.files = sorted(os.listdir(self.img_dir))
        self._imgs: np.ndarray | None = None
        self._masks: np.ndarray | None = None
        if cache:
            self._load_cache()

    def _load_cache(self):
        print(f"  Caching {len(self.files)} samples to RAM...", flush=True)
        imgs, masks = [], []
        for fname in self.files:
            imgs.append(preprocess(tifffile.imread(os.path.join(self.img_dir, fname))))
            masks.append(preprocess_mask(tifffile.imread(os.path.join(self.mask_dir, fname))))
        self._imgs = np.stack(imgs).astype(np.float32)
        self._masks = np.stack(masks).astype(np.float32)
        print(f"  Cached. Images: {self._imgs.nbytes / 1e9:.2f} GB", flush=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self._imgs is not None:
            img = torch.from_numpy(self._imgs[idx]).unsqueeze(0)
            mask = torch.from_numpy(self._masks[idx]).unsqueeze(0)
            return img, mask

        fname = self.files[idx]
        img = preprocess(tifffile.imread(os.path.join(self.img_dir, fname)))
        mask = preprocess_mask(tifffile.imread(os.path.join(self.mask_dir, fname))).astype(
            np.float32
        )
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
