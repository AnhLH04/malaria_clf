"""
dataset.py - Malaria Cell Classification Dataset
Supports annotation file format: relative_path label_idx
"""

import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = {
    0: "TJ",        # Trophozoite (Mature)
    1: "TA",        # Trophozoite (Young)
    2: "S",         # Schizont
    3: "G",         # Gametocyte
    4: "Unparasitized",
}

# Adjust if your label mapping differs
NUM_CLASSES = 5
PARASITE_CLASSES = [0, 1, 2, 3]   # TJ, TA, S, G
MAJORITY_CLASS   = 4               # Unparasitized


def get_transforms(split="train", img_size=224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomCrop(img_size),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            T.RandomRotation(30),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    else:  # val / test
        return T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])


class TwoViewTransform:
    """Returns two augmented views of same image for contrastive learning."""
    def __init__(self, base_transform):
        self.transform = base_transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class MalariaDataset(Dataset):
    """
    Reads annotation txt file:
        relative/path/to/cell.jpg  label_idx
    base_dir is prepended to each relative path.
    """
    def __init__(self, annotation_file, base_dir, transform=None, two_view=False):
        self.base_dir  = base_dir
        self.two_view  = two_view
        self.samples   = []

        with open(annotation_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rel_path = parts[0]
                label    = int(parts[1])
                abs_path = os.path.join(base_dir, rel_path)
                self.samples.append((abs_path, label))

        if two_view:
            train_tf = get_transforms("train")
            self.transform = TwoViewTransform(train_tf)
        else:
            self.transform = transform

        print(f"[Dataset] Loaded {len(self.samples)} samples from {annotation_file}")
        self._print_dist()

    def _print_dist(self):
        from collections import Counter
        counts = Counter(lbl for _, lbl in self.samples)
        for cls_idx in sorted(counts):
            print(f"  Class {cls_idx} ({CLASS_NAMES.get(cls_idx, '?')}): {counts[cls_idx]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Cannot open image {path}: {e}")

        if self.transform:
            img = self.transform(img)

        return img, label
