"""
Dataset loading for the WasteViT baseline pipeline.

Works with any torchvision ImageFolder-style dataset:
    <root>/<class_a>/*.jpg
    <root>/<class_b>/*.jpg
    ...
or an already-split dataset:
    <root>/train/<class_a>/*.jpg ...
    <root>/val/<class_a>/*.jpg ...
    <root>/test/<class_a>/*.jpg ...

Swap datasets by changing only config.yaml's data.root — no code changes needed.
"""
import os
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _has_split_folders(root: str) -> bool:
    return all(os.path.isdir(os.path.join(root, split)) for split in ("train", "val", "test"))


def build_transforms(image_size: int):
    # Standard ImageNet-style normalization constants. Fine to use even when
    # NOT using pretrained weights — they're just a sane, fixed normalization,
    # not transfer learning.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


def _split_indices(n: int, val_frac: float, test_frac: float, seed: int):
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
    return train_idx, val_idx, test_idx


def get_dataloaders(cfg: dict):
    data_cfg = cfg["data"]
    root = data_cfg["root"]
    image_size = data_cfg["image_size"]
    batch_size = data_cfg["batch_size"]
    num_workers = data_cfg["num_workers"]
    seed = data_cfg["seed"]

    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"Dataset root '{root}' does not exist. Download a dataset "
            f"(see README.md) and point config.yaml's data.root at it."
        )

    train_tf, eval_tf = build_transforms(image_size)

    if _has_split_folders(root):
        train_ds = datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
        val_ds = datasets.ImageFolder(os.path.join(root, "val"), transform=eval_tf)
        test_ds = datasets.ImageFolder(os.path.join(root, "test"), transform=eval_tf)
        class_names = train_ds.classes
    else:
        # Single folder of class subfolders -> split ourselves, reproducibly.
        full_train = datasets.ImageFolder(root, transform=train_tf)
        full_eval = datasets.ImageFolder(root, transform=eval_tf)
        class_names = full_train.classes

        train_idx, val_idx, test_idx = _split_indices(
            len(full_train), data_cfg["val_split"], data_cfg["test_split"], seed
        )
        train_ds = Subset(full_train, train_idx)
        val_ds = Subset(full_eval, val_idx)
        test_ds = Subset(full_eval, test_idx)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=g, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    print(f"[data] classes ({len(class_names)}): {class_names}")
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    return train_loader, val_loader, test_loader, class_names
