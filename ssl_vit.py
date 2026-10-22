"""
ssl_vit.py - two small experiments with self-supervised Vision Transformers.

  features : TASK 1. Turn every image into a feature vector with a pretrained ViT (no training),
             then train a tiny classifier on 1, 5, 10, 20, 50, 100 labelled images per class.
             Output: accuracy-vs-number-of-labels curve (logistic regression and kNN).
  mae      : TASK 2. Keep training a pretrained MAE (masked autoencoder) ViT on your images
             WITHOUT labels (self-supervised adaptation). Saves the adapted model.

Typical flow (Kaggle notebook cells, or the THWS VM):

  # 0) quick smoke test, 1 minute
  python ssl_vit.py features --data DATA --model facebook/dinov2-small --limit 200 --out res_smoke

  # 1) Task 1 baselines
  python ssl_vit.py features --data DATA --model facebook/dinov2-small --out res_dinov2
  python ssl_vit.py features --data DATA --model facebook/vit-mae-base  --out res_mae_before

  # 2) Task 2: adapt MAE on unlabelled images
  python ssl_vit.py mae --data DATA --epochs 20 --out mae_adapted

  # 3) Task 1 again with the adapted model, compare with res_mae_before
  python ssl_vit.py features --data DATA --model mae_adapted --out res_mae_after

DATA is a folder with one sub-folder per class (TrashNet: cardboard, glass, metal, ...).
The script searches below DATA for such a folder, so the exact path does not matter.
Labels are used ONLY to evaluate the features. MAE training never sees them.
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet statistics (used by all these models)


# ---------------------------------------------------------------- data
def find_class_root(root, max_depth=5):
    """Return the folder whose sub-folders are the classes (>=2 sub-folders that contain images)."""
    root = Path(root)

    def has_images(d):
        return any(p.suffix.lower() in IMG_EXT for p in d.iterdir() if p.is_file())

    def search(d, depth):
        subs = sorted(p for p in d.iterdir() if p.is_dir())
        if sum(has_images(s) for s in subs) >= 2:
            return d
        if depth < max_depth:
            for s in subs:
                found = search(s, depth + 1)
                if found:
                    return found
        return None

    found = search(root, 0)
    if found is None:
        sys.exit(f"No folder with >=2 class sub-folders of images found below {root}")
    return found


def list_images(class_root):
    class_root = Path(class_root)
    classes = sorted(p.name for p in class_root.iterdir()
                     if p.is_dir() and any(q.suffix.lower() in IMG_EXT for q in p.iterdir() if q.is_file()))
    paths, labels = [], []
    for i, c in enumerate(classes):
        for p in sorted((class_root / c).iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                paths.append(str(p))
                labels.append(i)
    return paths, np.array(labels), classes


def stratified_split(labels, test_frac=0.2, seed=0):
    """Fixed split. Returns (train_idx, test_idx). Labels of the train part are only used to sample few-shot sets."""
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * test_frac)))
        te += list(idx[:k])
        tr += list(idx[k:])
    return np.array(sorted(tr)), np.array(sorted(te))


# ---------------------------------------------------------------- evaluation (numpy + sklearn only)
def label_budget_curve(X, y, train_idx, test_idx, ns, repeats=5, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(seed)
    by_class = {c: train_idx[y[train_idx] == c] for c in np.unique(y)}
    rows = []
    for n in ns:
        if any(len(v) < n for v in by_class.values()):
            continue
        lr_acc, knn_acc = [], []
        for _ in range(repeats):
            idx = np.concatenate([rng.choice(v, n, replace=False) for v in by_class.values()])
            lr = LogisticRegression(max_iter=3000).fit(X[idx], y[idx])
            lr_acc.append((lr.predict(X[test_idx]) == y[test_idx]).mean())
            knn = KNeighborsClassifier(n_neighbors=min(5, n), metric="cosine").fit(X[idx], y[idx])
            knn_acc.append((knn.predict(X[test_idx]) == y[test_idx]).mean())
        rows.append({"labels_per_class": n,
                     "logreg_acc": float(np.mean(lr_acc)), "logreg_std": float(np.std(lr_acc)),
                     "knn_acc": float(np.mean(knn_acc)), "knn_std": float(np.std(knn_acc))})
    return rows


# ---------------------------------------------------------------- torch helpers (imported lazily)
def get_device(name=None):
    import torch
    return name or ("cuda" if torch.cuda.is_available() else "cpu")


def make_dataset(paths, train_aug, size=224):
    import torch
    from PIL import Image
    from torchvision import transforms as T

    if train_aug:
        tf = T.Compose([T.RandomResizedCrop(size, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
                        T.ToTensor(), T.Normalize(MEAN, STD)])
    else:
        tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            return tf(Image.open(paths[i]).convert("RGB"))

    return DS()


def load_encoder(name, device):
    """Pretrained ViT encoder. For MAE checkpoints, masking is switched off for feature extraction."""
    import torch
    from transformers import AutoConfig, AutoModel, ViTMAEModel

    cfg = AutoConfig.from_pretrained(name)
    if cfg.model_type == "vit_mae":
        model = ViTMAEModel.from_pretrained(name, mask_ratio=0.0)
        kind = "mae"
    else:
        model = AutoModel.from_pretrained(name)
        kind = "other"
    return model.to(device).eval(), kind


def extract_features(paths, model_name, device, batch_size=64):
    import torch

    model, kind = load_encoder(model_name, device)
    loader = torch.utils.data.DataLoader(make_dataset(paths, False), batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    feats = []
    with torch.no_grad():
        for x in loader:
            out = model(pixel_values=x.to(device))
            if kind == "mae":
                f = out.last_hidden_state.mean(1)          # average of all patch tokens
            else:
                f = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
            f = torch.nn.functional.normalize(f.float(), dim=-1)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats)


# ---------------------------------------------------------------- modes
def cmd_features(a):
    class_root = find_class_root(a.data)
    paths, y, classes = list_images(class_root)
    if a.limit:  # smoke test: keep a random subset, every class
        rng = random.Random(0)
        keep = sorted(rng.sample(range(len(paths)), min(a.limit, len(paths))))
        paths, y = [paths[i] for i in keep], y[keep]
    train_idx, test_idx = stratified_split(y)
    print(f"data: {class_root} | {len(paths)} images | classes: {classes}")
    print(f"train {len(train_idx)} / test {len(test_idx)} | model: {a.model}")

    device = get_device(a.device)
    X = extract_features(paths, a.model, device)
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "features.npy", X)

    ns = [n for n in a.ns]
    rows = label_budget_curve(X, y, train_idx, test_idx, ns)
    with (a.out / "label_curve.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["labels_per_class"])
        wr.writeheader()
        wr.writerows(rows)
    lines = [f"model {a.model} | {len(paths)} images | test {len(test_idx)}",
             "labels/class | logreg acc | kNN acc"]
    for r in rows:
        lines.append(f"{r['labels_per_class']:>12} | {r['logreg_acc']:.3f}±{r['logreg_std']:.3f} | "
                     f"{r['knn_acc']:.3f}±{r['knn_std']:.3f}")
    text = "\n".join(lines)
    (a.out / "summary.txt").write_text(text)
    print(text)


def cmd_mae(a):
    import torch
    from transformers import ViTMAEForPreTraining

    class_root = find_class_root(a.data)
    paths, y, classes = list_images(class_root)
    train_idx, _ = stratified_split(y)             # same split as Task 1: train images only, NO labels used
    paths = [paths[i] for i in train_idx]
    if a.limit:
        paths = paths[:a.limit]
    print(f"MAE adaptation on {len(paths)} unlabelled images from {class_root}")

    device = get_device(a.device)
    model = ViTMAEForPreTraining.from_pretrained(a.model).to(device).train()
    loader = torch.utils.data.DataLoader(make_dataset(paths, True), batch_size=a.batch_size,
                                         shuffle=True, num_workers=2, drop_last=len(paths) > a.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    first = None
    for ep in range(1, a.epochs + 1):
        total, n = 0.0, 0
        for x in loader:
            x = x.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(pixel_values=x).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(x)
            n += len(x)
        avg = total / n
        first = avg if first is None else first
        print(f"epoch {ep}/{a.epochs}  reconstruction loss {avg:.4f}  (epoch 1: {first:.4f})", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    print(f"saved adapted model to {a.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("features")
    f.add_argument("--data", required=True)
    f.add_argument("--model", default="facebook/dinov2-small")
    f.add_argument("--out", type=Path, default=Path("res"))
    f.add_argument("--ns", type=int, nargs="+", default=[1, 5, 10, 20, 50, 100])
    f.add_argument("--limit", type=int, default=0, help="use only N random images (smoke test)")
    f.add_argument("--device", default=None)
    f.set_defaults(fn=cmd_features)

    m = sub.add_parser("mae")
    m.add_argument("--data", required=True)
    m.add_argument("--model", default="facebook/vit-mae-base")
    m.add_argument("--out", type=Path, default=Path("mae_adapted"))
    m.add_argument("--epochs", type=int, default=20)
    m.add_argument("--batch-size", type=int, default=64)
    m.add_argument("--lr", type=float, default=5e-5)
    m.add_argument("--limit", type=int, default=0, help="use only N images (smoke test)")
    m.add_argument("--device", default=None)
    m.set_defaults(fn=cmd_mae)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
