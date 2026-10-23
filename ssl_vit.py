"""
ssl_vit.py - two small experiments with self-supervised Vision Transformers.

  features : TASK 1. Turn every image into a feature vector with a pretrained ViT (no training),
             then train a tiny classifier on 1, 5, 10, 20, 50, 100 labelled images per class.
             Output: accuracy-vs-number-of-labels curve (logistic regression and kNN).
  dino     : TASK 4. Same idea as 'mae' but for DINOv2: keep training a pretrained DINOv2 on your images
             WITHOUT labels (self-distillation, student/teacher). Saves the adapted teacher backbone.
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

  # 4) Task 4: adapt DINOv2 on unlabelled images, then test it with Task 1 and compare with res_dinov2
  python ssl_vit.py dino --data DATA --epochs 20 --out dino_adapted
  python ssl_vit.py features --data DATA --model dino_adapted --out res_dino_after

DATA is a folder with one sub-folder per class (TrashNet: cardboard, glass, metal, ...).
The script searches below DATA for such a folder, so the exact path does not matter.
Labels are used ONLY to evaluate the features. MAE training never sees them.
"""
import argparse
import copy
import csv
import math
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


# ---------------------------------------------------------------- views (multi-crop)
def make_views_dataset(paths, n_local, size_global=224, size_local=98, color=0.4):
    """Each item = tuple of tensors: (global1, global2, local1, ..., local_n). No label."""
    import torch
    from PIL import Image
    from torchvision import transforms as T

    bicubic = T.InterpolationMode.BICUBIC

    def base(size, scale, blur_p, blur_k):
        ops = [T.RandomResizedCrop(size, scale=scale, interpolation=bicubic), T.RandomHorizontalFlip()]
        if color > 0:
            ops += [T.RandomApply([T.ColorJitter(0.8 * color, 0.8 * color, 0.4 * color, 0.2 * color)], p=0.8),
                    T.RandomGrayscale(0.2)]
        ops += [T.RandomApply([T.GaussianBlur(blur_k, (0.1, 2.0))], p=blur_p), T.ToTensor(), T.Normalize(MEAN, STD)]
        return T.Compose(ops)

    g1 = base(size_global, (0.4, 1.0), 1.0, 23)
    g2 = base(size_global, (0.4, 1.0), 0.1, 23)
    loc = base(size_local, (0.05, 0.4), 0.5, 9)

    class DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            img = Image.open(paths[i]).convert("RGB")
            return (g1(img), g2(img)) + tuple(loc(img) for _ in range(n_local))

    return DS()


# ---------------------------------------------------------------- model
def build_dino(name, a, device):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel

    class Head(nn.Module):
        """MLP -> small bottleneck -> L2 norm -> normalised prototype layer (as in DINO)."""

        def __init__(self, in_dim, hidden, bottleneck, out_dim):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, bottleneck))
            self.last = nn.Parameter(torch.randn(out_dim, bottleneck) * 0.02)   # the "prototypes"

        def forward(self, x):
            x = F.normalize(self.mlp(x), dim=-1)
            return F.linear(x, F.normalize(self.last, dim=-1))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(name)
            d = self.backbone.config.hidden_size
            self.head = Head(d, a.head_hidden, a.bottleneck, a.out_dim)

        def features(self, x):
            out = self.backbone(pixel_values=x)
            return out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]

        def forward(self, x):
            return self.head(self.features(x))

    student = Net().to(device)
    if a.train_last_blocks > 0:                      # optional: train only the last K transformer blocks
        for p in student.backbone.embeddings.parameters():
            p.requires_grad = False
        for blk in student.backbone.encoder.layer[:-a.train_last_blocks]:
            for p in blk.parameters():
                p.requires_grad = False
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return student, teacher


def cosine(step, total, start, end):
    return end + 0.5 * (start - end) * (1 + math.cos(math.pi * min(step, total) / max(1, total)))


# ---------------------------------------------------------------- training
def cmd_dino(a):
    import torch
    import torch.nn.functional as F

    class_root = find_class_root(a.data)
    paths, y, classes = list_images(class_root)
    train_idx, _ = stratified_split(y)                 # same split as Task 1: train part only, NO labels used
    paths = [paths[i] for i in train_idx]
    if a.limit:                                        # smoke test: random subset (not one class only)
        paths = random.Random(0).sample(paths, min(a.limit, len(paths)))
    print(f"DINO adaptation on {len(paths)} unlabelled images from {class_root}")

    device = get_device(a.device)
    torch.manual_seed(a.seed)
    student, teacher = build_dino(a.model, a, device)
    loader = torch.utils.data.DataLoader(make_views_dataset(paths, a.n_local, color=a.color_strength),
                                         batch_size=a.batch_size, shuffle=True, num_workers=a.workers,
                                         drop_last=len(paths) > a.batch_size)
    steps_per_epoch = len(loader)
    total = a.epochs * steps_per_epoch
    warmup = max(1, int(a.warmup_frac * total))
    freeze_last = max(1, int(a.freeze_last_frac * total))

    bb = [p for p in student.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr}, {"params": student.head.parameters(), "lr": a.head_lr}],
                            weight_decay=a.weight_decay)
    base_lr = [g["lr"] for g in opt.param_groups]
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    center = torch.zeros(1, a.out_dim, device=device)
    print(f"{steps_per_epoch} steps/epoch, {total} steps | views per image: 2 global + {a.n_local} local | "
          f"backbone lr {a.lr}, head lr {a.head_lr}, teacher momentum {a.momentum}->1")

    step, first = 0, None
    for ep in range(1, a.epochs + 1):
        student.train()
        tot_loss, tot_std, n = 0.0, 0.0, 0
        for views in loader:
            views = [v.to(device, non_blocking=True) for v in views]
            f = (step + 1) / warmup if step < warmup else cosine(step - warmup, total - warmup, 1.0, 0.0)
            for g, b in zip(opt.param_groups, base_lr):
                g["lr"] = b * f
            mom = cosine(step, total, a.momentum, 1.0)
            B = views[0].shape[0]

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                with torch.no_grad():                                   # teacher: global views only
                    t_feat = teacher.features(torch.cat(views[:2]))
                    t_out = teacher.head(t_feat).float()
                s_glob = student(torch.cat(views[:2])).float()
                s_loc = student(torch.cat(views[2:])).float() if a.n_local > 0 else None

            t_prob = F.softmax((t_out - center) / a.teacher_temp, dim=-1).chunk(2)     # centering + sharpening
            s_logp_glob = F.log_softmax(s_glob / a.student_temp, dim=-1).chunk(2)
            s_logp_loc = F.log_softmax(s_loc / a.student_temp, dim=-1).chunk(a.n_local) if s_loc is not None else []

            loss, terms = 0.0, 0
            for iq, q in enumerate(t_prob):                               # teacher view iq ...
                for iv, lp in enumerate(list(s_logp_glob) + list(s_logp_loc)):   # ... vs every OTHER student view
                    if iv == iq:
                        continue
                    loss = loss + torch.sum(-q * lp, dim=-1).mean()
                    terms += 1
            loss = loss / terms

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if step < freeze_last:
                student.head.last.grad = None                             # keep prototypes fixed at the start
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 3.0)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                for ps, pt in zip(student.parameters(), teacher.parameters()):     # teacher = moving average
                    pt.mul_(mom).add_(ps.detach(), alpha=1 - mom)
                center = center * a.center_momentum + t_out.mean(0, keepdim=True) * (1 - a.center_momentum)
                # collapse monitor: spread of the teacher's image vectors across the batch (0 = all images equal)
                spread = F.normalize(t_feat.float(), dim=-1).std(0).mean().item()

            tot_loss += loss.item() * B
            tot_std += spread * B
            n += B
            step += 1
        avg = tot_loss / n
        first = avg if first is None else first
        print(f"epoch {ep}/{a.epochs}  DINO loss {avg:.4f}  (epoch 1: {first:.4f})  "
              f"feature spread {tot_std / n:.4f}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    teacher.backbone.save_pretrained(a.out)
    print(f"saved adapted TEACHER backbone to {a.out}")
    print("next: python ssl_vit.py features --data DATA --model", a.out, "--out res_dino_after")


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

    d = sub.add_parser("dino")
    d.add_argument("--data", required=True)
    d.add_argument("--model", default="facebook/dinov2-small")
    d.add_argument("--out", type=Path, default=Path("dino_adapted"))
    d.add_argument("--epochs", type=int, default=20)
    d.add_argument("--batch-size", type=int, default=32)
    d.add_argument("--lr", type=float, default=1e-5, help="backbone learning rate (small on purpose)")
    d.add_argument("--head-lr", type=float, default=1e-3, help="learning rate of the new projection head")
    d.add_argument("--weight-decay", type=float, default=0.04)
    d.add_argument("--warmup-frac", type=float, default=0.1, help="share of steps with learning-rate warm-up")
    d.add_argument("--freeze-last-frac", type=float, default=0.05, help="share of steps with frozen prototypes")
    d.add_argument("--momentum", type=float, default=0.996, help="teacher moving-average momentum at start")
    d.add_argument("--teacher-temp", type=float, default=0.04)
    d.add_argument("--student-temp", type=float, default=0.1)
    d.add_argument("--center-momentum", type=float, default=0.9)
    d.add_argument("--n-local", type=int, default=6, help="number of small local crops per image")
    d.add_argument("--out-dim", type=int, default=4096, help="number of prototypes")
    d.add_argument("--head-hidden", type=int, default=2048)
    d.add_argument("--bottleneck", type=int, default=256)
    d.add_argument("--train-last-blocks", type=int, default=0, help="train only the last K blocks (0 = all)")
    d.add_argument("--color-strength", type=float, default=0.4, help="colour jitter strength, 0 = off")
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--limit", type=int, default=0, help="use only N random images (smoke test)")
    d.add_argument("--device", default=None)
    d.set_defaults(fn=cmd_dino)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
