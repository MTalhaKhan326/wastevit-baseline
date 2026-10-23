"""
finetune_dino.py - TASK 3: supervised fine-tuning of a pretrained ViT (default DINOv2-small).

Compared with Task 1 (ssl_vit.py features), the model is NOT frozen here:
  Task 1 : frozen DINOv2 -> feature vectors -> tiny classifier (only the classifier learns)
  Task 3 : DINOv2 + a linear classification head, the WHOLE network learns from labelled images
           (backbone with a very small learning rate, new head with a larger one)

It uses the same data, the same fixed train/test split and the same "labels per class" budgets
as Task 1, so the numbers can be put in the same table.

  python finetune_dino.py --data DATA --ns 10 50 100 0 --seeds 0 1 2 --steps 300 --out ft_dinov2

--ns      labelled images per class used for training (0 = ALL training labels)
--seeds   every (budget, seed) pair is a fresh run: new random labelled subset, new head, backbone
          starts again from the pretrained weights. Result = mean +- std over seeds.
--steps   number of optimisation steps (batches). Same for every budget, so runs are comparable.

Only the 505 TEST images are used for the final accuracy, and only the LAST step is reported
(no picking of the best epoch on the test set).

Smoke test (about 1 minute):
  python finetune_dino.py --data DATA --ns 5 --seeds 0 --steps 20 --out ft_smoke
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np

from ssl_vit import find_class_root, list_images, stratified_split, make_dataset, get_device


def pick_labelled(y, train_idx, n, rng):
    """n labelled images per class from the training part (n = 0 -> all of them)."""
    if n == 0:
        return np.array(train_idx)
    out = []
    for c in np.unique(y):
        pool = train_idx[y[train_idx] == c]
        if len(pool) < n:
            return None
        out.append(rng.choice(pool, n, replace=False))
    return np.concatenate(out)


def build_model(name, n_classes, device):
    import torch
    from transformers import AutoModel

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(name)
            self.head = torch.nn.Linear(self.backbone.config.hidden_size, n_classes)

        def forward(self, x):
            out = self.backbone(pixel_values=x)
            f = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
            return self.head(f)

    return Net().to(device)


def lr_factor(step, total, warmup):
    """Linear warm-up, then cosine decay to 0."""
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * p))


def evaluate(model, paths, y, device, batch_size=64):
    import torch

    loader = torch.utils.data.DataLoader(make_dataset(paths, False), batch_size=batch_size,
                                         shuffle=False, num_workers=2)
    model.eval()
    pred = []
    with torch.no_grad():
        for x in loader:
            pred.append(model(x.to(device)).argmax(1).cpu().numpy())
    pred = np.concatenate(pred)
    acc = float((pred == y).mean())
    recalls = [float((pred[y == c] == c).mean()) for c in np.unique(y)]
    return acc, float(np.mean(recalls))


def run_one(a, paths, y, train_idx, test_idx, n, seed, n_classes, device):
    import torch

    rng = np.random.default_rng(seed)
    idx = pick_labelled(y, train_idx, n, rng)
    if idx is None:
        return None
    torch.manual_seed(seed)

    class DS(torch.utils.data.Dataset):
        def __init__(self):
            self.ds = make_dataset([paths[i] for i in idx], True)   # augmentation: random crop + flip

        def __len__(self):
            return len(idx)

        def __getitem__(self, i):
            return self.ds[i], int(y[idx[i]])

    ds = DS()
    sampler = torch.utils.data.RandomSampler(ds, replacement=True, num_samples=a.steps * a.batch_size,
                                             generator=torch.Generator().manual_seed(seed))
    loader = torch.utils.data.DataLoader(ds, batch_size=a.batch_size, sampler=sampler, num_workers=2,
                                         drop_last=True)

    model = build_model(a.model, n_classes, device)
    if a.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        groups = [{"params": model.head.parameters(), "lr": a.head_lr}]
    else:
        groups = [{"params": model.backbone.parameters(), "lr": a.lr},
                  {"params": model.head.parameters(), "lr": a.head_lr}]
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    base = [g["lr"] for g in opt.param_groups]
    warmup = max(1, int(0.1 * a.steps))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    model.train()
    losses = []
    for step, (x, t) in enumerate(loader):
        f = lr_factor(step, a.steps, warmup)
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * f
        x, t = x.to(device), t.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(model(x), t)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        if (step + 1) % max(1, a.steps // 5) == 0:
            print(f"    step {step + 1}/{a.steps}  train loss {np.mean(losses[-20:]):.4f}", flush=True)

    acc, bal = evaluate(model, [paths[i] for i in test_idx], y[test_idx], device)
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"labels_per_class": n, "n_labelled": len(idx), "seed": seed, "test_acc": acc,
            "test_bal_acc": bal, "final_train_loss": float(np.mean(losses[-20:]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--out", type=Path, default=Path("ft_res"))
    ap.add_argument("--ns", type=int, nargs="+", default=[10, 50, 100, 0], help="labels per class, 0 = all")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5, help="backbone learning rate (small on purpose)")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="learning rate of the new linear head")
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--freeze-backbone", action="store_true", help="sanity check: only the head learns")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    class_root = find_class_root(a.data)
    paths, y, classes = list_images(class_root)
    train_idx, test_idx = stratified_split(y)
    device = get_device(a.device)
    print(f"data: {class_root} | {len(paths)} images | classes: {classes}")
    print(f"train {len(train_idx)} / test {len(test_idx)} | model {a.model} | device {device}")
    print(f"steps {a.steps} x batch {a.batch_size} | backbone lr {a.lr} | head lr {a.head_lr} | "
          f"{'FROZEN backbone' if a.freeze_backbone else 'full fine-tuning'}")

    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in a.ns:
        for seed in a.seeds:
            print(f"== labels/class {n if n else 'ALL'} | seed {seed}", flush=True)
            r = run_one(a, paths, y, train_idx, test_idx, n, seed, len(classes), device)
            if r is None:
                print("   skipped: some class has fewer labelled training images than requested")
                break
            print(f"   test acc {r['test_acc']:.3f} | balanced {r['test_bal_acc']:.3f}", flush=True)
            rows.append(r)

    if rows:
        with (a.out / "finetune_runs.csv").open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

    lines = [f"model {a.model} | supervised fine-tuning | test {len(test_idx)} images",
             f"steps {a.steps}, batch {a.batch_size}, backbone lr {a.lr}, head lr {a.head_lr}, "
             f"seeds {a.seeds}" + (" | BACKBONE FROZEN" if a.freeze_backbone else ""),
             "labels/class | labelled | test acc (mean±std) | balanced acc"]
    for n in a.ns:
        rs = [r for r in rows if r["labels_per_class"] == n]
        if not rs:
            continue
        acc = np.array([r["test_acc"] for r in rs])
        bal = np.array([r["test_bal_acc"] for r in rs])
        name = "ALL" if n == 0 else str(n)
        lines.append(f"{name:>12} | {rs[0]['n_labelled']:>8} | {acc.mean():.3f}±{acc.std():.3f} | {bal.mean():.3f}")
    text = "\n".join(lines)
    (a.out / "summary.txt").write_text(text)
    print("\n" + text)


if __name__ == "__main__":
    main()
