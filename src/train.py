"""
WasteViT baseline training pipeline — no transfer learning, no self-supervised
pretraining. A plain torchvision CNN, trained from random weights, with full
MLflow experiment tracking.

Usage:
    python src/train.py --config config/config.yaml

Swap datasets or architectures by editing config/config.yaml only.
"""
import argparse
import json
import os
import random
import time

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from metrics import compute_metrics, plot_confusion_matrix, run_inference
from model import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, train_cfg):
    if train_cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
        )
    elif train_cfg["optimizer"] == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=train_cfg["lr"], momentum=0.9,
            weight_decay=train_cfg["weight_decay"],
        )
    raise ValueError(f"Unknown optimizer: {train_cfg['optimizer']}")


def build_scheduler(optimizer, train_cfg):
    if train_cfg["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    return None


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += images.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate_loss_acc(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += images.size(0)
    return total_loss / n, correct / n


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[setup] device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(cfg)
    num_classes = len(class_names)

    model = build_model(
        cfg["model"]["architecture"], num_classes, cfg["model"]["pretrained"]
    ).to(device)

    optimizer = build_optimizer(model, cfg["train"])
    scheduler = build_scheduler(optimizer, cfg["train"])
    criterion = nn.CrossEntropyLoss()

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    run_name = cfg["mlflow"]["run_name"] or f"{cfg['model']['architecture']}-{os.path.basename(cfg['data']['root'])}"

    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    with mlflow.start_run(run_name=run_name):
        # --- log config/params ---
        mlflow.log_params({
            "architecture": cfg["model"]["architecture"],
            "pretrained": cfg["model"]["pretrained"],
            "dataset_root": cfg["data"]["root"],
            "image_size": cfg["data"]["image_size"],
            "batch_size": cfg["data"]["batch_size"],
            "epochs": cfg["train"]["epochs"],
            "lr": cfg["train"]["lr"],
            "weight_decay": cfg["train"]["weight_decay"],
            "optimizer": cfg["train"]["optimizer"],
            "scheduler": cfg["train"]["scheduler"],
            "seed": cfg["train"]["seed"],
            "num_classes": num_classes,
            "classes": ",".join(class_names),
        })
        mlflow.log_artifact(config_path)

        best_val_acc = -1.0
        best_state = None
        epochs_without_improvement = 0
        history = []

        print(f"[train] starting {cfg['train']['epochs']} epochs")
        t0 = time.time()
        for epoch in range(1, cfg["train"]["epochs"] + 1):
            ep_t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)
            if scheduler is not None:
                scheduler.step()
            ep_time = time.time() - ep_t0

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            history.append({
                "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "time_sec": ep_time,
            })

            print(
                f"[epoch {epoch:02d}/{cfg['train']['epochs']}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({ep_time:.1f}s)"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg["train"]["early_stopping_patience"]:
                    print(f"[train] early stopping at epoch {epoch} (no val improvement for "
                          f"{cfg['train']['early_stopping_patience']} epochs)")
                    break

        total_time = time.time() - t0
        mlflow.log_metric("total_train_time_sec", total_time)
        mlflow.log_metric("best_val_acc", best_val_acc)

        history_path = os.path.join(out_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        mlflow.log_artifact(history_path)

        # --- restore best checkpoint and run final test evaluation ---
        if best_state is not None:
            model.load_state_dict(best_state)

        model_path = os.path.join(out_dir, "model_best.pt")
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)

        y_true, y_pred = run_inference(model, test_loader, device)
        metrics = compute_metrics(y_true, y_pred, class_names)

        mlflow.log_metrics({
            "test_accuracy": metrics["accuracy"],
            "test_macro_precision": metrics["macro_precision"],
            "test_macro_recall": metrics["macro_recall"],
            "test_macro_f1": metrics["macro_f1"],
        })
        for cname, m in metrics["per_class"].items():
            safe = cname.replace(" ", "_")
            mlflow.log_metrics({
                f"test_precision_{safe}": m["precision"],
                f"test_recall_{safe}": m["recall"],
                f"test_f1_{safe}": m["f1"],
            })

        metrics_path = os.path.join(out_dir, "test_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)

        cm_path = os.path.join(out_dir, "confusion_matrix.png")
        plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
        mlflow.log_artifact(cm_path)

        print(f"\n[done] total train time: {total_time:.1f}s")
        print(f"[done] best val acc: {best_val_acc:.4f}")
        print(f"[done] test accuracy: {metrics['accuracy']:.4f}")
        print(f"[done] test macro F1: {metrics['macro_f1']:.4f}")
        print(f"[done] per-class results:")
        for cname, m in metrics["per_class"].items():
            print(f"         {cname:20s} precision={m['precision']:.3f} recall={m['recall']:.3f} "
                  f"f1={m['f1']:.3f} support={m['support']}")
        print(f"[done] artifacts logged to MLflow run, and saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
