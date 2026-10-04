# WasteViT — from-scratch baseline pipeline

What this is: a general, reusable training pipeline for a plain CNN trained
**from random weights** (no transfer learning, no self-supervised pretraining)
on an image classification dataset, with full MLflow experiment tracking.

This is the baseline every later step (DINOv2, LoRA, self-supervised
pretraining) gets compared against.

## 1. What's in here

```
wastevit-baseline/
├── config/
│   └── config.yaml       <- change THIS to swap dataset / model / hyperparameters
├── src/
│   ├── dataset.py         <- loads any ImageFolder-style dataset
│   ├── model.py            <- builds any torchvision CNN, pretrained=False
│   ├── metrics.py          <- accuracy, per-class precision/recall/F1, confusion matrix
│   └── train.py             <- the training loop + MLflow logging
├── data/                   <- put datasets here (not included, see below)
└── outputs/                 <- best model, metrics, confusion matrix get saved here
```

## 2. Install (once)

```bash
pip install torch torchvision mlflow scikit-learn matplotlib seaborn pyyaml
```

## 3. Get a dataset

The pipeline expects **ImageFolder** layout — one subfolder per class:

```
data/<dataset_name>/<class_a>/*.jpg
data/<dataset_name>/<class_b>/*.jpg
...
```

or already pre-split:

```
data/<dataset_name>/train/<class_a>/*.jpg ...
data/<dataset_name>/val/<class_a>/*.jpg ...
data/<dataset_name>/test/<class_a>/*.jpg ...
```

If you give it an unsplit folder, the code splits it for you automatically
(reproducibly, via the `seed` in config.yaml) using `val_split` / `test_split`.

### Recommended first dataset: TrashNet

Easiest public waste dataset — already in ImageFolder format once unzipped.

**On Kaggle (no download needed, runs in Kaggle's own notebook):**
1. Create a new Kaggle Notebook.
2. Add data → search "TrashNet" (e.g. `feyzazkefe/trashnet`) → Add.
3. Upload this whole `wastevit-baseline/` folder (or just the `src/` and
   `config/` files) to the notebook's working directory.
4. Point `config.yaml`'s `data.root` at the Kaggle dataset path, e.g.
   `/kaggle/input/trashnet/dataset-resized`.
5. Run: `!python src/train.py --config config/config.yaml`

**On Google Colab:**
1. Upload `wastevit-baseline/` to Colab (or clone/upload via GitHub).
2. Download TrashNet with the Kaggle API (upload your `kaggle.json`):
   ```python
   !pip install kaggle
   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/
   !kaggle datasets download -d feyzazkefe/trashnet -p data/ --unzip
   ```
3. Set `data.root: "data/dataset-resized"` (check the actual unzipped folder
   name — TrashNet's zip layout varies slightly between Kaggle uploads).
4. `!python src/train.py --config config/config.yaml`

### Other datasets from the team's list

Any of the Kaggle waste-classification sets your team collected work the same
way — download, unzip, point `data.root` at the folder, run. No code changes.

## 4. Run training

```bash
python src/train.py --config config/config.yaml
```

This will:
- load the dataset (splitting it automatically if needed)
- build the chosen architecture with **random weights** (`pretrained: false`)
- train with early stopping on validation accuracy
- evaluate the best checkpoint on the held-out test set
- save `outputs/model_best.pt`, `outputs/test_metrics.json`,
  `outputs/confusion_matrix.png`, `outputs/history.json`
- log everything to MLflow (params, per-epoch metrics, final metrics,
  per-class precision/recall/F1, and all the above files as artifacts)

## 5. View MLflow results

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the printed localhost URL. You'll see every run, with its config,
metrics curves, and attached artifacts (confusion matrix image included).

On Kaggle/Colab (no UI access), just read `outputs/test_metrics.json`
directly, or query programmatically:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
runs = mlflow.search_runs(experiment_names=["wastevit-baseline-from-scratch"])
print(runs[["run_id", "metrics.test_accuracy", "metrics.test_macro_f1"]])
```

## 6. Switching dataset or model — no code changes

Just edit `config/config.yaml`:

```yaml
data:
  root: "data/some_other_dataset"   # <- change this
model:
  architecture: "resnet34"           # <- or this (any torchvision CNN name)
  pretrained: false                   # <- MUST stay false for this baseline task
```

Run again. Same `train.py`, same metrics, same MLflow logging — this is the
"general pipeline that works on any given dataset" the team asked for.

## 7. What this baseline is for

This is **not** meant to perform well — a CNN trained from scratch on a small
labeled dataset is expected to underperform. That's the point: it is the
honest, no-shortcuts reference point. Later comparisons (frozen DINOv2 +
linear probe, LoRA fine-tuning, self-supervised pretraining on unlabelled
belt images) all get compared against these numbers to show whether they
actually help — exactly what Prof. Frank's "baseline + honest comparison"
requirement calls for.
