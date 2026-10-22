# TrashNet test of self-supervised ViT features (Kaggle, 22 Oct 2026)

Purpose: check the pipeline (`ssl_vit.py`) on public data before running on the WeSort crops on the THWS VM.
These numbers are NOT WeSort results.

## Setup
- Data: TrashNet (dataset-resized), 2,527 images, 6 classes (cardboard, glass, metal, paper, plastic, trash)
- Fixed stratified split: 2,022 train / 505 test
- Features: frozen ViT encoder, then a small classifier on N labelled images per class
  (N = 1, 5, 10, 20, 50, 100; 5 random draws per N; mean accuracy reported)
- Classifiers: logistic regression (logreg) and kNN (cosine, k = min(5, N))
- Labels are used only for the small classifier and for testing. MAE adaptation used the 2,022 training images WITHOUT labels.
- Hardware: Kaggle GPU (Tesla T4). Chance level with 6 classes = 16.7%

## Task 1: label-efficiency curve (accuracy on 505 test images)

| Labels per class | DINOv2-small logreg | DINOv2-small kNN | MAE-base logreg | MAE-base kNN |
|---|---|---|---|---|
| 1   | 51.2% | 49.6% | 28.8% | 26.8% |
| 5   | 70.8% | 61.0% | 37.3% | 33.2% |
| 10  | 79.6% | 69.0% | 38.3% | 38.5% |
| 20  | 80.6% | 76.4% | 43.4% | 44.9% |
| 50  | 85.8% | 82.0% | 48.6% | 56.0% |
| 100 | 88.6% | 85.3% | 52.0% | 62.2% |

(MAE-base here = pretrained facebook/vit-mae-base before any adaptation.)

## Task 2: unsupervised adaptation of MAE-base (20 epochs, lr 5e-5, no labels)
Reconstruction loss: 0.0971 (epoch 1) -> 0.0948 (epoch 20), about -2%.

| Labels per class | MAE before (logreg) | MAE after (logreg) | MAE before (kNN) | MAE after (kNN) |
|---|---|---|---|---|
| 1   | 28.8% | 29.1% | 26.8% | 27.4% |
| 5   | 37.3% | 37.5% | 33.2% | 34.4% |
| 10  | 38.3% | 39.5% | 38.5% | 40.4% |
| 20  | 43.4% | 44.8% | 44.9% | 46.2% |
| 50  | 48.6% | 49.6% | 56.0% | 57.0% |
| 100 | 52.0% | 54.7% | 62.2% | 62.7% |

## Second adaptation run (100 epochs, lr 2e-4)
Loss 0.1210 (epoch 1) -> about 0.099 (epoch 100). The high learning rate first disturbed the pretrained weights,
and 100 epochs only recovered to about the starting level. Model and results were lost when the Kaggle session
restarted, so Task 1 was NOT run on it. No conclusion drawn.

## Reading
- DINOv2-small gives a strong curve with few labels (88.6% with 100 labels per class, 600 labels in total).
- Unadapted MAE-base is much weaker with a simple classifier (expected: MAE features usually need fine-tuning).
- Gentle unsupervised adaptation of MAE gave a small gain in all 12 comparisons (+0.2 to +2.7 points).
  One run, one seed, and the model changed little, so treat it as tentative.
- Test set is 505 images, so differences of 1-2 points are within noise.

## Open points / next
- Repeat on WeSort crops on the THWS VM (DINOv2 as main model, many unlabelled crops for adaptation).
- Adaptation with warm-up + cosine learning-rate schedule instead of a fixed lr.
- Save outputs: Kaggle /kaggle/working is erased when the session ends.
