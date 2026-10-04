"""
Model builder for the WasteViT baseline pipeline.

IMPORTANT: this is the "no transfer learning, no self-supervised" baseline
task — the model is built with pretrained=False, so every weight starts
random and is learned purely from the dataset you point config.yaml at.
"""
import torch.nn as nn
from torchvision import models


def build_model(architecture: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if pretrained:
        raise ValueError(
            "This baseline pipeline must NOT use pretrained weights "
            "(no transfer learning yet). Set model.pretrained: false in config.yaml."
        )

    if not hasattr(models, architecture):
        raise ValueError(
            f"Unknown torchvision architecture '{architecture}'. "
            f"Try resnet18, resnet34, resnet50, mobilenet_v2, efficientnet_b0, convnext_tiny, ..."
        )

    model_fn = getattr(models, architecture)
    # weights=None guarantees random initialization (no transfer learning).
    model = model_fn(weights=None)

    # Replace the final classification layer to match our number of classes.
    # Different torchvision architectures name their head differently.
    if hasattr(model, "fc"):  # resnet family
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier"):  # mobilenet, efficientnet, convnext, vgg
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential):
            last_linear_idx = None
            for i, layer in enumerate(classifier):
                if isinstance(layer, nn.Linear):
                    last_linear_idx = i
            if last_linear_idx is None:
                raise ValueError(f"Could not find a Linear layer in classifier of {architecture}")
            in_features = classifier[last_linear_idx].in_features
            classifier[last_linear_idx] = nn.Linear(in_features, num_classes)
        elif isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, num_classes)
        else:
            raise ValueError(f"Unsupported classifier type for {architecture}: {type(classifier)}")
    elif hasattr(model, "heads"):  # vision transformer (vit_*) — kept for completeness
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Don't know how to adapt the head of architecture '{architecture}'")

    return model
