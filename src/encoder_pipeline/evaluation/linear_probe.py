import numpy as np
import torch
import torch.nn as nn

# TODO: also implement retrieval AUC for raw embeddings as an additional embedding evaluation method.
from encoder_pipeline.evaluation.metrics import classification_metrics, per_class_metrics


def remap_labels(
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]], class_names: list[str], label_map: dict[str, str],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    """Collapses class labels through label_map (unmapped names pass through),
    the same way DataLoaderConfig.class_label_map does in training, re-indexing
    every split's labels to the new sorted class list. Returns the remapped
    embeddings and that class list."""
    mapped = [label_map.get(name, name) for name in class_names]
    new_classes = sorted(set(mapped))
    new_idx = {name: i for i, name in enumerate(new_classes)}
    old_to_new = np.array([new_idx[name] for name in mapped])

    to_new = np.eye(len(new_classes))[old_to_new]  # multi-label rows: a new class is set if any old one mapped to it is
    remapped = {split: (x, old_to_new[y] if y.ndim == 1 else (y @ to_new > 0).astype(y.dtype))
                for split, (x, y) in embeddings.items()}
    return remapped, new_classes


class LinearProbe:
    """Trains a single linear layer on frozen embeddings to predict labels
    -- full-batch Adam + cross-entropy, or sigmoid + BCE for multi-label (2-D) labels"""

    def __init__(self, epochs: int = 1000, lr: float = 3e-4) -> None:
        self.epochs = epochs
        self.lr = lr

    def evaluate(
        self, embeddings: dict[str, tuple[np.ndarray, np.ndarray]], class_names: list[str],
    ) -> tuple[dict[str, float], dict[str, list[float]], dict[str, np.ndarray]]:
        """Returns the final metrics, the per-epoch loss curves, and every
        split's final class scores (one row per sample, in input order)."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        multi_label = next(iter(embeddings.values()))[1].ndim == 2
        label_dtype = torch.float32 if multi_label else torch.long
        tensors = {
            split: (torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(y, dtype=label_dtype, device=device))
            for split, (x, y) in embeddings.items()
        }
        x_train, y_train = tensors["train"]

        # sized by the full class list: a train split can lack classes that val / test have
        model = nn.Linear(x_train.shape[1], len(class_names)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss() if multi_label else nn.CrossEntropyLoss()

        loss_curves: dict[str, list[float]] = {f"{split}_loss": [] for split in tensors}
        for _ in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            train_loss = criterion(model(x_train), y_train)
            train_loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                for split, (x, y) in tensors.items():
                    split_loss = train_loss.item() if split == "train" else criterion(model(x), y).item()
                    loss_curves[f"{split}_loss"].append(split_loss)

        model.eval()
        self.model = model  # the trained layer, for callers that keep it (EmbeddingsConfig.linear_probe_save)
        metrics: dict[str, float] = {}
        scores: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for split, (x, y) in tensors.items():
                logits = model(x)
                y_score = (torch.sigmoid(logits) if multi_label else torch.softmax(logits, dim=1)).cpu().numpy()
                y_true = y.cpu().numpy().astype(int) if multi_label else y.cpu().numpy()
                y_pred = (y_score > 0.5).astype(int) if multi_label else y_score.argmax(axis=1)
                scores[split] = y_score
                metrics[f"{split}_accuracy"] = float((y_pred == y_true).mean())
                for name, value in classification_metrics(y_true, y_pred, y_score).items():
                    metrics[f"{split}_{name}"] = value
                if split == "test":
                    for name, value in per_class_metrics(y_true, y_pred, y_score, class_names).items():
                        metrics[f"test_{name}"] = value
        return metrics, loss_curves, scores
