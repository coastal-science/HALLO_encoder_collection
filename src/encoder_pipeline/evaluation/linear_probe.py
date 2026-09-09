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

    remapped = {split: (x, old_to_new[y]) for split, (x, y) in embeddings.items()}
    return remapped, new_classes


class LinearProbe:
    """Trains a single linear layer on frozen embeddings to predict labels
    -- full-batch Adam + cross-entropy"""

    def __init__(self, epochs: int = 1000, lr: float = 3e-4) -> None:
        self.epochs = epochs
        self.lr = lr

    def evaluate(
        self, embeddings: dict[str, tuple[np.ndarray, np.ndarray]], class_names: list[str],
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensors = {
            split: (torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(y, dtype=torch.long, device=device))
            for split, (x, y) in embeddings.items()
        }
        x_train, y_train = tensors["train"]

        model = nn.Linear(x_train.shape[1], int(y_train.max().item()) + 1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

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
        metrics: dict[str, float] = {}
        with torch.no_grad():
            for split, (x, y) in tensors.items():
                logits = model(x)
                y_true, y_pred = y.cpu().numpy(), logits.argmax(dim=1).cpu().numpy()
                y_score = torch.softmax(logits, dim=1).cpu().numpy()
                metrics[f"{split}_accuracy"] = float((y_pred == y_true).mean())
                for name, value in classification_metrics(y_true, y_pred, y_score).items():
                    metrics[f"{split}_{name}"] = value
                if split == "test":
                    for name, value in per_class_metrics(y_true, y_pred, y_score, class_names).items():
                        metrics[f"test_{name}"] = value
        return metrics, loss_curves
