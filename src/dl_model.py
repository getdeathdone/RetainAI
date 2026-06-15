"""PyTorch MLP model for RetainAI churn prediction.

The deep learning model consumes already preprocessed tabular matrices produced
by `ChurnDataPreprocessor`. It is intentionally import-friendly for the future
FastAPI layer: model construction, inference, save and load operations are
encapsulated in `ChurnDeepLearningModel`.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset


LOGGER = logging.getLogger(__name__)


def select_device() -> torch.device:
    """Select the best available PyTorch device: CUDA, MPS, then CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class TabularChurnDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset wrapper for preprocessed tabular churn matrices."""

    def __init__(
        self,
        features: NDArray[np.float32] | NDArray[np.float64] | torch.Tensor,
        targets: NDArray[np.int_] | NDArray[np.float32] | torch.Tensor,
    ) -> None:
        self.features = self._to_feature_tensor(features)
        self.targets = self._to_target_tensor(targets)

        if self.features.ndim != 2:
            raise ValueError(f"features must be a 2D matrix. Got shape {tuple(self.features.shape)}.")

        if self.targets.ndim != 1:
            raise ValueError(f"targets must be a 1D vector. Got shape {tuple(self.targets.shape)}.")

        if len(self.features) != len(self.targets):
            raise ValueError(
                "features and targets must contain the same number of rows. "
                f"Got {len(self.features)} and {len(self.targets)}."
            )

        if len(self.features) == 0:
            raise ValueError("Dataset cannot be empty.")

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]

    @staticmethod
    def _to_feature_tensor(features: NDArray[np.float32] | NDArray[np.float64] | torch.Tensor) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            tensor = features.detach().clone().float()
        else:
            tensor = torch.as_tensor(np.asarray(features), dtype=torch.float32)

        if not torch.isfinite(tensor).all():
            raise ValueError("features contain NaN or infinite values.")

        return tensor

    @staticmethod
    def _to_target_tensor(targets: NDArray[np.int_] | NDArray[np.float32] | torch.Tensor) -> torch.Tensor:
        if isinstance(targets, torch.Tensor):
            tensor = targets.detach().clone().float()
        else:
            tensor = torch.as_tensor(np.asarray(targets), dtype=torch.float32)

        unique_values = set(torch.unique(tensor).cpu().numpy().tolist())
        if not unique_values.issubset({0.0, 1.0}):
            raise ValueError(f"targets must be binary with values 0/1. Got {unique_values}.")

        return tensor


class ChurnMLP(nn.Module):
    """Feed-forward neural network for tabular churn classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.25,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")

        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer size.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the [0, 1) range.")

        activation_layer: type[nn.Module]
        if activation.lower() == "relu":
            activation_layer = nn.ReLU
        elif activation.lower() == "gelu":
            activation_layer = nn.GELU
        else:
            raise ValueError("activation must be either 'relu' or 'gelu'.")

        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("All hidden layer sizes must be positive integers.")

            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    activation_layer(),
                    nn.Dropout(dropout),
                ]
            )
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw logits. Sigmoid is applied only for metrics/inference."""

        return self.network(features).squeeze(dim=-1)


@dataclass(slots=True)
class ChurnDeepLearningModel:
    """Training, evaluation and persistence wrapper for ChurnMLP."""

    input_dim: int
    hidden_dims: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.25
    activation: str = "gelu"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 30
    patience: int = 5
    threshold: float = 0.5
    artifact_path: Path = Path("artifacts/dl_model.pth")
    device: torch.device = field(default_factory=select_device)
    model: ChurnMLP = field(init=False)

    def __post_init__(self) -> None:
        self.model = ChurnMLP(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            activation=self.activation,
        ).to(self.device)
        LOGGER.info("Initialized ChurnMLP on device '%s'.", self.device)

    def fit(
        self,
        X_train: NDArray[np.float32] | NDArray[np.float64],
        y_train: NDArray[np.int_] | NDArray[np.float32],
        X_val: NDArray[np.float32] | NDArray[np.float64],
        y_val: NDArray[np.int_] | NDArray[np.float32],
    ) -> dict[str, list[float]]:
        """Train the neural network with validation and early stopping."""

        self._validate_input_dim(X_train, "X_train")
        self._validate_input_dim(X_val, "X_val")

        train_dataset = TabularChurnDataset(X_train, y_train)
        val_dataset = TabularChurnDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        pos_weight = self._compute_pos_weight(train_dataset.targets).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_roc_auc": [],
            "val_f1": [],
        }

        best_val_loss = float("inf")
        best_state_dict: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0

        LOGGER.info(
            "Starting DL training: train_rows=%s, val_rows=%s, input_dim=%s, pos_weight=%.4f.",
            len(train_dataset),
            len(val_dataset),
            self.input_dim,
            float(pos_weight.item()),
        )

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_one_epoch(train_loader, criterion, optimizer)
            val_loss, val_probabilities, val_targets = self._predict_loader_loss(val_loader, criterion)
            val_metrics = self._calculate_metrics(val_targets, val_probabilities)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_roc_auc"].append(val_metrics["roc_auc"])
            history["val_f1"].append(val_metrics["f1"])

            LOGGER.info(
                "Epoch %s/%s | train_loss=%.5f | val_loss=%.5f | val_roc_auc=%.5f | val_f1=%.5f",
                epoch,
                self.max_epochs,
                train_loss,
                val_loss,
                val_metrics["roc_auc"],
                val_metrics["f1"],
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                LOGGER.info("Early stopping triggered at epoch %s.", epoch)
                break

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        LOGGER.info("DL training finished. Best val_loss=%.5f.", best_val_loss)
        return history

    def evaluate(
        self,
        X: NDArray[np.float32] | NDArray[np.float64],
        y: NDArray[np.int_] | NDArray[np.float32],
    ) -> dict[str, float]:
        """Evaluate ROC-AUC and F1 on preprocessed matrices."""

        self._validate_input_dim(X, "X")
        dataset = TabularChurnDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        probabilities, targets = self.predict_proba_loader(loader)
        metrics = self._calculate_metrics(targets, probabilities)
        LOGGER.info("DL evaluation metrics: %s", metrics)
        return metrics

    def predict_proba(self, X: NDArray[np.float32] | NDArray[np.float64]) -> NDArray[np.float32]:
        """Return churn probabilities for preprocessed feature rows."""

        self._validate_input_dim(X, "X")
        dummy_targets = np.zeros(shape=(len(X),), dtype=np.float32)
        dataset = TabularChurnDataset(X, dummy_targets)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        probabilities, _ = self.predict_proba_loader(loader)
        return probabilities

    def predict_proba_loader(self, loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return probabilities and targets for all batches from a DataLoader."""

        self.model.eval()
        probabilities: list[NDArray[np.float32]] = []
        targets: list[NDArray[np.float32]] = []

        with torch.no_grad():
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(self.device)
                logits = self.model(batch_features)
                batch_probabilities = torch.sigmoid(logits)

                probabilities.append(batch_probabilities.detach().cpu().numpy().astype(np.float32))
                targets.append(batch_targets.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(probabilities), np.concatenate(targets)

    def save_model(self, artifact_path: str | Path | None = None) -> Path:
        """Save model weights and architecture config for later FastAPI loading."""

        output_path = Path(artifact_path) if artifact_path is not None else self.artifact_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {
            "state_dict": self.model.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "activation": self.activation,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
                "threshold": self.threshold,
            },
        }
        torch.save(checkpoint, output_path)
        LOGGER.info("Saved DL model artifact to %s.", output_path)
        return output_path

    @classmethod
    def load_model(
        cls,
        artifact_path: str | Path,
        map_location: str | torch.device | None = None,
    ) -> "ChurnDeepLearningModel":
        """Load a saved ChurnMLP checkpoint from disk."""

        input_path = Path(artifact_path)
        if not input_path.exists():
            raise FileNotFoundError(f"DL model artifact not found: {input_path}")

        target_device = torch.device(map_location) if map_location is not None else select_device()
        checkpoint = torch.load(input_path, map_location=target_device)
        config = checkpoint["config"]

        model_wrapper = cls(
            input_dim=int(config["input_dim"]),
            hidden_dims=tuple(config["hidden_dims"]),
            dropout=float(config["dropout"]),
            activation=str(config["activation"]),
            learning_rate=float(config.get("learning_rate", 1e-3)),
            weight_decay=float(config.get("weight_decay", 1e-4)),
            batch_size=int(config.get("batch_size", 128)),
            threshold=float(config.get("threshold", 0.5)),
            artifact_path=input_path,
            device=target_device,
        )
        model_wrapper.model.load_state_dict(checkpoint["state_dict"])
        model_wrapper.model.eval()
        LOGGER.info("Loaded DL model artifact from %s.", input_path)
        return model_wrapper

    def _train_one_epoch(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        total_rows = 0

        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            logits = self.model(batch_features)
            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            batch_size = batch_features.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_rows += batch_size

        return total_loss / max(total_rows, 1)

    def _predict_loader_loss(
        self,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
    ) -> tuple[float, NDArray[np.float32], NDArray[np.float32]]:
        self.model.eval()
        total_loss = 0.0
        total_rows = 0
        probabilities: list[NDArray[np.float32]] = []
        targets: list[NDArray[np.float32]] = []

        with torch.no_grad():
            for batch_features, batch_targets in loader:
                batch_features = batch_features.to(self.device)
                batch_targets_device = batch_targets.to(self.device)

                logits = self.model(batch_features)
                loss = criterion(logits, batch_targets_device)
                batch_probabilities = torch.sigmoid(logits)

                batch_size = batch_features.shape[0]
                total_loss += float(loss.item()) * batch_size
                total_rows += batch_size

                probabilities.append(batch_probabilities.detach().cpu().numpy().astype(np.float32))
                targets.append(batch_targets.detach().cpu().numpy().astype(np.float32))

        return (
            total_loss / max(total_rows, 1),
            np.concatenate(probabilities),
            np.concatenate(targets),
        )

    def _calculate_metrics(
        self,
        targets: NDArray[np.float32],
        probabilities: NDArray[np.float32],
    ) -> dict[str, float]:
        predictions = (probabilities >= self.threshold).astype(np.int64)
        target_ints = targets.astype(np.int64)

        try:
            roc_auc = float(roc_auc_score(target_ints, probabilities))
        except ValueError:
            LOGGER.warning("ROC-AUC is undefined because only one class is present.")
            roc_auc = float("nan")

        return {
            "roc_auc": roc_auc,
            "f1": float(f1_score(target_ints, predictions, zero_division=0)),
        }

    @staticmethod
    def _compute_pos_weight(targets: torch.Tensor) -> torch.Tensor:
        positives = torch.sum(targets == 1).float()
        negatives = torch.sum(targets == 0).float()

        if positives.item() == 0:
            LOGGER.warning("No positive class examples found; using pos_weight=1.0.")
            return torch.tensor(1.0, dtype=torch.float32)

        return negatives / positives

    def _validate_input_dim(self, X: NDArray[np.float32] | NDArray[np.float64], name: str) -> None:
        if not isinstance(X, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray.")

        if X.ndim != 2:
            raise ValueError(f"{name} must be a 2D matrix. Got shape {X.shape}.")

        if X.shape[1] != self.input_dim:
            raise ValueError(f"{name} has {X.shape[1]} features, expected {self.input_dim}.")

        if not np.isfinite(X).all():
            raise ValueError(f"{name} contains NaN or infinite values.")


def _build_fallback_dataset(
    n_train: int = 800,
    n_test: int = 200,
    n_features: int = 32,
    random_state: int = 42,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]]:
    """Create synthetic preprocessed matrices when sklearn preprocessing is unavailable."""

    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_train + n_test, n_features)).astype(np.float32)
    signal = 1.4 * X[:, 0] - 1.0 * X[:, 1] + 0.7 * X[:, 2] + rng.normal(scale=0.9, size=X.shape[0])
    churn_probability = 1 / (1 + np.exp(-signal))
    y = rng.binomial(1, churn_probability).astype(np.int64)

    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def _build_demo_dataset() -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]]:
    """Try the real preprocessor path first, then fall back to random matrices."""

    try:
        from preprocessing import ChurnDataPreprocessor, build_fake_feature_mart

        fake_df = build_fake_feature_mart(n_rows=1000, random_state=42)
        preprocessor = ChurnDataPreprocessor()
        prepared = preprocessor.prepare_data_with_metadata(fake_df)
        preprocessor.save_preprocessor("artifacts/preprocessor.joblib")
        LOGGER.info("Demo dataset prepared through ChurnDataPreprocessor.")
        return (
            prepared.X_train,
            prepared.X_test,
            prepared.y_train.to_numpy(dtype=np.int64),
            prepared.y_test.to_numpy(dtype=np.int64),
        )
    except ImportError as exc:
        LOGGER.warning(
            "Could not import ChurnDataPreprocessor. "
            "Using synthetic preprocessed matrices instead. Reason: %s",
            exc,
        )
        return _build_fallback_dataset()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    X_train_demo, X_test_demo, y_train_demo, y_test_demo = _build_demo_dataset()

    dl_model = ChurnDeepLearningModel(
        input_dim=X_train_demo.shape[1],
        hidden_dims=(64, 32),
        dropout=0.2,
        learning_rate=1e-3,
        batch_size=128,
        max_epochs=5,
        patience=3,
    )
    history = dl_model.fit(
        X_train=X_train_demo,
        y_train=y_train_demo,
        X_val=X_test_demo,
        y_val=y_test_demo,
    )
    metrics = dl_model.evaluate(X_test_demo, y_test_demo)
    artifact = dl_model.save_model("artifacts/dl_model.pth")

    LOGGER.info("Smoke test history: %s", history)
    LOGGER.info("Smoke test metrics: %s", metrics)
    LOGGER.info("Smoke test DL artifact: %s", artifact)
