from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .metrics import classification_metrics
from .ema import ExponentialMovingAverage

@dataclass
class EpochStats:
    loss: float
    accuracy: float

class EarlyStopping:

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best: float = float("inf")
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

def mixup_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.3,
    target_classes: Optional[set] = None,
) -> torch.Tensor:

    if target_classes is not None:
        mask = torch.zeros(y.size(0), dtype=torch.bool, device=y.device)
        for c in target_classes:
            mask |= (y == c)
        if mask.sum() < 2:
            return X
        X_sub = X[mask]
        idx = torch.randperm(X_sub.size(0), device=X.device)
        lam = float(np.random.beta(alpha, alpha))
        X_mixed = X.clone()
        X_mixed[mask] = lam * X_sub + (1.0 - lam) * X_sub[idx]
        return X_mixed
    else:
        lam = float(np.random.beta(alpha, alpha))
        idx = torch.randperm(X.size(0), device=X.device)
        return lam * X + (1.0 - lam) * X[idx]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        logger,
        ema: ExponentialMovingAverage | None = None,
        use_mixup: bool = False,
        mixup_alpha: float = 0.3,
        mixup_target_classes: Optional[set] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.ema = ema
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.mixup_target_classes = mixup_target_classes
        self.model.to(self.device)

    def _run_epoch(
        self,
        loader: DataLoader,
        train: bool,
        batch_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    ) -> EpochStats:
        if train:
            self.model.train()
        else:
            self.model.eval()

        losses: list[float] = []
        correct = 0
        total = 0

        for X, y in loader:
            X = X.to(self.device)
            y = y.to(self.device)

            if train and self.use_mixup:
                X = mixup_batch(
                    X, y,
                    alpha=self.mixup_alpha,
                    target_classes=self.mixup_target_classes,
                )

            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                logits = self.model(X)
                loss = self.criterion(logits, y)
                if train:
                    loss.backward()
                    self.optimizer.step()
                    if batch_scheduler is not None:
                        batch_scheduler.step()
                    if self.ema is not None:
                        self.ema.update(self.model)

            losses.append(loss.item())
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = float(np.mean(losses)) if losses else 0.0
        accuracy = float(correct / total) if total else 0.0
        return EpochStats(loss=avg_loss, accuracy=accuracy)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        model_path: Path,
        epoch_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None,
        batch_scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        early_stopping: EarlyStopping | None = None,
    ) -> dict:
        best_val_loss = float("inf")
        history = []

        for epoch in range(1, epochs + 1):
            train_stats = self._run_epoch(train_loader, train=True, batch_scheduler=batch_scheduler)
            val_stats = self._run_epoch(val_loader, train=False)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_stats.loss,
                    "train_accuracy": train_stats.accuracy,
                    "val_loss": val_stats.loss,
                    "val_accuracy": val_stats.accuracy,
                }
            )

            self.logger.info(
                "Epoch %d | train loss %.4f acc %.4f | val loss %.4f acc %.4f",
                epoch,
                train_stats.loss,
                train_stats.accuracy,
                val_stats.loss,
                val_stats.accuracy,
            )

            if val_stats.loss < best_val_loss:
                best_val_loss = val_stats.loss
                if self.ema is not None:
                    self.ema.apply_to(self.model)
                    self.save_model(model_path)
                    self.ema.restore(self.model)
                else:
                    self.save_model(model_path)
                self.logger.info("Epoch %d | checkpoint kaydedildi (val_loss=%.4f)", epoch, best_val_loss)

            if epoch_scheduler is not None:
                epoch_scheduler.step(val_stats.loss)

            if early_stopping is not None and early_stopping.step(val_stats.loss):
                self.logger.info(
                    "Early stopping tetiklendi (epoch %d, patience=%d, best_val_loss=%.4f)",
                    epoch,
                    early_stopping.patience,
                    best_val_loss,
                )
                break

        return {
            "best_val_loss": float(best_val_loss),
            "stopped_epoch": len(history),
            "history": history,
        }

    def get_probs(self, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()
        all_probs: list[np.ndarray] = []
        all_true: list[np.ndarray] = []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                logits = self.model(X)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                all_true.append(y.numpy())
        probs_arr = np.concatenate(all_probs) if all_probs else np.empty((0, 0))
        y_true_arr = np.concatenate(all_true) if all_true else np.array([], dtype=int)
        return probs_arr, y_true_arr

    def evaluate(self, loader: DataLoader, num_classes: int, threshold_optimizer=None) -> dict:
        probs, y_true = self.get_probs(loader)
        if probs.ndim < 2 or probs.shape[0] == 0:
            return classification_metrics(np.array([]), np.array([]), num_classes)

        if threshold_optimizer is not None:
            y_pred = threshold_optimizer.predict(probs)
        else:
            y_pred = np.argmax(probs, axis=1)

        return classification_metrics(y_true, y_pred, num_classes)

    def save_model(self, model_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_path)


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)