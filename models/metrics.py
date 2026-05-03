from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict[str, Any]:
    if num_classes <= 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class": [],
            "confusion_matrix": [],
        }

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.arange(num_classes)

    if y_true.size == 0:
        cm = np.zeros((num_classes, num_classes), dtype=int)
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class": [
                {
                    "class_index": int(idx),
                    "support": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }
                for idx in labels
            ],
            "confusion_matrix": cm.tolist(),
        }

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    per_class = []
    for idx in range(num_classes):
        per_class.append(
            {
                "class_index": int(idx),
                "support": int(support[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
            }
        )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
