from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

class ThresholdOptimizer:
    def __init__(
        self,
        n_steps: int = 100,
        scale_min: float = 0.6,
        scale_max: float = 1.5,
        adaptive_scale_max: bool = True,
        small_class_threshold: int = 150,
        small_class_scale_max: float = 3.0,
    ):
        self.n_steps = n_steps
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.adaptive_scale_max = adaptive_scale_max
        self.small_class_threshold = small_class_threshold
        self.small_class_scale_max = small_class_scale_max
        self.scales: np.ndarray | None = None

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> "ThresholdOptimizer":
        num_classes = probs.shape[1]
        scales = np.ones(num_classes, dtype=np.float64)
        support = np.bincount(y_true.astype(int), minlength=num_classes)
        baseline_preds = np.argmax(probs, axis=1)
        baseline_f1 = f1_score(y_true, baseline_preds, average="macro", zero_division=0)

        for c in range(num_classes):
            if self.adaptive_scale_max and support[c] < self.small_class_threshold:
                s_max = self.small_class_scale_max
            else:
                s_max = self.scale_max

            search_grid = np.linspace(self.scale_min, s_max, self.n_steps)

            best_scale = 1.0
            best_f1 = -1.0
            for scale in search_grid:
                test_scales = scales.copy()
                test_scales[c] = scale
                preds = np.argmax(probs * test_scales, axis=1)
                f1 = f1_score(y_true, preds, labels=[c], average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_scale = scale
            scales[c] = best_scale

        optimized_preds = np.argmax(probs * scales, axis=1)
        optimized_f1 = f1_score(y_true, optimized_preds, average="macro", zero_division=0)

        if optimized_f1 >= baseline_f1:
            self.scales = scales
        else:
            self.scales = np.ones(num_classes, dtype=np.float64)

        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        if self.scales is None:
            return np.argmax(probs, axis=1)
        return np.argmax(probs * self.scales, axis=1)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scales": self.scales.tolist() if self.scales is not None else None,
            "n_steps": self.n_steps,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "adaptive_scale_max": self.adaptive_scale_max,
            "small_class_threshold": self.small_class_threshold,
            "small_class_scale_max": self.small_class_scale_max,
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ThresholdOptimizer":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        obj = cls(
            n_steps=d.get("n_steps", 100),
            scale_min=d.get("scale_min", 0.6),
            scale_max=d.get("scale_max", 1.5),
            adaptive_scale_max=d.get("adaptive_scale_max", True),
            small_class_threshold=d.get("small_class_threshold", 150),
            small_class_scale_max=d.get("small_class_scale_max", 3.0),
        )
        if d.get("scales") is not None:
            obj.scales = np.array(d["scales"], dtype=np.float64)
        return obj

    def summary(self) -> dict[int, float]:
        if self.scales is None:
            return {}
        return {i: float(s) for i, s in enumerate(self.scales) if abs(s - 1.0) > 1e-4}