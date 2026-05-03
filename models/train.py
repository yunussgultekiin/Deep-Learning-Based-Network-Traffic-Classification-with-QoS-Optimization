from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from .config import parse_train_args
from .data import SplitData, make_loader, make_weighted_sampler, compute_class_weights
from .ema import ExponentialMovingAverage
from .ft_transformer import FTTransformer
from .losses import FocalLoss, ConfusionPenaltyLoss
from .metrics import classification_metrics
from .threshold import ThresholdOptimizer
from .trainer import EarlyStopping, Trainer, save_metrics
from .utils import get_logger, set_seed, pick_device

CONFUSED_PAIRS = {
    (0, 7):  1.5,
    (7, 0):  1.5,
    (6, 13): 1.2,
    (13, 6): 1.2,
    (1, 8):  1.3,
    (9, 7):  1.3,
    (1, 8):  1.0,
    (9, 7):  0.8,
}

MIXUP_TARGET_CLASSES = set()

def main() -> None:
    cfg = parse_train_args()
    logger = get_logger("train", cfg.log_level)
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    data = SplitData(cfg.splits_dir).load()
    X_train = data["X_train"]
    X_val   = data["X_val"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_val   = data["y_val"]
    y_test  = data["y_test"]
    input_dim   = int(X_train.shape[1])
    num_classes = int(np.max(y_train)) + 1
    run_test_eval: bool = True

    sampler = (
        make_weighted_sampler(y_train, power=cfg.class_weight_power)
        if cfg.use_weighted_sampler
        else None
    )
    if sampler is not None:
        logger.info("WeightedRandomSampler aktif (power=%.2f)", cfg.class_weight_power)

    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=sampler is None, sampler=sampler)
    val_loader   = make_loader(X_val,   y_val,   cfg.batch_size, shuffle=False)
    test_loader  = make_loader(X_test,  y_test,  cfg.batch_size, shuffle=False)

    model = FTTransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        dim=cfg.ft_dim,
        depth=cfg.ft_depth,
        heads=cfg.ft_heads,
        attn_dropout=cfg.ft_attn_dropout,
        ff_dropout=cfg.ft_ff_dropout,
        ff_mult=cfg.ft_ff_mult,
        emb_dropout=cfg.ft_emb_dropout,
        cls_dropout=cfg.ft_cls_dropout,
    )

    class_weights: torch.Tensor | None = None
    if cfg.class_weights_path.exists() and not cfg.auto_class_weights:
        weights = np.load(cfg.class_weights_path)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        logger.info("Class weights were loaded from the file: %s", cfg.class_weights_path)
    else:
        weights = compute_class_weights(y_train, power=cfg.class_weight_power)
        class_weights = torch.tensor(weights, dtype=torch.float32)
        cfg.models_dir.mkdir(parents=True, exist_ok=True)
        np.save(cfg.class_weights_path, weights)
        logger.info(
            "Class weights were calculated from the training data (power=%.2f) and saved as: %s",
            cfg.class_weight_power,
            cfg.class_weights_path,
        )
        top3_idx = np.argsort(weights)[::-1][:3]
        for idx in top3_idx:
            logger.info(
                "Class %2d → weight=%.4f (support=%d)",
                idx, weights[idx], int(np.sum(y_train == idx)),
            )

    if cfg.loss == "focal":
        base_criterion = FocalLoss(
            gamma=cfg.focal_gamma,
            weight=class_weights.to(device) if class_weights is not None else None,
        )
        logger.info("FocalLoss (gamma=%.1f) is being used.", cfg.focal_gamma)
    else:
        base_criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None,
            label_smoothing=cfg.label_smoothing,
        )

    if cfg.use_confusion_penalty:
        criterion = ConfusionPenaltyLoss(
            base_loss=base_criterion,
            confused_pairs=CONFUSED_PAIRS,
            alpha=cfg.confusion_penalty_alpha,
        )
        logger.info(
            "ConfusionPenaltyLoss active (alpha=%.2f, %d even)",
            cfg.confusion_penalty_alpha,
            len(CONFUSED_PAIRS),
        )
        for (tc, pc), w in CONFUSED_PAIRS.items():
            logger.info("  (%d→%d) penalty_weight=%.1f", tc, pc, w)
    else:
        criterion = base_criterion

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    epoch_scheduler = None
    batch_scheduler = None
    if cfg.scheduler_type == "plateau":
        epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=cfg.scheduler_patience,
            factor=cfg.scheduler_factor, min_lr=cfg.min_lr,
        )
    elif cfg.scheduler_type == "onecycle":
        total_steps = cfg.epochs * len(train_loader)
        batch_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr, total_steps=total_steps,
            pct_start=cfg.onecycle_pct_start,
            div_factor=cfg.onecycle_div_factor,
            final_div_factor=cfg.onecycle_final_div_factor,
        )

    ema = ExponentialMovingAverage(model, decay=cfg.ema_decay) if cfg.use_ema else None

    early_stopping = EarlyStopping(
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
    )
    logger.info(
        "Early stopping active — patience=%d, min_delta=%.1e",
        cfg.early_stopping_patience, cfg.early_stopping_min_delta,
    )

    logger.info(
        "FTTransformer: dim=%d, depth=%d, heads=%d",
        cfg.ft_dim, cfg.ft_depth, cfg.ft_heads,
    )

    if cfg.use_mixup:
        logger.info(
            "MixUp active — alpha=%.2f, target classes: %s",
            cfg.mixup_alpha, sorted(MIXUP_TARGET_CLASSES),
        )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        logger=logger,
        ema=ema,
        use_mixup=cfg.use_mixup,
        mixup_alpha=cfg.mixup_alpha,
        mixup_target_classes=MIXUP_TARGET_CLASSES if cfg.use_mixup else None,
    )
    logger.info("Training started on %s", device)

    train_summary = trainer.fit(
        train_loader, val_loader, cfg.epochs, cfg.model_path,
        epoch_scheduler=epoch_scheduler,
        batch_scheduler=batch_scheduler,
        early_stopping=early_stopping,
    )

    logger.info("The best checkpoint is loading.: %s", cfg.model_path)
    model.load_state_dict(torch.load(cfg.model_path, map_location=device))
    threshold_optimizer: ThresholdOptimizer | None = None

    if cfg.optimize_thresholds:
        logger.info("Threshold optimization is starting (on the validation set)...")
        val_probs, val_true = trainer.get_probs(val_loader)
        threshold_optimizer = ThresholdOptimizer(
            n_steps=100,
            scale_min=0.6,
            scale_max=1.5,
            adaptive_scale_max=True,
            small_class_threshold=cfg.threshold_small_class_limit,
            small_class_scale_max=cfg.threshold_small_class_scale_max,
        )
        threshold_optimizer.fit(val_probs, val_true)
        threshold_optimizer.save(cfg.threshold_path)

        adjusted_classes = threshold_optimizer.summary()
        if adjusted_classes:
            logger.info(
                "Threshold calibration is complete. The classes that have been set are:%s",
                {k: f"{v:.3f}" for k, v in adjusted_classes.items()},
            )
        else:
            logger.info("Threshold optimization favored uniform scale.")

    metrics: dict = {"train": train_summary}

    if run_test_eval:
        logger.info("Final evaluation (one-off) on the test set.")
        test_metrics_raw = trainer.evaluate(test_loader, num_classes)
        metrics["test_raw"] = test_metrics_raw
        logger.info(
            "Test (raw)  — acc=%.4f, macro_f1=%.4f",
            test_metrics_raw["accuracy"], test_metrics_raw["macro_f1"],
        )

        if threshold_optimizer is not None:
            test_metrics_cal = trainer.evaluate(
                test_loader, num_classes, threshold_optimizer=threshold_optimizer
            )
            metrics["test"] = test_metrics_cal
            logger.info(
                "Test (cal.) — acc=%.4f, macro_f1=%.4f",
                test_metrics_cal["accuracy"], test_metrics_cal["macro_f1"],
            )
        else:
            metrics["test"] = test_metrics_raw
    else:
        logger.info("Test evaluation skipped (run_test_eval=False)")

    save_metrics(metrics, cfg.metrics_path)
    logger.info("Metrics were recorded: %s", cfg.metrics_path)

    meta = {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "ft_dim": cfg.ft_dim,
        "ft_depth": cfg.ft_depth,
        "ft_heads": cfg.ft_heads,
        "ft_attn_dropout": cfg.ft_attn_dropout,
        "ft_ff_dropout": cfg.ft_ff_dropout,
        "ft_ff_mult": cfg.ft_ff_mult,
        "ft_emb_dropout": cfg.ft_emb_dropout,
        "ft_cls_dropout": cfg.ft_cls_dropout,
        "loss": cfg.loss,
        "focal_gamma": cfg.focal_gamma,
        "label_smoothing": cfg.label_smoothing,
        "use_weighted_sampler": cfg.use_weighted_sampler,
        "class_weight_power": cfg.class_weight_power,
        "auto_class_weights": cfg.auto_class_weights,
        "scheduler_type": cfg.scheduler_type,
        "scheduler_patience": cfg.scheduler_patience,
        "scheduler_factor": cfg.scheduler_factor,
        "min_lr": cfg.min_lr,
        "onecycle_pct_start": cfg.onecycle_pct_start,
        "onecycle_div_factor": cfg.onecycle_div_factor,
        "onecycle_final_div_factor": cfg.onecycle_final_div_factor,
        "use_ema": cfg.use_ema,
        "ema_decay": cfg.ema_decay,
        "early_stopping_patience": cfg.early_stopping_patience,
        "early_stopping_min_delta": cfg.early_stopping_min_delta,
        "optimize_thresholds": cfg.optimize_thresholds,
        "use_confusion_penalty": cfg.use_confusion_penalty,
        "confusion_penalty_alpha": cfg.confusion_penalty_alpha,
        "confused_pairs": {f"{k[0]}-{k[1]}": v for k, v in CONFUSED_PAIRS.items()},
        "use_mixup": cfg.use_mixup,
        "mixup_alpha": cfg.mixup_alpha,
        "mixup_target_classes": sorted(MIXUP_TARGET_CLASSES),
        "threshold_small_class_limit": cfg.threshold_small_class_limit,
        "threshold_small_class_scale_max": cfg.threshold_small_class_scale_max,
        "stopped_epoch": train_summary.get("stopped_epoch"),
    }
    meta_path = Path(str(cfg.model_path) + ".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info("Model metadata recorded: %s", meta_path)


if __name__ == "__main__":
    main()