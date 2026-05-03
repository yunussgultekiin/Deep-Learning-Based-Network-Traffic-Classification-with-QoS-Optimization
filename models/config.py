from dataclasses import dataclass, field
from pathlib import Path
import argparse
from typing import Dict, Tuple

@dataclass
class TrainConfig:
    splits_dir: Path = Path("data/processed/splits")
    models_dir: Path = Path("saved_models")
    class_weights_path: Path = Path("saved_models/class_weights.npy")
    label_encoder_path: Path = Path("saved_models/label_encoder.pkl")
    model_path: Path = Path("saved_models/dnn_model.pt")
    metrics_path: Path = Path("saved_models/metrics.json")
    threshold_path: Path = Path("saved_models/thresholds.json")
    batch_size: int = 256
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    ft_dim: int = 256
    ft_depth: int = 8
    ft_heads: int = 8
    ft_attn_dropout: float = 0.1
    ft_ff_dropout: float = 0.1
    ft_ff_mult: int = 4
    ft_emb_dropout: float = 0.1
    ft_cls_dropout: float = 0.1
    loss: str = "focal"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.05
    use_weighted_sampler: bool = False
    auto_class_weights: bool = True
    class_weight_power: float = 1.3
    scheduler_type: str = "onecycle"
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    onecycle_pct_start: float = 0.1
    onecycle_div_factor: float = 25.0
    onecycle_final_div_factor: float = 10000.0
    use_ema: bool = True
    ema_decay: float = 0.999
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    optimize_thresholds: bool = True
    use_confusion_penalty: bool = True
    confusion_penalty_alpha: float = 0.1
    use_mixup: bool = False
    mixup_alpha: float = 0.3
    threshold_small_class_limit: int = 150
    threshold_small_class_scale_max: float = 3.0
    seed: int = 42
    device: str = "auto"
    log_level: str = "INFO"

def parse_train_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train FT-Transformer classifier")
    parser.add_argument("--splits-dir", type=Path, default=TrainConfig.splits_dir)
    parser.add_argument("--models-dir", type=Path, default=TrainConfig.models_dir)
    parser.add_argument("--class-weights-path", type=Path, default=TrainConfig.class_weights_path)
    parser.add_argument("--label-encoder-path", type=Path, default=TrainConfig.label_encoder_path)
    parser.add_argument("--model-path", type=Path, default=TrainConfig.model_path)
    parser.add_argument("--metrics-path", type=Path, default=TrainConfig.metrics_path)
    parser.add_argument("--threshold-path", type=Path, default=TrainConfig.threshold_path)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    parser.add_argument("--ft-dim", type=int, default=TrainConfig.ft_dim)
    parser.add_argument("--ft-depth", type=int, default=TrainConfig.ft_depth)
    parser.add_argument("--ft-heads", type=int, default=TrainConfig.ft_heads)
    parser.add_argument("--ft-attn-dropout", type=float, default=TrainConfig.ft_attn_dropout)
    parser.add_argument("--ft-ff-dropout", type=float, default=TrainConfig.ft_ff_dropout)
    parser.add_argument("--ft-ff-mult", type=int, default=TrainConfig.ft_ff_mult)
    parser.add_argument("--ft-emb-dropout", type=float, default=TrainConfig.ft_emb_dropout)
    parser.add_argument("--ft-cls-dropout", type=float, default=TrainConfig.ft_cls_dropout)
    parser.add_argument("--loss", type=str, default=TrainConfig.loss, choices=["ce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=TrainConfig.focal_gamma)
    parser.add_argument("--label-smoothing", type=float, default=TrainConfig.label_smoothing)
    parser.add_argument("--use-weighted-sampler", action=argparse.BooleanOptionalAction, default=TrainConfig.use_weighted_sampler)
    parser.add_argument("--auto-class-weights", action=argparse.BooleanOptionalAction, default=TrainConfig.auto_class_weights)
    parser.add_argument("--class-weight-power", type=float, default=TrainConfig.class_weight_power)
    parser.add_argument("--scheduler-type", type=str, default=TrainConfig.scheduler_type, choices=["none", "plateau", "onecycle"])
    parser.add_argument("--scheduler-patience", type=int, default=TrainConfig.scheduler_patience)
    parser.add_argument("--scheduler-factor", type=float, default=TrainConfig.scheduler_factor)
    parser.add_argument("--min-lr", type=float, default=TrainConfig.min_lr)
    parser.add_argument("--onecycle-pct-start", type=float, default=TrainConfig.onecycle_pct_start)
    parser.add_argument("--onecycle-div-factor", type=float, default=TrainConfig.onecycle_div_factor)
    parser.add_argument("--onecycle-final-div-factor", type=float, default=TrainConfig.onecycle_final_div_factor)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=TrainConfig.use_ema)
    parser.add_argument("--ema-decay", type=float, default=TrainConfig.ema_decay)
    parser.add_argument("--early-stopping-patience", type=int, default=TrainConfig.early_stopping_patience)
    parser.add_argument("--early-stopping-min-delta", type=float, default=TrainConfig.early_stopping_min_delta)
    parser.add_argument("--optimize-thresholds", action=argparse.BooleanOptionalAction, default=TrainConfig.optimize_thresholds)
    parser.add_argument("--use-confusion-penalty", action=argparse.BooleanOptionalAction, default=TrainConfig.use_confusion_penalty)
    parser.add_argument("--confusion-penalty-alpha", type=float, default=TrainConfig.confusion_penalty_alpha)
    parser.add_argument("--use-mixup", action=argparse.BooleanOptionalAction, default=TrainConfig.use_mixup)
    parser.add_argument("--mixup-alpha", type=float, default=TrainConfig.mixup_alpha)
    parser.add_argument("--threshold-small-class-limit", type=int, default=TrainConfig.threshold_small_class_limit)
    parser.add_argument("--threshold-small-class-scale-max", type=float, default=TrainConfig.threshold_small_class_scale_max)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--device", type=str, default=TrainConfig.device)
    parser.add_argument("--log-level", type=str, default=TrainConfig.log_level)
    args = parser.parse_args()
    return TrainConfig(
        splits_dir=args.splits_dir,
        models_dir=args.models_dir,
        class_weights_path=args.class_weights_path,
        label_encoder_path=args.label_encoder_path,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        threshold_path=args.threshold_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ft_dim=args.ft_dim,
        ft_depth=args.ft_depth,
        ft_heads=args.ft_heads,
        ft_attn_dropout=args.ft_attn_dropout,
        ft_ff_dropout=args.ft_ff_dropout,
        ft_ff_mult=args.ft_ff_mult,
        ft_emb_dropout=args.ft_emb_dropout,
        ft_cls_dropout=args.ft_cls_dropout,
        loss=args.loss,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        use_weighted_sampler=args.use_weighted_sampler,
        auto_class_weights=args.auto_class_weights,
        class_weight_power=args.class_weight_power,
        scheduler_type=args.scheduler_type,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_lr=args.min_lr,
        onecycle_pct_start=args.onecycle_pct_start,
        onecycle_div_factor=args.onecycle_div_factor,
        onecycle_final_div_factor=args.onecycle_final_div_factor,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        optimize_thresholds=args.optimize_thresholds,
        use_confusion_penalty=args.use_confusion_penalty,
        confusion_penalty_alpha=args.confusion_penalty_alpha,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        threshold_small_class_limit=args.threshold_small_class_limit,
        threshold_small_class_scale_max=args.threshold_small_class_scale_max,
        seed=args.seed,
        device=args.device,
        log_level=args.log_level,
    )

@dataclass
class EvalConfig:
    splits_dir: Path = Path("data/processed/splits")
    model_path: Path = Path("saved_models/dnn_model.pt")
    metrics_path: Path = Path("saved_models/metrics.json")
    threshold_path: Path = Path("saved_models/thresholds.json")
    batch_size: int = 256
    device: str = "auto"
    log_level: str = "INFO"

def parse_eval_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate FT-Transformer classifier")
    parser.add_argument("--splits-dir", type=Path, default=EvalConfig.splits_dir)
    parser.add_argument("--model-path", type=Path, default=EvalConfig.model_path)
    parser.add_argument("--metrics-path", type=Path, default=EvalConfig.metrics_path)
    parser.add_argument("--threshold-path", type=Path, default=EvalConfig.threshold_path)
    parser.add_argument("--batch-size", type=int, default=EvalConfig.batch_size)
    parser.add_argument("--device", type=str, default=EvalConfig.device)
    parser.add_argument("--log-level", type=str, default=EvalConfig.log_level)
    args = parser.parse_args()
    return EvalConfig(
        splits_dir=args.splits_dir,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        threshold_path=args.threshold_path,
        batch_size=args.batch_size,
        device=args.device,
        log_level=args.log_level,
    )