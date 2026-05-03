from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from .config import parse_eval_args
from .data import SplitData, make_loader
from .ft_transformer import FTTransformer
from .threshold import ThresholdOptimizer
from .trainer import Trainer, save_metrics
from .utils import get_logger, pick_device

QOS_CLASS_ORDER = ["VOIP", "STREAMING", "BROWSING", "CHAT", "MAIL", "FT", "P2P"]
QOS_CLASS_CONFIG = {
    "VOIP": {
        "class_name": "CLASS-VOIP",
        "dscp": "ef",
        "policy": ["priority percent 20"],
    },
    "STREAMING": {
        "class_name": "CLASS-STREAMING",
        "dscp": "af41",
        "policy": ["bandwidth percent 20"],
    },
    "BROWSING": {
        "class_name": "CLASS-BROWSING",
        "dscp": "af21",
        "policy": ["bandwidth percent 15"],
    },
    "CHAT": {
        "class_name": "CLASS-CHAT",
        "dscp": "af11",
        "policy": ["bandwidth percent 5"],
    },
    "MAIL": {
        "class_name": "CLASS-MAIL",
        "dscp": "cs1",
        "policy": ["bandwidth percent 5"],
    },
    "FT": {
        "class_name": "CLASS-FT",
        "dscp": "default",
        "policy": ["bandwidth percent 5"],
    },
    "P2P": {
        "class_name": "CLASS-P2P",
        "dscp": "default",
        "policy": ["bandwidth percent 5"],
    },
}

def build_qos_rules(predicted_labels: list[str], packet_tracer: bool = False) -> str:
    base_labels = {normalize_label(label) for label in predicted_labels}
    ordered = [label for label in QOS_CLASS_ORDER if label in base_labels]
    if not ordered:
        return ""

    match_prefix = "match ip dscp" if packet_tracer else "match dscp"
    lines: list[str] = []

    for label in ordered:
        cfg = QOS_CLASS_CONFIG[label]
        lines.append(f"class-map match-any {cfg['class_name']}")
        lines.append(f" {match_prefix} {cfg['dscp']}")

    lines.append("policy-map QOS-OUT")
    for label in ordered:
        cfg = QOS_CLASS_CONFIG[label]
        lines.append(f" class {cfg['class_name']}")
        for policy_line in cfg["policy"]:
            lines.append(f"  {policy_line}")
    lines.append(" class class-default")
    lines.append("  fair-queue")
    lines.append("interface GigabitEthernet0/0")
    lines.append(" service-policy output QOS-OUT")

    return "\n".join(lines) + "\n"


def build_pt_cli(predicted_labels: list[str]) -> str:
    rules = build_qos_rules(predicted_labels, packet_tracer=True)
    if not rules:
        return ""

    header = (
        "! ============================================================\n"
        "! AUTO-GENERATED QoS CONFIG – Cisco Packet Tracer (2911)\n"
        "! Paste this block into Router CLI (enable mode)\n"
        "! ============================================================\n"
        "enable\n"
        "configure terminal\n"
        "no ip domain-lookup\n"
    )
    footer = "end\n"
    return header + rules + footer

def load_meta(model_path) -> dict:
    meta_path = str(model_path) + ".meta.json"
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def detect_target_column(columns: list[str]) -> str:
    for col in columns:
        if col.lower() in {"traffic_type", "target", "label", "class"}:
            return col
    return columns[-1]


def normalize_label(label: str) -> str:
    return label[4:] if label.startswith("VPN-") else label


def load_live_samples(
    dataset_path: Path,
    scaler_path: Path,
    sample_size: int,
    input_dim: int,
    logger,
) -> np.ndarray | None:
    if not dataset_path.exists():
        logger.warning("Live dataset not found: %s", dataset_path)
        return None

    df = pd.read_csv(dataset_path)
    if df.empty:
        logger.warning("Live dataset is empty: %s", dataset_path)
        return None

    target_col = detect_target_column(list(df.columns))
    if target_col not in df.columns:
        logger.warning("Target column not found in live dataset: %s", dataset_path)
        return None

    class_count = int(df[target_col].nunique())
    desired_size = max(sample_size, class_count * 2)
    desired_size = min(desired_size, len(df))
    if desired_size != sample_size:
        logger.info(
            "Live sampling size adjusted: requested=%d, classes=%d, using=%d",
            sample_size, class_count, desired_size,
        )

    samples: list[pd.DataFrame] = []
    used_idx: set[int] = set()
    for _, group in df.groupby(target_col):
        if group.empty:
            continue
        row = group.sample(n=1, replace=False)
        samples.append(row)
        used_idx.update(row.index.tolist())

    remaining = df.drop(index=used_idx) if used_idx else df
    extra_needed = max(0, desired_size - len(samples))
    if extra_needed > 0 and not remaining.empty:
        extra = remaining.sample(n=min(extra_needed, len(remaining)), replace=False)
        samples.append(extra)

    if not samples:
        logger.warning("No rows available for live sampling: %s", dataset_path)
        return None

    sample_df = pd.concat(samples, ignore_index=True)
    features = sample_df.drop(columns=[target_col])
    if features.shape[1] != input_dim:
        logger.warning(
            "Live dataset has %d features; expected %d. Skipping QoS generation.",
            features.shape[1], input_dim,
        )
        return None

    X_live = features.to_numpy(dtype=np.float32)

    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X_live = scaler.transform(X_live).astype(np.float32)
    else:
        logger.warning("Scaler not found (%s); using raw features.", scaler_path)

    return X_live

def predict_labels(
    model: torch.nn.Module,
    X_live: np.ndarray,
    device: torch.device,
    threshold_optimizer: ThresholdOptimizer | None,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_live).float().to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    if threshold_optimizer is not None:
        return threshold_optimizer.predict(probs)
    return np.argmax(probs, axis=1)

def main() -> None:
    cfg = parse_eval_args()
    logger = get_logger("evaluate", cfg.log_level)
    device = pick_device(cfg.device)

    data = SplitData(cfg.splits_dir).load()
    X_test = data["X_test"]
    y_test = data["y_test"]

    meta = load_meta(cfg.model_path)
    input_dim = int(meta.get("input_dim", X_test.shape[1]))
    num_classes = int(meta.get("num_classes", int(np.max(y_test)) + 1))

    model = FTTransformer(
        input_dim=input_dim,
        num_classes=num_classes,
        dim=int(meta.get("ft_dim", 128)),
        depth=int(meta.get("ft_depth", 4)),
        heads=int(meta.get("ft_heads", 8)),
        attn_dropout=float(meta.get("ft_attn_dropout", 0.1)),
        ff_dropout=float(meta.get("ft_ff_dropout", 0.1)),
        ff_mult=int(meta.get("ft_ff_mult", 4)),
        emb_dropout=float(meta.get("ft_emb_dropout", 0.1)),
        cls_dropout=float(meta.get("ft_cls_dropout", 0.1)),
    )
    model.load_state_dict(torch.load(cfg.model_path, map_location=device))

    test_loader = make_loader(X_test, y_test, cfg.batch_size, shuffle=False)
    trainer = Trainer(
        model,
        torch.optim.AdamW(model.parameters()),
        torch.nn.CrossEntropyLoss(),
        device,
        logger,
    )

    threshold_optimizer: ThresholdOptimizer | None = None
    if cfg.threshold_path.exists():
        threshold_optimizer = ThresholdOptimizer.load(cfg.threshold_path)
        adjusted = threshold_optimizer.summary()
        if adjusted:
            logger.info(
                "Threshold file loaded: %s — classes configured: %s",
                cfg.threshold_path,
                {k: f"{v:.3f}" for k, v in adjusted.items()},
            )
        else:
            logger.info("Threshold file loaded; uniform scale (argmax equivalent)")
    else:
        logger.info(
            "Threshold file not found (%s), using standard argmax",
            cfg.threshold_path,
        )

    metrics_raw = trainer.evaluate(test_loader, num_classes)
    logger.info(
        "Test (raw)  — acc=%.4f, macro_f1=%.4f",
        metrics_raw["accuracy"],
        metrics_raw["macro_f1"],
    )

    if threshold_optimizer is not None:
        metrics_cal = trainer.evaluate(
            test_loader, num_classes, threshold_optimizer=threshold_optimizer
        )
        logger.info(
            "Test (cal.) — acc=%.4f, macro_f1=%.4f",
            metrics_cal["accuracy"],
            metrics_cal["macro_f1"],
        )
        save_metrics({"test_raw": metrics_raw, "test": metrics_cal}, cfg.metrics_path)
    else:
        save_metrics({"test": metrics_raw}, cfg.metrics_path)

    logger.info("Evaluation metrics were recorded: %s", cfg.metrics_path)

    live_dataset_path = Path("data/processed/traffic_features_engineered.csv")
    scaler_path       = Path("saved_models/scaler.pkl")
    label_encoder_path = Path("saved_models/label_encoder.pkl")
    rules_path        = Path("qos_rules.txt")
    pt_rules_path     = Path("qos_rules_packet_tracer.txt")

    X_live = load_live_samples(
        live_dataset_path, scaler_path,
        sample_size=20, input_dim=input_dim, logger=logger,
    )
    if X_live is None:
        logger.warning("QoS rule generation skipped (live sample unavailable).")
        return

    if not label_encoder_path.exists():
        logger.warning("Label encoder not found: %s", label_encoder_path)
        return

    label_encoder = joblib.load(label_encoder_path)
    pred_indices  = predict_labels(model, X_live, device, threshold_optimizer)
    pred_labels   = label_encoder.inverse_transform(pred_indices)

    base_labels = [normalize_label(label) for label in pred_labels]
    unique, counts = np.unique(base_labels, return_counts=True)
    logger.info(
        "Live traffic predictions: %s",
        {label: int(count) for label, count in zip(unique, counts)},
    )

    rules_text = build_qos_rules(list(pred_labels), packet_tracer=False)
    if not rules_text:
        logger.warning("No QoS rules generated (unrecognized classes).")
        return
    rules_path.write_text(rules_text, encoding="utf-8")
    logger.info("QoS rules (standard IOS) → %s", rules_path)

    pt_cli = build_pt_cli(list(pred_labels))
    pt_rules_path.write_text(pt_cli, encoding="utf-8")
    logger.info("QoS rules (Packet Tracer CLI) → %s", pt_rules_path)


if __name__ == "__main__":
    main()
