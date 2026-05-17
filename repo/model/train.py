import os
import sys
import argparse
import logging
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from utils.config import load_config, get_model_config
from utils.dataset import load_dataset
from utils.metrics import compute_all_metrics
from utils.tokenizer import DateTokenizer, DECADE_MIN, DECADE_MAX
from utils.validators import validate_all
from models import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a date-generation model.")
    p.add_argument("--model", default="gan",
                   choices=["gan", "vae", "transformer", "diffusion"])
    p.add_argument("--data", default="../data/data.txt",
                   help="Path to data.txt")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epoch count from config.")
    p.add_argument("--config", default=None,
                   help="Path to config YAML (optional).")
    p.add_argument("--weights", default="./weights",
                   help="Directory to save trained weights.")
    return p.parse_args()


def set_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Seeds set to {seed}.")


def plot_losses(train_losses, val_losses, model_name, save_dir="./plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", linewidth=2, marker='o', markersize=4)
    plt.plot(val_losses, label="Validation Loss", linewidth=2, marker='s', markersize=4)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(f"{model_name.upper()} - Training Curve", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/{model_name}_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Loss plot saved: {save_dir}/{model_name}_loss.png")


def plot_metrics(metrics, model_name, save_dir="./plots"):
    os.makedirs(save_dir, exist_ok=True)
    plot_metrics_dict = {
        "Validity": metrics.get("validity_rate", 0),
        "Weekday": metrics.get("weekday_match", 0),
        "Month": metrics.get("month_match", 0),
        "Leap": metrics.get("leap_match", 0),
        "Decade": metrics.get("decade_match", 0),
        "Diversity": metrics.get("diversity_score", 0),
    }
    plt.figure(figsize=(12, 6))
    bars = plt.bar(plot_metrics_dict.keys(), plot_metrics_dict.values(),
                   color='steelblue', edgecolor='black', alpha=0.8)
    plt.ylim(0, 1.05)
    plt.ylabel("Score", fontsize=12)
    plt.title(f"{model_name.upper()} - Evaluation Metrics", fontsize=14)
    for bar, val in zip(bars, plot_metrics_dict.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    plt.savefig(f"{save_dir}/{model_name}_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Metrics plot saved: {save_dir}/{model_name}_metrics.png")


def plot_all_models_comparison(all_metrics, save_dir="./plots"):
    os.makedirs(save_dir, exist_ok=True)
    models = list(all_metrics.keys())
    metrics_to_plot = ["validity_rate", "weekday_match", "month_match", "leap_match", "decade_match", "diversity_score"]
    metric_labels = ["Validity", "Weekday", "Month", "Leap", "Decade", "Diversity"]
    x = range(len(metrics_to_plot))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, model in enumerate(models):
        values = [all_metrics[model].get(m, 0) for m in metrics_to_plot]
        offset = (i - len(models)/2) * width
        bars = ax.bar([p + offset for p in x], values, width, label=model.upper())
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('All Models Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/all_models_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Comparison plot saved: {save_dir}/all_models_comparison.png")


def evaluate_model(model, val_batches, tokenizer, n_samples=200):
    generated, conditions_raw = [], []
    for cond_batch, _ in val_batches:
        for cond_vec in cond_batch:
            if len(generated) >= n_samples:
                break
            raw = model.generate(cond_vec)
            d, m, y = tokenizer.decode_date(raw)
            generated.append((d, m, y))
            ds, ms, leap, decade = tokenizer.decode_condition(cond_vec)
            conditions_raw.append((ds, ms, str(leap), str(decade)))
        if len(generated) >= n_samples:
            break
    return compute_all_metrics(generated, conditions_raw)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seeds(config.get("seed", 42))
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    epochs = config["training"]["epochs"]
    batch_size = config["training"]["batch_size"]
    logger.info(f"Model: {args.model}  |  Epochs: {epochs}  |  Batch: {batch_size}")
    train_batches, val_batches, test_batches, tokenizer = load_dataset(
        data_path=args.data,
        train_split=config["data"].get("train_split", 0.8),
        val_split=config["data"].get("val_split", 0.1),
        batch_size=batch_size,
        seed=config.get("seed", 42),
    )
    model_cfg = get_model_config(config, args.model)
    model_cfg["condition_dim"] = tokenizer.CONDITION_DIM
    model = get_model(args.model, model_cfg)
    os.makedirs(args.weights, exist_ok=True)
    best_validity = -1.0
    train_losses = []
    val_metrics_history = []
    for epoch in range(1, epochs + 1):
        if args.model == "gan":
            d_loss, g_loss = model.train_epoch(train_batches)
            train_loss = g_loss
            train_info = f"D_loss={d_loss:.4f}  G_loss={g_loss:.4f}"
        elif args.model == "vae":
            total, mse, kl = model.train_epoch(train_batches)
            train_loss = total
            train_info = f"Loss={total:.4f}  MSE={mse:.4f}  KL={kl:.4f}"
        elif args.model == "transformer":
            train_loss = model.train_epoch(train_batches)
            train_info = f"CE_loss={train_loss:.4f}"
        else:
            train_loss = model.train_epoch(train_batches)
            train_info = f"Noise_MSE={train_loss:.4f}"
        train_losses.append(train_loss)
        if epoch % 10 == 0 or epoch == 1:
            metrics = evaluate_model(model, val_batches, tokenizer)
            val_metrics_history.append(metrics)
            val_str = "  ".join(f"{k}={v:.3f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch:4d}/{epochs}  {train_info}  ||  {val_str}")
            if metrics["validity_rate"] > best_validity:
                best_validity = metrics["validity_rate"]
                model.save(args.weights)
                logger.info(f"  → Best model saved (validity={best_validity:.4f})")
        else:
            logger.info(f"Epoch {epoch:4d}/{epochs}  {train_info}")
    logger.info("=" * 60)
    logger.info("Final evaluation on TEST set:")
    test_metrics = evaluate_model(model, test_batches, tokenizer, n_samples=500)
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 60)
    val_validities = [m["validity_rate"] for m in val_metrics_history]
    plot_losses(train_losses, val_validities, args.model)
    plot_metrics(test_metrics, args.model)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()