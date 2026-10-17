"""
Full HQCT experiment script.

This script mirrors the paper-style training workflow and is intended to be run
from the project root. It supports:
- classical baseline
- HQCT-P
- HQCT-K
- single-dataset or multi-dataset sweeps
- few-shot experiments
- training curve export and metrics JSON

Usage examples:
    python experiments/hqct_experiment.py --dataset ag_news --attn_type hqct-k
    python experiments/hqct_experiment.py --dataset ag_news --attn_type classical --epochs 10
    python experiments/hqct_experiment.py --dataset imdb --few_shot 16 --epochs 20
    python experiments/hqct_experiment.py --run_all
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hqct.models import HQCT
from hqct.data import get_dataloaders
from hqct.train import evaluate, train_model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    dataset: str,
    attn_type: str,
    vocab_size: int,
    num_classes: int,
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    d_ff: int = 512,
    max_seq_len: int = 128,
    dropout: float = 0.1,
    n_qubits: int = 6,
    q_layers: int = 3,
    noise: float = 0.0,
):
    return HQCT(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        attn_type=attn_type,
        n_qubits=n_qubits,
        q_layers=q_layers,
        noise=noise,
        pad_idx=0,
    )


def save_training_plot(history, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="train loss", color="tab:blue")
    axes[0].plot(history["val_loss"], label="val loss", color="tab:orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss curves")
    axes[0].legend()

    axes[1].plot(history["val_acc"], label="val acc", color="tab:green")
    if history["val_acc"]:
        axes[1].axhline(max(history["val_acc"]), color="red", linestyle="--", alpha=0.7,
                        label=f"best={max(history['val_acc']):.3f}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_single_experiment(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, num_classes, vocab_size = get_dataloaders(
        name=args.dataset,
        batch_size=args.batch_size,
        max_length=args.max_length,
        few_shot_k=args.few_shot,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = build_model(
        dataset=args.dataset,
        attn_type=args.attn_type,
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_length,
        dropout=args.dropout,
        n_qubits=args.n_qubits,
        q_layers=args.q_layers,
        noise=args.noise,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    train_result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr_classical=args.lr_classical,
        lr_quantum=args.lr_quantum,
        weight_decay_quantum=args.weight_decay_quantum,
        patience=args.patience,
        seed=args.seed,
    )

    test_loss, test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    run_name = f"{args.dataset}_{args.attn_type}_q{args.n_qubits}"
    output_dir = PROJECT_ROOT / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "dataset": args.dataset,
        "attn_type": args.attn_type,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "patience": args.patience,
        "few_shot": args.few_shot,
        "lr_classical": args.lr_classical,
        "lr_quantum": args.lr_quantum,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "n_qubits": args.n_qubits,
        "q_layers": args.q_layers,
        "noise": args.noise,
        "num_classes": num_classes,
        "vocab_size": vocab_size,
        "best_val_acc": float(train_result["best_val_acc"]),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "history": train_result["history"],
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_training_plot(train_result["history"], output_dir / "training_curves.png")

    print(f"\n=== Experiment Summary ===")
    print(f"Dataset: {args.dataset}")
    print(f"Attention: {args.attn_type}")
    print(f"Best validation accuracy: {train_result['best_val_acc'] * 100:.2f}%")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print(f"Saved results to: {output_dir}")

    return metrics


def run_all_experiments(args):
    configs = [
        {"dataset": "ag_news", "attn_type": "classical"},
        {"dataset": "ag_news", "attn_type": "hqct-p"},
        {"dataset": "ag_news", "attn_type": "hqct-k"},
    ]

    for config in configs:
        parser = argparse.Namespace(**vars(args))
        parser.dataset = config["dataset"]
        parser.attn_type = config["attn_type"]
        print(f"\nRunning: {parser.dataset} / {parser.attn_type}\n")
        run_single_experiment(parser)


def main():
    parser = argparse.ArgumentParser(description="Run HQCT experiments")
    parser.add_argument("--dataset", type=str, default="ag_news",
                        choices=["ag_news", "imdb", "sst2", "multinli"])
    parser.add_argument("--attn_type", type=str, default="hqct-k",
                        choices=["classical", "hqct-p", "hqct-k"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--few_shot", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr_classical", type=float, default=1e-3)
    parser.add_argument("--lr_quantum", type=float, default=5e-4)
    parser.add_argument("--weight_decay_quantum", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_qubits", type=int, default=6)
    parser.add_argument("--q_layers", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--run_all", action="store_true",
                        help="Run the standard comparison sweep on AG News for all attention types")

    args = parser.parse_args()

    if args.run_all:
        run_all_experiments(args)
        return

    run_single_experiment(args)


if __name__ == "__main__":
    main()

