#!/usr/bin/env python
"""
Main entry point for training HQCT models.

Example usage:
  # Full-data HQCT-K on AG News
  python scripts/train.py --dataset ag_news --attn_type hqct-k --epochs 30

  # Few-shot (16 examples per class)
  python scripts/train.py --dataset ag_news --attn_type hqct-k --few_shot 16

  # Classical baseline
  python scripts/train.py --dataset imdb --attn_type classical

  # Ablation: different qubit counts
  python scripts/train.py --dataset ag_news --n_qubits 4
  python scripts/train.py --dataset ag_news --n_qubits 8
"""

import argparse
import os
import sys
import json
from pathlib import Path

import torch

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hqct.models import HQCT
from hqct.data import get_dataloaders
from hqct.train import train_model, evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Train Hybrid Quantum-Classical Transformer")
    p.add_argument("--dataset", type=str, default="ag_news",
                   choices=["ag_news", "imdb", "sst2", "multinli"])
    p.add_argument("--attn_type", type=str, default="hqct-k",
                   choices=["classical", "hqct-k", "hqct-p"])
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=512)
    p.add_argument("--n_qubits", type=int, default=6)
    p.add_argument("--q_layers", type=int, default=3)
    p.add_argument("--noise", type=float, default=0.0,
                   help="Depolarizing noise probability (0 = noiseless)")
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr_classical", type=float, default=1e-3)
    p.add_argument("--lr_quantum", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--few_shot", type=int, default=None,
                   help="If set, use only this many examples per class")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="runs")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Data
    max_len = args.max_seq_len
    if max_len is None:
        max_len = 256 if args.dataset == "imdb" else 128

    train_loader, val_loader, test_loader, num_classes, vocab_size = get_dataloaders(
        name=args.dataset,
        batch_size=args.batch_size,
        max_length=max_len,
        few_shot_k=args.few_shot,
        seed=args.seed,
    )
    print(f"Dataset: {args.dataset} | classes={num_classes} | vocab={vocab_size}")
    print(f"Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

    # Model
    model = HQCT(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=max_len,
        attn_type=args.attn_type,
        n_qubits=args.n_qubits,
        q_layers=args.q_layers,
        noise=args.noise,
    ).to(device)

    n_params = model.count_all_parameters()
    print(f"Model: {args.attn_type} | total parameters: {n_params:,}")
    print(f"Quantum config: qubits={args.n_qubits}, depth={args.q_layers}, noise={args.noise}")

    # Train
    result = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr_classical=args.lr_classical,
        lr_quantum=args.lr_quantum,
        patience=args.patience,
        seed=args.seed,
    )

    # Final test evaluation
    from torch.nn import CrossEntropyLoss
    test_loss, test_acc = evaluate(model, test_loader, CrossEntropyLoss(), device)
    print(f"\n=== Final Test Accuracy: {test_acc*100:.2f}% ===")
    print(f"Best validation accuracy: {result['best_val_acc']*100:.2f}%")

    # Save
    out_dir = Path(args.output_dir) / f"{args.dataset}_{args.attn_type}_q{args.n_qubits}"
    if args.few_shot:
        out_dir = out_dir.parent / (out_dir.name + f"_fs{args.few_shot}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_dir / "model.pt")
    metrics = {
        "dataset": args.dataset,
        "attn_type": args.attn_type,
        "n_qubits": args.n_qubits,
        "q_layers": args.q_layers,
        "noise": args.noise,
        "few_shot": args.few_shot,
        "test_acc": test_acc,
        "best_val_acc": result["best_val_acc"],
        "total_params": n_params,
        "history": result["history"],
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
