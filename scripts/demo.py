#!/usr/bin/env python
"""
Quick smoke-test / demonstration of the HQCT models.

Runs a few forward passes on random data so you can verify that
the quantum circuits, hybrid attention, and classical baseline
all work without downloading large datasets.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from hqct.models import HQCT


def run_demo(attn_type: str, n_qubits: int = 2, seq_len: int = 4, batch: int = 1, q_layers: int = 1):
    print(f"\n{'='*60}")
    print(f"Demo: {attn_type} | qubits={n_qubits} | seq_len={seq_len} | depth={q_layers}")
    print("=" * 60)

    model = HQCT(
        vocab_size=500,
        num_classes=4,
        d_model=32,          # tiny for speed
        n_layers=1,
        n_heads=2,
        d_ff=64,
        max_seq_len=seq_len,
        attn_type=attn_type,
        n_qubits=n_qubits,
        q_layers=q_layers,
        noise=0.0,
    )

    input_ids = torch.randint(1, 500, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len, dtype=torch.long)

    # Forward
    logits = model(input_ids, attention_mask)
    print(f"Input shape : {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Logits sample:\n{logits.detach()}")

    # Backward (check gradients flow)
    loss = logits.sum()
    loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"Gradients flow: {has_grad}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
    return model


if __name__ == "__main__":
    torch.manual_seed(0)

    # Classical baseline (fast)
    run_demo("classical", n_qubits=2, seq_len=8)

    # HQCT-P (quantum projection)
    run_demo("hqct-p", n_qubits=2, seq_len=4, q_layers=1)

    # HQCT-K is O(N²) circuit evaluations – keep sequences very short
    print("\nNote: HQCT-K evaluates a quantum circuit for every query-key pair.")
    print("Using minimal sequences for the demo.")
    run_demo("hqct-k", n_qubits=2, seq_len=3, q_layers=1)

    print("\n[OK] All demos completed successfully.")
