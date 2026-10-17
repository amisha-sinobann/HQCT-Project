"""
Training loop, evaluation, and utilities for HQCT.
"""

from typing import Dict, Optional, Tuple
import time
import copy

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np


def separate_parameters(model: nn.Module):
    """Split parameters into classical and quantum groups for different LRs."""
    classical, quantum = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Quantum variational weights live inside QuantumKernel / QuantumProjection
        if "weights" in name and ("quantum" in name or "q_proj" in name or "k_proj" in name):
            quantum.append(p)
        else:
            classical.append(p)
    return classical, quantum


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in tqdm(loader, desc="train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for batch in tqdm(loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        n += labels.size(0)

    return total_loss / max(n, 1), correct / max(n, 1)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 30,
    lr_classical: float = 1e-3,
    lr_quantum: float = 5e-4,
    weight_decay_quantum: float = 1e-4,
    patience: int = 5,
    seed: int = 42,
) -> Dict:
    """
    Full training procedure matching the paper (Section 3.4 & 4.2).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    classical_params, quantum_params = separate_parameters(model)
    param_groups = [
        {"params": classical_params, "lr": lr_classical, "weight_decay": 0.0},
    ]
    if quantum_params:
        param_groups.append(
            {
                "params": quantum_params,
                "lr": lr_quantum,
                "weight_decay": weight_decay_quantum,
            }
        )

    optimizer = Adam(param_groups)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc*100:.2f}% | "
            f"time={elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_acc": best_val_acc,
        "history": history,
    }
