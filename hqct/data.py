"""
Dataset loading and preprocessing for the four benchmarks used in the paper:
AG News, IMDB, SST-2, MultiNLI.
"""

from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np


DATASET_REPO_ALIASES = {
    "ag_news": "fancyzhx/ag_news",
    "imdb": "stanfordnlp/imdb",
    "multi_nli": "nyu-mll/multi_nli",
}


def load_dataset_with_fallback(name: str, subset: Optional[str] = None):
    """Support both current and legacy Hugging Face dataset identifiers."""
    candidates = [DATASET_REPO_ALIASES.get(name, name)]
    if name not in DATASET_REPO_ALIASES:
        candidates = [name]
    else:
        candidates = [DATASET_REPO_ALIASES[name], name]

    last_error = None
    for repo in candidates:
        try:
            if subset is None:
                return load_dataset(repo)
            return load_dataset(repo, subset)
        except Exception as exc:  # pragma: no cover - fallback selector for dataset aliases
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError(f"Unable to load dataset: {name}")


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_benchmark(
    name: str,
    tokenizer_name: str = "bert-base-uncased",
    max_length: Optional[int] = None,
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset, int]:
    """
    Returns train, val, test datasets and number of classes.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    name = name.lower()

    if name == "ag_news":
        ds = load_dataset_with_fallback("ag_news")
        train_texts = ds["train"]["text"]
        train_labels = ds["train"]["label"]
        test_texts = ds["test"]["text"]
        test_labels = ds["test"]["label"]
        num_classes = 4
        max_length = max_length or 128

    elif name == "imdb":
        ds = load_dataset_with_fallback("imdb")
        train_texts = ds["train"]["text"]
        train_labels = ds["train"]["label"]
        test_texts = ds["test"]["text"]
        test_labels = ds["test"]["label"]
        num_classes = 2
        max_length = max_length or 256

    elif name == "sst2":
        ds = load_dataset("glue", "sst2")
        train_texts = ds["train"]["sentence"]
        train_labels = ds["train"]["label"]
        # GLUE SST-2 has no public test labels; use validation as test
        test_texts = ds["validation"]["sentence"]
        test_labels = ds["validation"]["label"]
        num_classes = 2
        max_length = max_length or 128

    elif name == "multinli":
        ds = load_dataset_with_fallback("multi_nli")
        # Use matched validation as test (paper reports matched)
        train_texts = [
            p + " [SEP] " + h
            for p, h in zip(ds["train"]["premise"], ds["train"]["hypothesis"])
        ]
        train_labels = ds["train"]["label"]
        # filter out -1 labels if any
        valid_idx = [i for i, y in enumerate(train_labels) if y != -1]
        train_texts = [train_texts[i] for i in valid_idx]
        train_labels = [train_labels[i] for i in valid_idx]

        val = ds["validation_matched"]
        test_texts = [
            p + " [SEP] " + h for p, h in zip(val["premise"], val["hypothesis"])
        ]
        test_labels = val["label"]
        num_classes = 3
        max_length = max_length or 128

    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Hold out validation set from training data
    rng = np.random.RandomState(seed)
    n = len(train_texts)
    indices = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_ds = TextClassificationDataset(
        [train_texts[i] for i in train_idx],
        [train_labels[i] for i in train_idx],
        tokenizer,
        max_length,
    )
    val_ds = TextClassificationDataset(
        [train_texts[i] for i in val_idx],
        [train_labels[i] for i in val_idx],
        tokenizer,
        max_length,
    )
    test_ds = TextClassificationDataset(
        test_texts, test_labels, tokenizer, max_length
    )

    return train_ds, val_ds, test_ds, num_classes, tokenizer.vocab_size


def make_few_shot(
    dataset: Dataset,
    num_classes: int,
    k: int = 16,
    seed: int = 42,
) -> Dataset:
    """
    Construct a few-shot subset with exactly k examples per class.
    """
    rng = np.random.RandomState(seed)
    labels = [dataset[i]["labels"].item() for i in range(len(dataset))]
    indices = []
    for c in range(num_classes):
        cls_idx = [i for i, y in enumerate(labels) if y == c]
        chosen = rng.choice(cls_idx, size=min(k, len(cls_idx)), replace=False)
        indices.extend(chosen.tolist())
    return Subset(dataset, indices)


def get_dataloaders(
    name: str,
    batch_size: int = 32,
    max_length: Optional[int] = None,
    few_shot_k: Optional[int] = None,
    num_workers: int = 2,
    seed: int = 42,
):
    train_ds, val_ds, test_ds, num_classes, vocab_size = load_benchmark(
        name, max_length=max_length, seed=seed
    )

    if few_shot_k is not None:
        train_ds = make_few_shot(train_ds, num_classes, k=few_shot_k, seed=seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader, num_classes, vocab_size
