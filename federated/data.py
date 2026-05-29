"""Data loading and federated client splits for 20 Newsgroups.

Two split strategies:
  iid    — randomly shuffle all examples and divide evenly across clients.
  noniid — Dirichlet(alpha) split per class, creating label skew.
"""

import numpy as np
import torch
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizerFast

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 20
MAX_LENGTH = 128


# ---------------------------------------------------------------------------
# Raw data helpers
# ---------------------------------------------------------------------------

def load_20newsgroups():
    """Return (train, test) sklearn Bunch objects."""
    train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    test  = fetch_20newsgroups(subset="test",  remove=("headers", "footers", "quotes"))
    return train, test


def tokenize(texts, tokenizer):
    """Tokenize a list of strings; return (input_ids, attention_mask) tensors."""
    enc = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return enc["input_ids"], enc["attention_mask"]


# ---------------------------------------------------------------------------
# Split strategies
# ---------------------------------------------------------------------------

def iid_split(n: int, num_clients: int, seed: int) -> list[np.ndarray]:
    """Randomly shuffle n indices and divide evenly across clients."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return [arr for arr in np.array_split(idx, num_clients)]


def dirichlet_split(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
) -> list[np.ndarray]:
    """Non-IID split: for each class draw Dirichlet proportions over clients.

    alpha=0.3 creates strong label skew; alpha→∞ approaches IID.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    num_classes = int(labels.max()) + 1
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        cls_idx = np.where(labels == c)[0]
        rng.shuffle(cls_idx)

        props = rng.dirichlet(alpha * np.ones(num_clients))
        counts = (props * len(cls_idx)).astype(int)

        # Fix rounding so all samples are assigned
        diff = len(cls_idx) - counts.sum()
        for i in range(abs(diff)):
            counts[i % num_clients] += 1 if diff > 0 else -1

        start = 0
        for cid in range(num_clients):
            n = max(0, counts[cid])
            client_indices[cid].extend(cls_idx[start : start + n].tolist())
            start += n

    return [np.array(idx) for idx in client_indices]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_federated_loaders(
    split_type: str,
    num_clients: int,
    batch_size: int,
    seed: int = 42,
    alpha: float = 0.3,
):
    """Build DataLoaders for all clients and a central test loader.

    Returns:
        client_loaders : list of DataLoader, one per client
        test_loader    : DataLoader over the full test split
        tokenizer      : the DistilBertTokenizerFast used
    """
    print("Loading 20 Newsgroups...")
    train_data, test_data = load_20newsgroups()

    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    print(f"Tokenizing {len(train_data.data)} train examples (max_length={MAX_LENGTH})...")
    train_ids, train_mask = tokenize(train_data.data, tokenizer)

    print(f"Tokenizing {len(test_data.data)} test examples...")
    test_ids, test_mask = tokenize(test_data.data, tokenizer)

    train_labels = np.array(train_data.target)

    print(f"Creating {split_type.upper()} split — {num_clients} clients, seed={seed}" +
          (f", alpha={alpha}" if split_type == "noniid" else ""))

    if split_type == "iid":
        splits = iid_split(len(train_labels), num_clients, seed)
    elif split_type == "noniid":
        splits = dirichlet_split(train_labels, num_clients, alpha, seed)
    else:
        raise ValueError(f"Unknown split type '{split_type}'. Use 'iid' or 'noniid'.")

    client_loaders = []
    for cid, idx in enumerate(splits):
        ds = TensorDataset(
            train_ids[idx],
            train_mask[idx],
            torch.tensor(train_labels[idx], dtype=torch.long),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
        client_loaders.append(loader)

    # Print a sample of client sizes
    sizes = [len(s) for s in splits]
    print(f"  Client sizes — min={min(sizes)}, max={max(sizes)}, "
          f"mean={np.mean(sizes):.0f}, first3={sizes[:3]}")

    # Central test loader
    test_ds = TensorDataset(
        test_ids,
        test_mask,
        torch.tensor(test_data.target, dtype=torch.long),
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    print(f"Central test set: {len(test_data.target)} examples\n")

    return client_loaders, test_loader, tokenizer