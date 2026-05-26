"""RAVAN BERT training utilities for the Flower app."""

import logging
import os
import time

import torch
from sklearn.datasets import fetch_20newsgroups
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LOG_LEVEL_ENV = "FEDERATED_BERT_LOG_LEVEL"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
NUM_LABELS = 20
MAX_LENGTH = 128
DATA_SEED = 42
RAVAN_HEADS = 4
RAVAN_RANK = 8

logger = logging.getLogger(__name__)


def configure_logging():
    """Configure app logging from FEDERATED_BERT_LOG_LEVEL."""
    level_name = os.environ.get(LOG_LEVEL_ENV, "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT)
    logging.getLogger("federated_bert").setLevel(level)


class RavanLinear(nn.Module):
    """Frozen linear layer plus trainable RAVAN heads."""

    def __init__(self, linear, heads, rank):
        super().__init__()
        self.linear = linear
        self.heads = heads
        self.rank = rank

        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.register_buffer(
            "B",
            torch.randn(heads, linear.out_features, rank) / rank**0.5,
        )
        self.register_buffer(
            "A",
            torch.randn(heads, rank, linear.in_features) / rank**0.5,
        )
        self.H = nn.Parameter(torch.zeros(heads, rank, rank))
        self.scales = nn.Parameter(torch.ones(heads))

    def forward(self, x):
        base_output = self.linear(x)
        ravan_update = torch.zeros_like(base_output)

        for head in range(self.heads):
            projected = x @ self.A[head].transpose(0, 1)
            mixed = projected @ self.H[head].transpose(0, 1)
            head_update = mixed @ self.B[head].transpose(0, 1)
            ravan_update = ravan_update + self.scales[head] * head_update

        return base_output + ravan_update


def add_ravan_to_bert_attention(model, heads, rank):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for layer in model.bert.encoder.layer:
        attention = layer.attention.self
        attention.query = RavanLinear(attention.query, heads, rank)
        attention.value = RavanLinear(attention.value, heads, rank)

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def log_trainable_parameters(model):
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "trainable parameters: %s/%s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )


def make_model():
    """Build a fresh BERT classifier with federated RAVAN adapters."""
    logger.info(
        "loading RAVAN model %s labels=%s heads=%s rank=%s",
        MODEL_NAME,
        NUM_LABELS,
        RAVAN_HEADS,
        RAVAN_RANK,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )
    add_ravan_to_bert_attention(model, RAVAN_HEADS, RAVAN_RANK)
    log_trainable_parameters(model)
    return model


def load_client_data(partition_id, num_partitions, batch_size):
    """Load one deterministic train/test partition for one simulated client."""
    logger.info(
        "loading client data partition_id=%s num_partitions=%s batch_size=%s",
        partition_id,
        num_partitions,
        batch_size,
    )
    train_data = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes"),
    )
    test_data = fetch_20newsgroups(
        subset="test",
        remove=("headers", "footers", "quotes"),
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = make_partition_dataset(
        tokenizer,
        train_data.data,
        train_data.target,
        partition_id,
        num_partitions,
    )
    test_dataset = make_partition_dataset(
        tokenizer,
        test_data.data,
        test_data.target,
        partition_id,
        num_partitions,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    logger.info(
        "loaded client partition_id=%s train_examples=%s test_examples=%s",
        partition_id,
        len(train_dataset),
        len(test_dataset),
    )
    return train_loader, test_loader


def load_central_test_data(batch_size):
    """Load the full test split for server-side global model evaluation."""
    logger.info("loading central test data batch_size=%s", batch_size)
    test_data = fetch_20newsgroups(
        subset="test",
        remove=("headers", "footers", "quotes"),
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = make_dataset(tokenizer, test_data.data, test_data.target)
    logger.info("loaded central test data examples=%s", len(test_dataset))
    return DataLoader(test_dataset, batch_size=batch_size)


def make_partition_dataset(tokenizer, texts, labels, partition_id, num_partitions):
    generator = torch.Generator().manual_seed(DATA_SEED)
    shuffled_indices = torch.randperm(len(labels), generator=generator)
    partition_indices = shuffled_indices[partition_id::num_partitions].tolist()

    partition_texts = [texts[index] for index in partition_indices]
    partition_labels = [labels[index] for index in partition_indices]
    return make_dataset(tokenizer, partition_texts, partition_labels)


def make_dataset(tokenizer, texts, labels):
    logger.debug("tokenizing dataset examples=%s max_length=%s", len(labels), MAX_LENGTH)
    tokens = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return TensorDataset(
        tokens["input_ids"],
        tokens["attention_mask"],
        torch.tensor(labels),
    )


def train(model, train_loader, epochs, learning_rate, device):
    """Train the RAVAN adapters and classifier on one client's local partition."""
    logger.info(
        "starting local training examples=%s batches=%s epochs=%s lr=%s device=%s",
        len(train_loader.dataset),
        len(train_loader),
        epochs,
        learning_rate,
        device,
    )
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    total_loss = 0.0

    started_at = time.perf_counter()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            output.loss.backward()
            optimizer.step()
            total_loss += output.loss.item()
            epoch_loss += output.loss.item()

        logger.debug(
            "finished epoch=%s/%s mean_loss=%.4f",
            epoch + 1,
            epochs,
            epoch_loss / len(train_loader),
        )

    mean_loss = total_loss / (epochs * len(train_loader))
    logger.info(
        "finished local training mean_loss=%.4f duration_seconds=%.2f",
        mean_loss,
        time.perf_counter() - started_at,
    )
    return mean_loss


def test(model, data_loader, device):
    """Return mean loss and accuracy for a BERT classifier."""
    logger.info(
        "starting evaluation examples=%s batches=%s device=%s",
        len(data_loader.dataset),
        len(data_loader),
        device,
    )
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            predictions = output.logits.argmax(dim=1)

            total_loss += output.loss.item()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    mean_loss = total_loss / len(data_loader)
    accuracy = correct / total
    logger.info("finished evaluation loss=%.4f accuracy=%.4f", mean_loss, accuracy)
    return mean_loss, accuracy
