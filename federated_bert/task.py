"""Classic BERT training utilities for the Flower app."""

import logging
import os
import time

import torch
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LOG_LEVEL_ENV = "FEDERATED_BERT_LOG_LEVEL"
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
NUM_LABELS = 20
MAX_LENGTH = 128
DATA_SEED = 42

logger = logging.getLogger(__name__)


def configure_logging():
    """Configure app logging from FEDERATED_BERT_LOG_LEVEL."""
    level_name = os.environ.get(LOG_LEVEL_ENV, "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT)
    logging.getLogger("federated_bert").setLevel(level)


def make_model():
    """Build a fresh BERT classifier whose full state is federated."""
    logger.info("loading model %s with %s labels", MODEL_NAME, NUM_LABELS)
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
    )


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
    """Train all BERT parameters on one client's local partition."""
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
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
