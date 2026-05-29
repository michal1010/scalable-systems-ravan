from time import perf_counter

import torch
from torch import nn
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


model_name = "google/bert_uncased_L-2_H-128_A-2"
batch_size = 32
epochs = 5
ravan_heads = 4
ravan_rank = 8


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

        # B and A are frozen random bases. H starts at zero so the initial
        # adapter update is zero and the pretrained layer output is unchanged.
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

    # A new task head still has to learn the 20 Newsgroups labels.
    # for parameter in model.classifier.parameters():
    #     parameter.requires_grad = True


def print_trainable_parameters(model):
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"trainable parameters: {trainable:,}/{total:,} ({100 * trainable / total:.2f}%)")

def main():
    train_data = fetch_20newsgroups(
        subset="train", remove=("headers", "footers", "quotes")
    )
    test_data = fetch_20newsgroups(
        subset="test", remove=("headers", "footers", "quotes")
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_tokens = tokenizer(
        train_data.data,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    test_tokens = tokenizer(
        test_data.data,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    train_dataset = TensorDataset(
        train_tokens["input_ids"],
        train_tokens["attention_mask"],
        torch.tensor(train_data.target),
    )
    test_dataset = TensorDataset(
        test_tokens["input_ids"],
        test_tokens["attention_mask"],
        torch.tensor(test_data.target),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=20
    )
    add_ravan_to_bert_attention(model, ravan_heads, ravan_rank)
    model = model.to(device)
    print_trainable_parameters(model)

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_start = perf_counter()

    for epoch in range(epochs):
        model.train()
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

        print(f"finished epoch {epoch + 1}")

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_time = perf_counter() - training_start
    print(f"training time: {training_time:.2f} seconds")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = output.logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f"test accuracy: {correct / total:.3f}")


if __name__ == "__main__":
    main()
