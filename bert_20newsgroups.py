from time import perf_counter

import torch
from sklearn.datasets import fetch_20newsgroups
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


model_name = "google/bert_uncased_L-2_H-128_A-2"
batch_size = 32
epochs = 20

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
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

if device.type == "cuda":
    torch.cuda.synchronize()
training_start = perf_counter()

for parameter in model.classifier.parameters():
    parameter.requires_grad = False

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
