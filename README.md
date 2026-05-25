# Scalable Systems RAVAN

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Federated BERT

Run from inside the Flower app folder:

```bash
cd federated_bert
flwr run .
```

## Run RAVAN BERT

Run from the project root:

```bash
python bert_20newsgroups_ravan.py
```

## Performance Settings

Trick to make the clients run not in parallel. For laptops with limited memory.

```
[tool.flwr.federations.local-simulation.options.backend.client-resources]
num-cpus = 8
num-gpus = 0.0
```