"""Flower server for three-client RAVAN BERT FedAvg."""

import logging

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from task import configure_logging, load_central_test_data, make_model, test


NUM_CLIENTS = 3
app = ServerApp()
logger = logging.getLogger(__name__)

configure_logging()


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.debug("selected device=%s cuda_available=%s", device, torch.cuda.is_available())
    return device


@app.main()
def main(grid: Grid, context: Context):
    """Run FedAvg over three Flower clients."""
    logger.info(
        "starting Flower server num_clients=%s num_rounds=%s run_config=%s",
        NUM_CLIENTS,
        context.run_config["num-server-rounds"],
        context.run_config,
    )
    global_model = make_model()
    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=NUM_CLIENTS,
        min_evaluate_nodes=NUM_CLIENTS,
        min_available_nodes=NUM_CLIENTS,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(global_model.state_dict()),
        train_config=ConfigRecord({"lr": context.run_config["learning-rate"]}),
        num_rounds=context.run_config["num-server-rounds"],
        evaluate_fn=global_evaluate,
    )

    logger.info("finished federated training; saving model to federated_bert_final.pt")
    torch.save(result.arrays.to_torch_state_dict(), "federated_bert_final.pt")
    logger.info("saved model to federated_bert_final.pt")


def global_evaluate(server_round, arrays):
    """Evaluate the round model on the full centralized test split."""
    logger.info("starting server-side evaluation round=%s", server_round)
    model = make_model()
    model.load_state_dict(arrays.to_torch_state_dict())
    test_loader = load_central_test_data(batch_size=64)
    loss, accuracy = test(model, test_loader, get_device())
    logger.info(
        "finished server-side evaluation round=%s loss=%.4f accuracy=%.4f",
        server_round,
        loss,
        accuracy,
    )
    return MetricRecord({"loss": loss, "accuracy": accuracy})
