"""Flower clients for RAVAN BERT fine-tuning."""

import logging

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from task import configure_logging, load_client_data, make_model, test, train


app = ClientApp()
logger = logging.getLogger(__name__)

configure_logging()


def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.debug("selected device=%s cuda_available=%s", device, torch.cuda.is_available())
    return device


@app.train()
def train_client(msg: Message, context: Context):
    """Train one client model on one 20 Newsgroups partition."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]
    learning_rate = msg.content["config"]["lr"]
    logger.info(
        "client train start partition_id=%s/%s batch_size=%s local_epochs=%s lr=%s",
        partition_id,
        num_partitions,
        batch_size,
        local_epochs,
        learning_rate,
    )

    model = make_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    train_loader, _ = load_client_data(partition_id, num_partitions, batch_size)

    train_loss = train(
        model,
        train_loader,
        local_epochs,
        learning_rate,
        get_device(),
    )
    logger.info(
        "client train finished partition_id=%s train_loss=%.4f examples=%s",
        partition_id,
        train_loss,
        len(train_loader.dataset),
    )
    content = RecordDict(
        {
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord(
                {
                    "train_loss": train_loss,
                    "num-examples": len(train_loader.dataset),
                }
            ),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate_client(msg: Message, context: Context):
    """Evaluate one global model copy on one client test partition."""
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    logger.info(
        "client eval start partition_id=%s/%s batch_size=%s",
        partition_id,
        num_partitions,
        batch_size,
    )

    model = make_model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    _, test_loader = load_client_data(partition_id, num_partitions, batch_size)

    loss, accuracy = test(model, test_loader, get_device())
    logger.info(
        "client eval finished partition_id=%s loss=%.4f accuracy=%.4f examples=%s",
        partition_id,
        loss,
        accuracy,
        len(test_loader.dataset),
    )
    content = RecordDict(
        {
            "metrics": MetricRecord(
                {
                    "eval_loss": loss,
                    "eval_accuracy": accuracy,
                    "num-examples": len(test_loader.dataset),
                }
            )
        }
    )
    return Message(content=content, reply_to=msg)
