import argparse
import json
import math
import os
import sys
import time

import torch
from torch import nn, optim
from torch.optim import Adam

top_config = {}
data_config = {}
cwd = os.getcwd()
cfp = os.path.dirname(os.path.abspath(__file__))
os.chdir(cfp)
root_full_path = os.path.abspath("..")
sys.path.append(root_full_path)

from nn_modules.transformer.top_former import TopFormer
from data.data_loader import create_transformer_prematch_data_loaders, preprocess
from utils.helper import count_parameters, initialize_weights, epoch_time

TASK_NUM_CLASSES = {
    "fulltime_label": 3,
    "htft_label": 9,
    "handicap_label": 3,
}


def normalize_label_key(raw_label_key: str) -> str:
    return raw_label_key.replace("-", "_")

config_full_path = os.path.join(root_full_path, "config/config.json")
if not os.path.exists(config_full_path):
    print(f"{config_full_path} doesn't exist.")
    exit(0)
with open(config_full_path, "r", encoding="utf-8") as file:
    top_config = json.load(file)

data_config_full_path = os.path.join(root_full_path, "config/data_config.json")
if not os.path.exists(data_config_full_path):
    print(f"{data_config_full_path} doesn't exist.")
    exit(0)
with open(data_config_full_path, "r", encoding="utf-8") as file:
    data_config = json.load(file)
os.chdir(cwd)

task_label_key = "fulltime_label"
model = None
optimizer = None
scheduler = None
loss = None
train_data_loader = None
validation_data_loader = None
split_samples = None


def get_checkpoint_output_dir():
    train_params = top_config.get("train_params", {})
    return train_params.get("output_dir", train_params.get("save_path", "./output"))


def build_checkpoint_filename(label_key: str, epoch_num: int, validation_loss: float, run_timestamp: str) -> str:
    return (
        f"top_eleven-task={label_key}-epoch={epoch_num:03d}"
        f"-val_loss={validation_loss:.4f}-ts={run_timestamp}.pt"
    )


def build_checkpoint_metadata(
    checkpoint_path: str,
    label_key: str,
    epoch_num: int,
    validation_loss: float,
    run_timestamp: str,
    device: str,
):
    train_params = top_config.get("train_params", {})
    optimize_params = train_params.get("optimize_params", {})
    return {
        "checkpoint_path": checkpoint_path,
        "task": label_key,
        "epoch": epoch_num,
        "validation_loss": float(validation_loss),
        "timestamp": run_timestamp,
        "device": device,
        "sample_counts": {
            "train": len(split_samples.get("train", [])),
            "validation": len(split_samples.get("validation", [])),
        },
        "train_params": {
            "batch_size": train_params.get("batch_size"),
            "warmup": train_params.get("warmup"),
            "learning_rate": optimize_params.get("lr"),
            "weight_decay": optimize_params.get("weight_decay"),
            "eps": optimize_params.get("eps"),
        },
        "model_params": top_config.get("model_params", {}),
        "label_mapping": TASK_NUM_CLASSES,
    }


def write_checkpoint_sidecar_json(checkpoint_path: str, metadata: dict):
    sidecar_path = os.path.splitext(checkpoint_path)[0] + ".json"
    with open(sidecar_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return sidecar_path


def configure_for_task(label_key: str):
    global task_label_key, model, optimizer, scheduler, loss, train_data_loader, validation_data_loader, split_samples

    label_key = normalize_label_key(label_key)

    if label_key not in TASK_NUM_CLASSES:
        raise ValueError(f"Unknown label_key={label_key}. Available: {list(TASK_NUM_CLASSES.keys())}")

    task_label_key = label_key
    processed_dir = os.path.join(root_full_path, data_config["paths"]["processed_dir"])
    train_data_loader, validation_data_loader, split_samples = create_transformer_prematch_data_loaders(
        processed_dir=processed_dir,
        season_split=data_config["season_split"],
        source_length=top_config["model_params"]["source_length"],
        target_length=top_config["model_params"]["target_length"],
        d_model=top_config["model_params"]["d_model"],
        batch_size=top_config["train_params"]["batch_size"],
        label_key=label_key,
    )

    if len(split_samples.get("train", [])) == 0 or len(split_samples.get("validation", [])) == 0:
        raise ValueError(
            "Empty train/validation split. Check config/data_config.json season_split and processed dataset."
        )

    model = TopFormer(
        num_layers=top_config["model_params"]["num_layers"],
        d_model=top_config["model_params"]["d_model"],
        nhead=top_config["model_params"]["nhead"],
        num_classes=TASK_NUM_CLASSES[label_key],
    )
    print(f"The model has {count_parameters(model)}: trainable parameters")
    model.apply(initialize_weights)

    optimizer = Adam(
        params=model.parameters(),
        lr=top_config["train_params"]["optimize_params"]["lr"],
        weight_decay=top_config["train_params"]["optimize_params"]["weight_decay"],
        eps=top_config["train_params"]["optimize_params"]["eps"],
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer)
    loss = nn.CrossEntropyLoss()


def train(model, data_loader, optimizer, loss_function, device='cuda'):
    """The interface to train the model using training data.

    Parameters
    ----------
    model : nn.Module
        The model to be trained.
    data_loader : DataLoader
        The interface for accessing training data and its corresponding labels 
        as the ground truth.
    optimizer : Adam
        The optimization algorithm for updating model parameters.
    loss_function : Loss nn.CrossEntropyLoss
        The loss function as objective and constraint for learning model parameters.
    device : str, optional
        The computing device for training the model (default: 'cuda').

    Returns
    -------
    float
        The loss averaged over all training samples in an epoch.
    """
    model.train()
    epoch_loss = 0
    model.to(device)
    for i, batch in enumerate(data_loader):
        token_values, gt = preprocess(batch)
        token_values = token_values.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()
        output = model(token_values)
        logits = output[-1]
        loss = loss_function(logits, gt)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # TODO(diwei): use other progressbar
        print('step :', round((i / (len(data_loader) + + sys.float_info.epsilon)) * 100, 2),
              '% , loss :', loss.item())

    return epoch_loss / len(data_loader)


def validate(model, data_loader, loss_function, device='cuda'):
    """The interface to validate the model using validation data.

    Parameters
    ----------
    model : nn.Module
        The model to be validated.
    data_loader : DataLoader
        The interface for accessing training data and its corresponding labels 
        as the ground truth.
    loss_function : Loss nn.CrossEntropyLoss
        The loss function as a criterion for model validation.
    device : str, optional
        The computing device for validating the model (default: 'cuda').

    Returns
    -------
    float
        The loss averaged over all training samples in an epoch.
    """
    model.eval()
    epoch_loss = 0
    model.to(device)
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            token_values, gt = preprocess(batch)
            token_values = token_values.to(device)
            gt = gt.to(device)
            output = model(token_values)

            logits = output[-1]
            loss = loss_function(logits, gt)
            epoch_loss += loss.item()

    return epoch_loss / len(data_loader)


def run(total_epoch: int, best_loss: float):
    """The interface to run the training process.

    Parameters
    ----------
    total_epoch : int
        The number of iterations.
    best_loss : float
        The minimum bias between the ground truth and the predicted value.
    """
    train_losses, validation_losses = [], []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir = get_checkpoint_output_dir()
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    warmup_epochs = int(top_config["train_params"].get("warmup", 0))
    print(
        f"Training task={task_label_key} on device={device}, train_samples={len(split_samples['train'])}, "
        f"validation_samples={len(split_samples['validation'])}"
    )
    for step in range(total_epoch):
        start_time = time.time()
        train_loss = train(model, train_data_loader, optimizer, loss, device=device)
        validation_loss = validate(model, validation_data_loader, loss, device=device)
        end_time = time.time()

        if warmup_epochs <= 0 or (step + 1) > warmup_epochs:
            scheduler.step(validation_loss)

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if validation_loss < best_loss:
            best_loss = validation_loss
            os.makedirs(output_dir, exist_ok=True)
            checkpoint_file = build_checkpoint_filename(
                label_key=task_label_key,
                epoch_num=step + 1,
                validation_loss=validation_loss,
                run_timestamp=run_timestamp,
            )
            checkpoint_path = os.path.join(output_dir, checkpoint_file)
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_metadata = build_checkpoint_metadata(
                checkpoint_path=checkpoint_path,
                label_key=task_label_key,
                epoch_num=step + 1,
                validation_loss=validation_loss,
                run_timestamp=run_timestamp,
                device=device,
            )
            sidecar_path = write_checkpoint_sidecar_json(checkpoint_path, checkpoint_metadata)
            print(f"Saved checkpoint: {checkpoint_path}")
            print(f"Saved checkpoint metadata: {sidecar_path}")

        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(
            f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(
            f'\tVal Loss: {validation_loss:.3f} |  Val PPL: {math.exp(validation_loss):7.3f}')


def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer baseline on processed pre-match dataset")
    parser.add_argument(
        "--label-key",
        default="fulltime_label",
        type=normalize_label_key,
        choices=list(TASK_NUM_CLASSES.keys()),
        help="Target label to train. Accepts either underscores or hyphens.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=top_config["train_params"]["epoch"],
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Optional learning rate override for this run.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Optional warmup epoch override for this run.",
    )
    return parser.parse_args()


def apply_runtime_overrides(args):
    if args.lr is not None:
        top_config["train_params"]["optimize_params"]["lr"] = float(args.lr)
    if args.warmup is not None:
        top_config["train_params"]["warmup"] = int(args.warmup)


# keep module import behavior compatible for smoke tests
configure_for_task("fulltime_label")


if __name__ == '__main__':
    args = parse_args()
    apply_runtime_overrides(args)
    configure_for_task(args.label_key)
    run(total_epoch=args.epochs, best_loss=float('inf'))
