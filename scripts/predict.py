"""Per-match probability inference for EV backtest.

Runs a trained checkpoint over a data split and writes one JSONL record per
match containing:
    - match_id
    - task
    - probs        : softmax probability for each class (list of floats)
    - predicted_label  : argmax class index
    - true_label   : ground-truth class index (-1 when unavailable)

Usage:
    python scripts/predict.py \\
        --label-key fulltime_label \\
        --checkpoint output/top_eleven-task=fulltime_label-....pt \\
        --split test \\
        --output output/predictions/fulltime_label_test.jsonl
"""

import argparse
import json
import os
import sys

import torch

cwd = os.getcwd()
cfp = os.path.dirname(os.path.abspath(__file__))
os.chdir(cfp)
root_full_path = os.path.abspath("..")
sys.path.append(root_full_path)
os.chdir(cwd)

from nn_modules.transformer.top_former import TopFormer
from data.data_loader import create_transformer_prematch_loader_for_split, preprocess
from utils.checkpoint import load_checkpoint_state_dict


TASK_NUM_CLASSES = {
    "fulltime_label": 3,
    "htft_label": 9,
    "handicap_label": 3,
}


def normalize_label_key(raw):
    return raw.replace("-", "_")


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_inference(model, data_loader, device):
    """Run forward pass over loader, return per-match probability records.

    Parameters
    ----------
    model : nn.Module
        Loaded and eval-mode transformer.
    data_loader : DataLoader
        Loader for a single split (shuffle=False preserves order).
    device : str
        "cuda" or "cpu".

    Returns
    -------
    list[dict]
        One dict per sample:
            match_id, probs (list[float]), predicted_label (int), true_label (int)
    """
    model.eval()
    records = []
    with torch.no_grad():
        for batch in data_loader:
            token_values, gt = preprocess(batch)
            token_values = token_values.to(device)
            gt = gt.to(device)

            logits = model(token_values)[-1]
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            match_ids = batch["match_id"]
            for i, mid in enumerate(match_ids):
                records.append(
                    {
                        "match_id": str(mid),
                        "probs": [round(float(p), 6) for p in probs[i].cpu()],
                        "predicted_label": int(preds[i].item()),
                        "true_label": int(gt[i].item()),
                    }
                )
    return records


def default_output_path(label_key, split):
    return os.path.join(root_full_path, "output", "predictions", f"{label_key}_{split}.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description="Per-match probability inference for EV backtest")
    parser.add_argument(
        "--label-key",
        default="fulltime_label",
        type=normalize_label_key,
        choices=list(TASK_NUM_CLASSES.keys()),
        help="Task to predict.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .pt checkpoint file.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to run inference on.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output JSONL. Defaults to output/predictions/<task>_<split>.jsonl",
    )
    return parser.parse_args()


def main():
    top_config = load_json(os.path.join(root_full_path, "config/config.json"))
    data_config = load_json(os.path.join(root_full_path, "config/data_config.json"))

    args = parse_args()
    batch_size = args.batch_size or top_config["train_params"]["batch_size"]
    num_classes = TASK_NUM_CLASSES[args.label_key]

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TopFormer(
        num_layers=top_config["model_params"]["num_layers"],
        d_model=top_config["model_params"]["d_model"],
        nhead=top_config["model_params"]["nhead"],
        num_classes=num_classes,
    )
    state = load_checkpoint_state_dict(args.checkpoint, device=device)
    load_result = model.load_state_dict(state, strict=False)
    if load_result.missing_keys:
        print(f"Warning: missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"Warning: unexpected keys: {load_result.unexpected_keys}")
    model.to(device)

    processed_dir = os.path.join(root_full_path, data_config["paths"]["processed_dir"])
    loader, split_samples = create_transformer_prematch_loader_for_split(
        processed_dir=processed_dir,
        season_split=data_config["season_split"],
        split_name=args.split,
        source_length=top_config["model_params"]["source_length"],
        target_length=top_config["model_params"]["target_length"],
        d_model=top_config["model_params"]["d_model"],
        batch_size=batch_size,
        label_key=args.label_key,
        shuffle=False,
    )

    print(
        f"Predicting: task={args.label_key}, split={args.split}, "
        f"samples={len(split_samples.get(args.split, []))}, device={device}"
    )
    records = run_inference(model, loader, device=device)

    output_path = args.output or default_output_path(args.label_key, args.split)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            r["task"] = args.label_key
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} prediction records → {output_path}")


if __name__ == "__main__":
    main()
