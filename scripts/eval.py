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


def normalize_label_key(raw_label_key):
	return raw_label_key.replace("-", "_")


def load_json(path):
	with open(path, "r", encoding="utf-8") as file:
		return json.load(file)


def multiclass_log_loss(probs, labels, eps=1e-12):
	probs = probs.clamp(min=eps, max=1.0 - eps)
	true_probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
	return float((-torch.log(true_probs)).mean().item())


def multiclass_brier(probs, labels, num_classes):
	one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
	return float(((probs - one_hot) ** 2).sum(dim=1).mean().item())


def accuracy_score(preds, labels):
	return float((preds == labels).float().mean().item())


def expected_calibration_error(probs, labels, num_bins=15):
	confidences, predictions = probs.max(dim=1)
	accuracies = predictions.eq(labels).float()

	ece = torch.zeros(1, device=probs.device)
	bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device)

	for bin_index in range(num_bins):
		lower = bin_boundaries[bin_index]
		upper = bin_boundaries[bin_index + 1]
		in_bin = (confidences > lower) & (confidences <= upper)
		prop_in_bin = in_bin.float().mean()
		if prop_in_bin.item() > 0:
			acc_in_bin = accuracies[in_bin].mean()
			conf_in_bin = confidences[in_bin].mean()
			ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin
	return float(ece.item())


def compute_metrics(logits, labels, num_classes):
	probs = torch.softmax(logits, dim=1)
	preds = probs.argmax(dim=1)
	return {
		"log_loss": multiclass_log_loss(probs, labels),
		"brier": multiclass_brier(probs, labels, num_classes=num_classes),
		"accuracy": accuracy_score(preds, labels),
		"ece": expected_calibration_error(probs, labels),
	}


def evaluate(model, data_loader, device, num_classes):
	model.eval()
	logits_list = []
	labels_list = []
	promoted_flags = []

	with torch.no_grad():
		for batch in data_loader:
			x, u, gt = preprocess(batch)
			x = x.to(device)
			u = u.to(device)
			gt = gt.to(device)

			output = model(x, u)
			logits = output[-1]

			logits_list.append(logits.detach())
			labels_list.append(gt.detach())
			promoted_flags.append(batch["promoted_match"].to(device))

	all_logits = torch.cat(logits_list, dim=0)
	all_labels = torch.cat(labels_list, dim=0)
	all_promoted = torch.cat(promoted_flags, dim=0).long()

	overall = compute_metrics(all_logits, all_labels, num_classes=num_classes)

	promoted_mask = all_promoted == 1
	non_promoted_mask = all_promoted == 0

	promoted_metrics = None
	if promoted_mask.any():
		promoted_metrics = compute_metrics(
			all_logits[promoted_mask],
			all_labels[promoted_mask],
			num_classes=num_classes,
		)

	non_promoted_metrics = None
	if non_promoted_mask.any():
		non_promoted_metrics = compute_metrics(
			all_logits[non_promoted_mask],
			all_labels[non_promoted_mask],
			num_classes=num_classes,
		)

	return {
		"overall": overall,
		"promoted_slice": {
			"count": int(promoted_mask.sum().item()),
			"metrics": promoted_metrics,
		},
		"non_promoted_slice": {
			"count": int(non_promoted_mask.sum().item()),
			"metrics": non_promoted_metrics,
		},
		"sample_count": int(all_labels.numel()),
	}


def parse_args(default_epoch):
	parser = argparse.ArgumentParser(description="Evaluate Transformer checkpoint on selected split")
	parser.add_argument(
		"--label-key",
		default="fulltime_label",
		type=normalize_label_key,
		choices=list(TASK_NUM_CLASSES.keys()),
		help="Target label to evaluate. Accepts either underscores or hyphens.",
	)
	parser.add_argument(
		"--split",
		default="validation",
		choices=["train", "validation", "test"],
		help="Dataset split to evaluate.",
	)
	parser.add_argument(
		"--checkpoint",
		default=None,
		help="Path to model checkpoint. If omitted, provide a file from train_params.output_dir (default ./output).",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=None,
		help="Override batch size for evaluation.",
	)
	parser.add_argument(
		"--output",
		default="data/processed/eval_report.json",
		help="Path to write JSON evaluation report.",
	)
	parser.add_argument(
		"--epochs",
		type=int,
		default=default_epoch,
		help="Unused placeholder to keep script interface aligned with train workflow.",
	)
	return parser.parse_args()


def main():
	top_config = load_json(os.path.join(root_full_path, "config/config.json"))
	data_config = load_json(os.path.join(root_full_path, "config/data_config.json"))

	args = parse_args(default_epoch=top_config["train_params"]["epoch"])
	batch_size = args.batch_size or top_config["train_params"]["batch_size"]
	num_classes = TASK_NUM_CLASSES[args.label_key]

	model = TopFormer(
		num_layers=top_config["model_params"]["num_layers"],
		d_model=top_config["model_params"]["d_model"],
		nhead=top_config["model_params"]["nhead"],
		num_classes=num_classes,
	)

	if not args.checkpoint:
		raise ValueError("--checkpoint is required for evaluation.")
	if not os.path.exists(args.checkpoint):
		raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

	device = "cuda" if torch.cuda.is_available() else "cpu"
	state = load_checkpoint_state_dict(args.checkpoint, device=device)
	model.load_state_dict(state)
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

	report = {
		"task": args.label_key,
		"split": args.split,
		"device": device,
		"checkpoint": args.checkpoint,
		"split_sample_count": len(split_samples.get(args.split, [])),
		"metrics": evaluate(model, loader, device=device, num_classes=num_classes),
	}

	output_path = os.path.join(root_full_path, args.output)
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	with open(output_path, "w", encoding="utf-8") as file:
		json.dump(report, file, ensure_ascii=False, indent=2)

	print(json.dumps(report, ensure_ascii=False, indent=2))
	print(f"Wrote eval report to: {output_path}")


if __name__ == "__main__":
	main()
