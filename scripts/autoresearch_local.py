import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
AUTORESEARCH_DIR = OUTPUT_DIR / "autoresearch"
REPORTS_DIR = AUTORESEARCH_DIR / "reports"
LEADERBOARD_JSONL = AUTORESEARCH_DIR / "leaderboard.jsonl"
LEADERBOARD_CSV = AUTORESEARCH_DIR / "leaderboard.csv"


def normalize_task_label(raw_task):
    return raw_task.strip().replace("-", "_")


def parse_list(raw_value, cast_type):
    values = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(cast_type(part))
    return values


def parse_tasks(raw_tasks):
    normalized = []
    seen = set()
    for task in parse_list(raw_tasks, str):
        normalized_task = normalize_task_label(task)
        if normalized_task in seen:
            continue
        seen.add(normalized_task)
        normalized.append(normalized_task)
    return normalized


def get_checkpoint_output_dir():
    config_path = ROOT / "config" / "config.json"
    if not config_path.exists():
        return OUTPUT_DIR

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    train_params = payload.get("train_params", {})
    raw_output_dir = train_params.get("output_dir", train_params.get("save_path", "./output"))
    output_dir = Path(raw_output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    return output_dir.resolve()


def run_command(command):
    print("\n$", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def collect_sidecars(task_label, sidecar_root_dir):
    sidecars = []
    pattern = f"top_eleven-task={task_label}-*.json"
    for path in sidecar_root_dir.rglob(pattern):
        sidecars.append(path.resolve())
    return set(sidecars)


def pick_best_new_sidecar(task_label, before_sidecars, sidecar_root_dir):
    after_sidecars = collect_sidecars(task_label, sidecar_root_dir)
    new_sidecars = sorted(after_sidecars - before_sidecars)
    if not new_sidecars:
        raise RuntimeError(f"No new sidecar files found for task={task_label} after training run.")

    best_payload = None
    for sidecar_path in new_sidecars:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if best_payload is None:
            best_payload = payload
            continue
        if payload["validation_loss"] < best_payload["validation_loss"]:
            best_payload = payload
        elif payload["validation_loss"] == best_payload["validation_loss"] and payload["epoch"] > best_payload["epoch"]:
            best_payload = payload

    return best_payload


def run_eval(task_label, checkpoint_path, split_name, report_path):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "eval.py"),
        "--label-key",
        task_label,
        "--checkpoint",
        checkpoint_path,
        "--split",
        split_name,
        "--output",
        str(report_path.relative_to(ROOT)),
    ]
    run_command(command)
    return json.loads(report_path.read_text(encoding="utf-8"))


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = [
        "rank",
        "run_id",
        "task",
        "lr",
        "warmup",
        "epochs",
        "checkpoint",
        "val_log_loss",
        "val_brier",
        "val_accuracy",
        "val_ece",
        "test_log_loss",
        "test_brier",
        "test_accuracy",
        "test_ece",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def sort_rows(rows):
    return sorted(
        rows,
        key=lambda row: (
            row["val_log_loss"],
            row["val_brier"],
            row["val_ece"],
        ),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal local autoresearch-like loop for top-eleven")
    parser.add_argument(
        "--tasks",
        default="fulltime_label,htft_label,handicap_label",
        help="Comma-separated label keys.",
    )
    parser.add_argument(
        "--lrs",
        default="0.0003",
        help="Comma-separated learning rates.",
    )
    parser.add_argument(
        "--warmups",
        default="2",
        help="Comma-separated warmup epoch counts.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Epochs per training run.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Optional cap on total runs. 0 means no cap.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = parse_tasks(args.tasks)
    lrs = parse_list(args.lrs, float)
    warmups = parse_list(args.warmups, int)
    checkpoint_output_dir = get_checkpoint_output_dir()

    AUTORESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(LEADERBOARD_JSONL)
    run_counter = 0

    for task in tasks:
        for lr in lrs:
            for warmup in warmups:
                run_counter += 1
                if args.max_runs > 0 and run_counter > args.max_runs:
                    break

                run_id = time.strftime("%Y%m%d-%H%M%S") + f"-r{run_counter:03d}"
                print(f"\n=== Run {run_counter}: task={task}, lr={lr}, warmup={warmup}, epochs={args.epochs} ===")

                before_sidecars = collect_sidecars(task, checkpoint_output_dir)
                train_command = [
                    sys.executable,
                    str(ROOT / "scripts" / "train.py"),
                    "--label-key",
                    task,
                    "--epochs",
                    str(args.epochs),
                    "--lr",
                    str(lr),
                    "--warmup",
                    str(warmup),
                ]
                run_command(train_command)

                best_sidecar = pick_best_new_sidecar(task, before_sidecars, checkpoint_output_dir)
                checkpoint_path = best_sidecar["checkpoint_path"]

                val_report_path = REPORTS_DIR / f"{run_id}-{task}-validation.json"
                test_report_path = REPORTS_DIR / f"{run_id}-{task}-test.json"

                validation_report = run_eval(task, checkpoint_path, "validation", val_report_path)
                test_report = run_eval(task, checkpoint_path, "test", test_report_path)

                val_metrics = validation_report["metrics"]["overall"]
                test_metrics = test_report["metrics"]["overall"]

                row = {
                    "run_id": run_id,
                    "task": task,
                    "lr": lr,
                    "warmup": warmup,
                    "epochs": args.epochs,
                    "checkpoint": checkpoint_path,
                    "val_log_loss": val_metrics["log_loss"],
                    "val_brier": val_metrics["brier"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_ece": val_metrics["ece"],
                    "test_log_loss": test_metrics["log_loss"],
                    "test_brier": test_metrics["brier"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_ece": test_metrics["ece"],
                }
                rows.append(row)

                ranked_rows = sort_rows(rows)
                for index, ranked_row in enumerate(ranked_rows, start=1):
                    ranked_row["rank"] = index
                write_csv(LEADERBOARD_CSV, ranked_rows)
                write_jsonl(LEADERBOARD_JSONL, ranked_rows)

                best = ranked_rows[0]
                print(
                    "Current best:",
                    f"task={best['task']}",
                    f"lr={best['lr']}",
                    f"warmup={best['warmup']}",
                    f"val_log_loss={best['val_log_loss']:.4f}",
                    f"test_log_loss={best['test_log_loss']:.4f}",
                )

            if args.max_runs > 0 and run_counter >= args.max_runs:
                break
        if args.max_runs > 0 and run_counter >= args.max_runs:
            break

    ranked_rows = sort_rows(rows)
    for index, ranked_row in enumerate(ranked_rows, start=1):
        ranked_row["rank"] = index

    print("\n=== Final Leaderboard (top 10) ===")
    for row in ranked_rows[:10]:
        print(
            f"#{row['rank']} task={row['task']} lr={row['lr']} warmup={row['warmup']} "
            f"val_log_loss={row['val_log_loss']:.4f} test_log_loss={row['test_log_loss']:.4f}"
        )

    print(f"Wrote JSONL: {LEADERBOARD_JSONL}")
    print(f"Wrote CSV: {LEADERBOARD_CSV}")


if __name__ == "__main__":
    main()
