"""LightGBM inference: train on train-split, generate JSONL predictions for any split.

Outputs one JSONL record per match in the format expected by scripts/backtest_ev.py:
    {"match_id": "...", "task": "fulltime_label", "probs": [...], "predicted_label": 0, "true_label": 1}

Usage:
    conda run -n top-eleven python scripts/predict_lgbm.py --task fulltime_label --split validation
    conda run -n top-eleven python scripts/predict_lgbm.py --task fulltime_label --split test
    conda run -n top-eleven python scripts/predict_lgbm.py --task handicap_label --split validation
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

cfp = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(cfp, ".."))
sys.path.insert(0, ROOT)

from data.data_loader import (
    load_json_or_jsonl as load_records,
    flatten_prematch_features,
    extract_market_tokens,
    split_samples_by_season,
)
from data.label_mapping import map_fulltime_label, map_htft_label, map_handicap_label
from utils.calibration import resolve_task_temperature

TASK_NUM_CLASSES = {
    "fulltime_label": 3,
    "htft_label": 9,
    "handicap_label": 3,
}


def apply_temperature(probs, temperature):
    """Apply temperature scaling to a probability vector."""
    if temperature <= 0 or abs(temperature - 1.0) < 1e-9:
        return [float(x) for x in probs]
    logs = [math.log(max(1e-12, float(p))) / temperature for p in probs]
    m = max(logs)
    exps = [math.exp(x - m) for x in logs]
    z = sum(exps)
    return [x / z for x in exps]


def result_to_label(task, meta):
    try:
        if task == "fulltime_label":
            return map_fulltime_label(meta.get("final_result", ""))
        if task == "htft_label":
            return map_htft_label(
                meta.get("halftime_home_goals", 0),
                meta.get("halftime_away_goals", 0),
                meta.get("home_goals", 0),
                meta.get("away_goals", 0),
            )
        if task == "handicap_label":
            return map_handicap_label(
                meta.get("home_goals", 0),
                meta.get("away_goals", 0),
                meta.get("handicap_line"),
            )
    except (ValueError, TypeError):
        pass
    return -1


def build_xy(samples, prematch_by_id, meta_by_id, market_by_id, task):
    X, y, ids = [], [], []
    for s in samples:
        mid = s.get("match_id")
        pf = prematch_by_id.get(mid)
        meta = meta_by_id.get(mid)
        if pf is None or meta is None:
            continue
        label = result_to_label(task, meta)
        if label < 0:
            continue
        feats = flatten_prematch_features(pf) + extract_market_tokens(market_by_id.get(mid, []))
        X.append(feats)
        y.append(label)
        ids.append(mid)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="fulltime_label", choices=list(TASK_NUM_CLASSES.keys()))
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Manual probability temperature override (default: auto by task)",
    )
    parser.add_argument(
        "--temperature-report-dir",
        default="output/backtest",
        help="Directory containing <task>_test_temperature_report.json",
    )
    parser.add_argument("--output", default=None, help="Output JSONL path")
    args = parser.parse_args()

    import lightgbm as lgb

    data_config_path = os.path.join(ROOT, "config/data_config.json")
    data_config = json.loads(open(data_config_path).read())
    processed_dir = os.path.join(ROOT, data_config["paths"]["processed_dir"])

    prematch_records = load_records(os.path.join(processed_dir, "prematch_features.jsonl"))
    meta_records = load_records(os.path.join(processed_dir, "match_meta.jsonl"))
    market_records = load_records(os.path.join(processed_dir, "lottery_market.jsonl"))

    prematch_by_id = {r["match_id"]: r for r in prematch_records}
    meta_by_id = {r["match_id"]: dict(r) for r in meta_records}
    meta_dt_by_id = {r["match_id"]: r.get("datetime_utc", "") for r in meta_records}

    market_by_id = defaultdict(list)
    for m in market_records:
        market_by_id[m["match_id"]].append(m)
        if m.get("play_type") == "handicap_1x2" and m.get("handicap_line") is not None:
            mid = m["match_id"]
            if mid in meta_by_id:
                meta_by_id[mid]["handicap_line"] = m["handicap_line"]

    all_samples = [{"match_id": r["match_id"]} for r in meta_records]
    split_samples = split_samples_by_season(all_samples, meta_records, data_config["season_split"])

    X_train, y_train, _ = build_xy(split_samples["train"], prematch_by_id, meta_by_id, market_by_id, args.task)
    X_target, y_target, target_ids = build_xy(
        split_samples[args.split], prematch_by_id, meta_by_id, market_by_id, args.task
    )

    # Early-stopping set: always use the validation season from the split config.
    # When --split validation, the target split IS the validation season, so using
    # the same data for early-stopping would let the model overfit to it and produce
    # optimistic "validation" predictions.  We therefore use a held-out tail of the
    # training set for early-stopping in that case, and the true validation season
    # otherwise.
    if args.split == "validation":
        # Hold out the most recent 15% of training matches (by datetime) for early stopping.
        train_ids_sorted = sorted(
            [s["match_id"] for s in split_samples["train"]],
            key=lambda mid: meta_dt_by_id.get(mid, ""),
        )
        n_es = max(1, int(len(train_ids_sorted) * 0.15))
        es_ids = set(train_ids_sorted[-n_es:])
        tr_ids_reduced = [s for s in split_samples["train"] if s["match_id"] not in es_ids]
        es_ids_list = [{"match_id": mid} for mid in train_ids_sorted[-n_es:]]
        X_train, y_train, _ = build_xy(tr_ids_reduced, prematch_by_id, meta_by_id, market_by_id, args.task)
        X_es, y_es, _ = build_xy(es_ids_list, prematch_by_id, meta_by_id, market_by_id, args.task)
        early_stop_set = (X_es, y_es)
    else:
        X_val, y_val, _ = build_xy(split_samples["validation"], prematch_by_id, meta_by_id, market_by_id, args.task)
        early_stop_set = (X_val, y_val)

    print(f"Training LightGBM: task={args.task}, train={len(X_train)}, {args.split}={len(X_target)}")

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=TASK_NUM_CLASSES[args.task],
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[early_stop_set],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=9999)],
    )

    probs = model.predict_proba(X_target)
    temperature, temperature_source = resolve_task_temperature(
        root=ROOT,
        task=args.task,
        explicit_temperature=args.temperature,
        report_dir=args.temperature_report_dir,
    )
    probs = np.array([apply_temperature(p, temperature) for p in probs], dtype=np.float32)
    print(f"Using temperature={temperature:.4f} (source={temperature_source}) for task={args.task}")

    output_path = os.path.abspath(
        args.output or os.path.join(ROOT, "output", "predictions", f"{args.task}_{args.split}.jsonl")
    )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for mid, p, true_label in zip(target_ids, probs, y_target):
            rec = {
                "match_id": mid,
                "task": args.task,
                "probs": [round(float(x), 6) for x in p],
                "predicted_label": int(np.argmax(p)),
                "true_label": int(true_label),
                "datetime_utc": meta_dt_by_id.get(mid, ""),
                "temperature": round(float(temperature), 6),
                "temperature_source": temperature_source,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(target_ids)} predictions → {output_path}")


if __name__ == "__main__":
    main()
