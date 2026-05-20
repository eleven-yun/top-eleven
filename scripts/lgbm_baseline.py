"""Sprint D: LightGBM tabular baseline for top-eleven.

Trains a LightGBM multi-class classifier on the same feature vector used by the
transformer model (from flatten_prematch_features), using the same train/val/test
season split defined in data_config.json.

Reports:
  - Val log-loss, val accuracy
  - Test log-loss, test accuracy

Usage:
    conda run -n top-eleven python scripts/lgbm_baseline.py --task fulltime_label
    conda run -n top-eleven python scripts/lgbm_baseline.py --task htft_label
    conda run -n top-eleven python scripts/lgbm_baseline.py --task handicap_label
"""
import argparse
import json
import os
import sys

import numpy as np

cwd = os.getcwd()
cfp = os.path.dirname(os.path.abspath(__file__))
os.chdir(cfp)
ROOT = os.path.abspath("..")
sys.path.insert(0, ROOT)
os.chdir(cwd)

from data.data_loader import load_json_or_jsonl as load_records, flatten_prematch_features, extract_market_tokens, split_samples_by_season
from data.label_mapping import map_fulltime_label, map_htft_label, map_handicap_label

from nn_modules.embedding.token_schema import TOKEN_NAMES

# Feature names for the combined feature vector:
#   flatten_prematch_features() -> 40 values: TOKEN_NAMES[0:35] + TOKEN_NAMES[41:46]
#   extract_market_tokens()     ->  6 values: TOKEN_NAMES[35:41]
FEATURE_NAMES_40 = list(TOKEN_NAMES[0:35]) + list(TOKEN_NAMES[41:46])
FEATURE_NAMES_46 = FEATURE_NAMES_40 + list(TOKEN_NAMES[35:41])


TASK_NUM_CLASSES = {
    "fulltime_label": 3,
    "htft_label": 9,
    "handicap_label": 3,
}

LABEL_KEY_TO_META_FIELD = {
    "fulltime_label": "final_result",
    "htft_label": "htft_result",
    "handicap_label": "handicap_result",
}

FULLTIME_CLASSES = {"home_win": 0, "draw": 1, "away_win": 2}
HANDICAP_CLASSES = {"home_win": 0, "draw": 1, "away_win": 2}

HTFT_CLASSES = {
    "HH": 0, "HD": 1, "HA": 2,
    "DH": 3, "DD": 4, "DA": 5,
    "AH": 6, "AD": 7, "AA": 8,
}


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


def build_xy(samples, prematch_by_id, meta_by_id, task, market_by_id=None):
    X, y = [], []
    for s in samples:
        mid = s.get("match_id", s.get("id"))
        pf = prematch_by_id.get(mid)
        meta = meta_by_id.get(mid)
        if pf is None or meta is None:
            continue
        label = result_to_label(task, meta)
        if label < 0:
            continue
        feats = flatten_prematch_features(pf)
        if market_by_id is not None:
            feats = feats + extract_market_tokens(market_by_id.get(mid, []))
        X.append(feats)
        y.append(label)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def log_loss(y_true, y_prob):
    eps = 1e-7
    y_prob = np.clip(y_prob, eps, 1 - eps)
    n = len(y_true)
    return -np.sum(np.log(y_prob[np.arange(n), y_true])) / n


def accuracy(y_true, y_prob):
    preds = np.argmax(y_prob, axis=1)
    return float(np.mean(preds == y_true))


def main():
    parser = argparse.ArgumentParser(description="LightGBM tabular baseline")
    parser.add_argument("--task", default="fulltime_label", choices=list(TASK_NUM_CLASSES.keys()))
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--output", default=None, help="Optional path to write JSON report")
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

    # Merge handicap_line from lottery_market into meta_by_id
    # and group market records by match_id for Sprint C
    from collections import defaultdict
    market_by_id = defaultdict(list)
    for m in market_records:
        market_by_id[m["match_id"]].append(m)
        if m.get("play_type") == "handicap_1x2" and m.get("handicap_line") is not None:
            mid = m["match_id"]
            if mid in meta_by_id:
                meta_by_id[mid]["handicap_line"] = m["handicap_line"]

    # Build dummy samples list (just match_ids from meta)
    all_samples = [{"match_id": r["match_id"]} for r in meta_records]

    split_samples = split_samples_by_season(all_samples, meta_records, data_config["season_split"])

    X_train, y_train = build_xy(split_samples["train"], prematch_by_id, meta_by_id, args.task, market_by_id)
    X_val, y_val = build_xy(split_samples["validation"], prematch_by_id, meta_by_id, args.task, market_by_id)
    X_test, y_test = build_xy(split_samples["test"], prematch_by_id, meta_by_id, args.task, market_by_id)

    n_classes = TASK_NUM_CLASSES[args.task]
    print(f"\n=== LightGBM Baseline: {args.task} ===")
    print(f"Train: {len(X_train)} samples | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Features: {X_train.shape[1]} | Classes: {n_classes}")

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=n_classes,
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
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=100)],
    )

    val_prob = model.predict_proba(X_val)
    test_prob = model.predict_proba(X_test)

    val_ll = log_loss(y_val, val_prob)
    val_acc = accuracy(y_val, val_prob)
    test_ll = log_loss(y_test, test_prob)
    test_acc = accuracy(y_test, test_prob)

    print(f"\nVal  log-loss: {val_ll:.4f}  accuracy: {val_acc*100:.1f}%")
    print(f"Test log-loss: {test_ll:.4f}  accuracy: {test_acc*100:.1f}%")

    # Feature importance (top 15)
    feat_names = FEATURE_NAMES_46 if X_train.shape[1] == 46 else FEATURE_NAMES_40
    importances = sorted(
        zip(feat_names[:X_train.shape[1]], model.feature_importances_),
        key=lambda x: -x[1],
    )
    print("\nTop 15 feature importances:")
    for name, imp in importances[:15]:
        print(f"  {imp:6.0f}  {name}")

    report = {
        "task": args.task,
        "model": "LightGBM",
        "n_features": int(X_train.shape[1]),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "hyperparams": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "best_iteration": int(model.best_iteration_) if hasattr(model, "best_iteration_") else None,
        },
        "val_log_loss": round(val_ll, 4),
        "val_accuracy": round(val_acc, 4),
        "test_log_loss": round(test_ll, 4),
        "test_accuracy": round(test_acc, 4),
        "feature_importances": {name: int(imp) for name, imp in importances},
    }

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to {args.output}")

    return report


if __name__ == "__main__":
    main()
