"""Sprint D ensemble: soft-vote LightGBM + Transformer predictions.

Loads LightGBM (all 46 features including odds) and transformer checkpoint,
computes weighted average of softmax probabilities, then reports:
  - Transformer-only performance
  - LightGBM-only performance
  - Ensemble performance (sweep over alpha values)

Usage:
    conda run -n top-eleven python scripts/ensemble_lgbm_transformer.py \
        --task fulltime_label \
        --checkpoint output/checkpoints/<run>/checkpoint_best.pt
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

cfp = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(cfp, ".."))
sys.path.insert(0, ROOT)

import lightgbm as lgb
import torch

from data.data_loader import (
    load_json_or_jsonl as load_records,
    flatten_prematch_features,
    extract_market_tokens,
    split_samples_by_season,
    create_transformer_prematch_loader_for_split,
)
from data.label_mapping import map_fulltime_label, map_htft_label, map_handicap_label
from nn_modules.embedding.token_schema import TOKEN_NAMES
from nn_modules.transformer.top_former import TopFormer
from utils.checkpoint import load_checkpoint_state_dict

TASK_NUM_CLASSES = {
    "fulltime_label": 3,
    "htft_label": 9,
    "handicap_label": 3,
}

# Feature name ordering for LightGBM 46-feature vector
FEATURE_NAMES_40 = list(TOKEN_NAMES[0:35]) + list(TOKEN_NAMES[41:46])
FEATURE_NAMES_46 = FEATURE_NAMES_40 + list(TOKEN_NAMES[35:41])


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


def build_lgbm_xy(samples, prematch_by_id, meta_by_id, market_by_id, task):
    X, y, ids = [], [], []
    for s in samples:
        mid = s.get("match_id", s.get("id"))
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


def log_loss(y_true, y_prob):
    eps = 1e-7
    y_prob = np.clip(y_prob, eps, 1 - eps)
    n = len(y_true)
    return -np.sum(np.log(y_prob[np.arange(n), y_true])) / n


def accuracy(y_true, y_prob):
    return float(np.mean(np.argmax(y_prob, axis=1) == y_true))


def get_transformer_probs(checkpoint_path, task, split_name, data_config, processed_dir, device="cpu"):
    """Run the transformer model on a split; return (probs_by_id, labels_by_id)."""
    checkpoint = load_checkpoint_state_dict(checkpoint_path, device)
    config_path = os.path.join(ROOT, "config", "config.json")
    config = json.loads(open(config_path).read())
    model_params = config.get("model_params", {})
    train_params = config.get("train_params", {})

    model = TopFormer(
        num_layers=model_params["num_layers"],
        d_model=model_params["d_model"],
        nhead=model_params["nhead"],
        num_classes=TASK_NUM_CLASSES[task],
    )
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    loader, _ = create_transformer_prematch_loader_for_split(
        processed_dir=processed_dir,
        season_split=data_config["season_split"],
        split_name=split_name,
        source_length=model_params.get("source_length", 46),
        target_length=model_params.get("target_length", 1),
        d_model=model_params["d_model"],
        batch_size=512,
        label_key=task,
        shuffle=False,
    )

    probs_by_id, labels_by_id = {}, {}
    with torch.no_grad():
        for batch in loader:
            token_values = batch["token_values"].to(device)
            gt = batch["gt"].numpy()
            match_ids = batch["match_id"]
            logits = model(token_values)[-1]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for i, (mid, label) in enumerate(zip(match_ids, gt)):
                if label >= 0:
                    probs_by_id[mid] = probs[i]
                    labels_by_id[mid] = int(label)
    return probs_by_id, labels_by_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="fulltime_label", choices=list(TASK_NUM_CLASSES.keys()))
    parser.add_argument("--checkpoint", required=True, help="Path to transformer checkpoint .pt file")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    args = parser.parse_args()

    data_config_path = os.path.join(ROOT, "config/data_config.json")
    data_config = json.loads(open(data_config_path).read())
    processed_dir = os.path.join(ROOT, data_config["paths"]["processed_dir"])

    prematch_records = load_records(os.path.join(processed_dir, "prematch_features.jsonl"))
    meta_records = load_records(os.path.join(processed_dir, "match_meta.jsonl"))
    market_records = load_records(os.path.join(processed_dir, "lottery_market.jsonl"))

    prematch_by_id = {r["match_id"]: r for r in prematch_records}
    meta_by_id = {r["match_id"]: dict(r) for r in meta_records}

    market_by_id = defaultdict(list)
    for m in market_records:
        market_by_id[m["match_id"]].append(m)
        if m.get("play_type") == "handicap_1x2" and m.get("handicap_line") is not None:
            mid = m["match_id"]
            if mid in meta_by_id:
                meta_by_id[mid]["handicap_line"] = m["handicap_line"]

    all_samples = [{"match_id": r["match_id"]} for r in meta_records]
    split_samples = split_samples_by_season(all_samples, meta_records, data_config["season_split"])

    # --- LightGBM ---
    X_train, y_train, _ = build_lgbm_xy(split_samples["train"], prematch_by_id, meta_by_id, market_by_id, args.task)
    X_val, y_val, _ = build_lgbm_xy(split_samples["validation"], prematch_by_id, meta_by_id, market_by_id, args.task)
    X_eval, y_eval, eval_ids = build_lgbm_xy(split_samples[args.split], prematch_by_id, meta_by_id, market_by_id, args.task)

    lgbm_model = lgb.LGBMClassifier(
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
    lgbm_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=9999)],
    )
    lgbm_probs = lgbm_model.predict_proba(X_eval)

    # --- Transformer ---
    trans_prob_by_id, trans_label_by_id = get_transformer_probs(
        args.checkpoint, args.task, args.split, data_config, processed_dir
    )

    # Align by match_id
    lgbm_prob_by_id = {mid: p for mid, p in zip(eval_ids, lgbm_probs)}
    lgbm_label_by_id = {mid: l for mid, l in zip(eval_ids, y_eval)}

    common_ids = [mid for mid in eval_ids if mid in trans_prob_by_id]
    if not common_ids:
        print("No overlapping match IDs between LightGBM and transformer")
        return

    lp = np.array([lgbm_prob_by_id[m] for m in common_ids])
    tp = np.array([trans_prob_by_id[m] for m in common_ids])
    yl = np.array([lgbm_label_by_id[m] for m in common_ids])

    print(f"\n=== Ensemble evaluation: {args.task} on {args.split} ({len(common_ids)} samples) ===")
    print(f"LightGBM only:    log-loss={log_loss(yl, lp):.4f}  acc={accuracy(yl, lp)*100:.1f}%")
    print(f"Transformer only: log-loss={log_loss(yl, tp):.4f}  acc={accuracy(yl, tp)*100:.1f}%")
    print()

    best_alpha, best_loss = None, float("inf")
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ep = alpha * lp + (1 - alpha) * tp
        ll = log_loss(yl, ep)
        acc = accuracy(yl, ep)
        marker = ""
        if ll < best_loss:
            best_loss = ll
            best_alpha = alpha
            marker = " <-- best"
        print(f"  alpha={alpha:.1f} (LGBM weight): log-loss={ll:.4f}  acc={acc*100:.1f}%{marker}")


if __name__ == "__main__":
    main()
