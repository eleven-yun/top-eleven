"""Phase 7 — Issue-based batch inference for China Football Lottery.

Trains LightGBM on all historical data strictly before a target date,
then outputs ranked pre-match picks (handicap + fulltime) for that date.

Usage:
    # Predict all matches on a specific date (simulation / backtest mode):
    conda run -n top-eleven python scripts/issue_predict.py --date 2025-03-15

    # Predict matches on today's date (live / forward-looking mode):
    conda run -n top-eleven python scripts/issue_predict.py

    # Custom thresholds:
    conda run -n top-eleven python scripts/issue_predict.py --date 2025-03-15 \\
        --ev-threshold 0.05 --min-confidence 0.55

    # Filter to specific tasks only:
    conda run -n top-eleven python scripts/issue_predict.py --date 2025-03-15 \\
        --tasks handicap_label

    # Output pick slip to file:
    conda run -n top-eleven python scripts/issue_predict.py --date 2025-03-15 \\
        --output output/picks/2025-03-15.json

Output: JSON file + formatted CLI table with:
    - Match info (home vs away, league, date, handicap line)
    - Task (handicap/fulltime)
    - Recommended outcome (home_win / draw / away_win)
    - Odds, EV, model confidence
    - Stake (¥2 per bet, minimum lottery ticket)
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from datetime import date, datetime, timezone

import numpy as np

# Suppress sklearn feature-name warning when predicting with numpy arrays
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

cfp = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(cfp, ".."))
sys.path.insert(0, ROOT)

from data.data_loader import (
    load_json_or_jsonl as load_records,
    flatten_prematch_features,
    extract_market_tokens,
)
from data.label_mapping import map_fulltime_label, map_handicap_label

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_NUM_CLASSES = {
    "fulltime_label": 3,
    "handicap_label": 3,
}

TASK_PLAY_TYPE = {
    "fulltime_label": "fulltime_1x2",
    "handicap_label": "handicap_1x2",
}

# Label index → human-readable name (handicap and fulltime use same 3-class schema)
LABEL_NAMES = {0: "home_win", 1: "draw", 2: "away_win"}

STAKE_PER_BET = 2.0  # ¥2 minimum lottery ticket


# ---------------------------------------------------------------------------
# EV helpers
# ---------------------------------------------------------------------------

def compute_ev(prob: float, odds: float) -> float:
    """EV_i = prob_i * odds_i - 1  (profit per ¥1 staked)."""
    return prob * odds - 1.0


def pick_from_probs(probs, odds_dict, ev_threshold, min_confidence):
    """Return the best bet for a match or None.

    Finds the highest-EV qualifying outcome (EV >= threshold AND prob >= min_conf).
    odds_dict maps label_index → float|None.
    Returns dict with keys: label, label_name, prob, odds, ev.
    """
    best = None
    for label_idx, prob in enumerate(probs):
        odds = odds_dict.get(label_idx)
        if odds is None or odds <= 1.0:
            continue
        ev = compute_ev(prob, odds)
        if ev >= ev_threshold and prob >= min_confidence:
            if best is None or ev > best["ev"]:
                best = {
                    "label": label_idx,
                    "label_name": LABEL_NAMES[label_idx],
                    "prob": round(float(prob), 4),
                    "odds": round(float(odds), 4),
                    "ev": round(float(ev), 4),
                }
    return best


# ---------------------------------------------------------------------------
# Data loading & feature building
# ---------------------------------------------------------------------------

def load_all_data(root):
    data_config = json.loads(open(os.path.join(root, "config/data_config.json")).read())
    processed_dir = os.path.join(root, data_config["paths"]["processed_dir"])

    prematch_records = load_records(os.path.join(processed_dir, "prematch_features.jsonl"))
    meta_records = load_records(os.path.join(processed_dir, "match_meta.jsonl"))
    market_records = load_records(os.path.join(processed_dir, "lottery_market.jsonl"))

    prematch_by_id = {r["match_id"]: r for r in prematch_records}
    meta_by_id = {r["match_id"]: dict(r) for r in meta_records}

    market_by_id = defaultdict(list)
    for m in market_records:
        market_by_id[m["match_id"]].append(m)
        # Merge handicap_line into meta for label computation
        if m.get("play_type") == "handicap_1x2" and m.get("handicap_line") is not None:
            mid = m["match_id"]
            if mid in meta_by_id:
                meta_by_id[mid]["handicap_line"] = m["handicap_line"]

    return prematch_by_id, meta_by_id, market_by_id, meta_records


def build_features(mid, prematch_by_id, market_by_id):
    """Return 46-dim feature vector or None if data is missing."""
    pf = prematch_by_id.get(mid)
    if pf is None:
        return None
    return flatten_prematch_features(pf) + extract_market_tokens(market_by_id.get(mid, []))


def match_date(meta_by_id, mid):
    """Return date string YYYY-MM-DD or ''."""
    dt = meta_by_id.get(mid, {}).get("datetime_utc", "")
    return dt[:10] if dt else ""


def label_for_task(task, meta):
    """Return ground-truth label int or -1 if unavailable/future."""
    try:
        if task == "fulltime_label":
            return map_fulltime_label(meta.get("final_result", ""))
        if task == "handicap_label":
            home_goals = meta.get("home_goals")
            away_goals = meta.get("away_goals")
            handicap_line = meta.get("handicap_line")
            if home_goals is None or away_goals is None or handicap_line is None:
                return -1
            return map_handicap_label(
                home_goals,
                away_goals,
                handicap_line,
            )
    except (ValueError, TypeError):
        pass
    return -1


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

def train_lgbm(X_train, y_train, X_val, y_val, num_class, n_estimators=500, lr=0.05):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_class,
        n_estimators=n_estimators,
        learning_rate=lr,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=9999)],
    )
    return model


# ---------------------------------------------------------------------------
# Core prediction logic
# ---------------------------------------------------------------------------

def run_issue_predict(
    pred_date_str,
    tasks,
    ev_threshold,
    min_confidence,
    root=ROOT,
    val_fraction=0.15,
):
    """Run inference for all matches on pred_date_str.

    Trains LightGBM on all matches strictly before pred_date_str.
    val_fraction: fraction of training data used as early-stopping set (most recent).

    Returns: list of pick dicts, sorted by EV descending.
    """
    prematch_by_id, meta_by_id, market_by_id, meta_records = load_all_data(root)

    # Partition all match_ids by date
    train_ids = []
    target_ids = []
    for r in meta_records:
        mid = r["match_id"]
        d = r.get("datetime_utc", "")[:10]
        if not d:
            continue
        if d < pred_date_str:
            train_ids.append(mid)
        elif d == pred_date_str:
            target_ids.append(mid)

    print(f"Prediction date: {pred_date_str}")
    print(f"  Training matches (before date): {len(train_ids)}")
    print(f"  Target matches (on date):       {len(target_ids)}")

    if not target_ids:
        print("  No matches found for this date in the dataset.")
        return []

    # Use most recent val_fraction of train as early-stopping set.
    # Sort by datetime_utc so the "most recent" slice is chronologically correct
    # (meta_records are grouped by league/season, not globally by date).
    train_ids_sorted = sorted(train_ids, key=lambda mid: meta_by_id.get(mid, {}).get("datetime_utc", ""))
    n_val = max(1, int(len(train_ids_sorted) * val_fraction))
    val_ids_set = set(train_ids_sorted[-n_val:])

    all_picks = []

    for task in tasks:
        num_class = TASK_NUM_CLASSES[task]
        play_type = TASK_PLAY_TYPE[task]

        # Build train / val sets
        X_tr, y_tr, X_val_arr, y_val_arr = [], [], [], []
        for mid in train_ids:
            feats = build_features(mid, prematch_by_id, market_by_id)
            meta = meta_by_id.get(mid)
            if feats is None or meta is None:
                continue
            lbl = label_for_task(task, meta)
            if lbl < 0:
                continue
            if mid in val_ids_set:
                X_val_arr.append(feats)
                y_val_arr.append(lbl)
            else:
                X_tr.append(feats)
                y_tr.append(lbl)

        if len(X_tr) < 100 or len(X_val_arr) < 10:
            print(f"  [{task}] Not enough training data — skipping.")
            continue

        X_tr = np.array(X_tr, dtype=np.float32)
        y_tr = np.array(y_tr, dtype=np.int32)
        X_val_arr = np.array(X_val_arr, dtype=np.float32)
        y_val_arr = np.array(y_val_arr, dtype=np.int32)

        model = train_lgbm(X_tr, y_tr, X_val_arr, y_val_arr, num_class)
        print(f"  [{task}] Trained on {len(X_tr)} matches (val={len(X_val_arr)}), "
              f"best_iter={model.best_iteration_}")

        # Build target features
        X_tgt, tgt_ids_valid = [], []
        for mid in target_ids:
            feats = build_features(mid, prematch_by_id, market_by_id)
            if feats is None:
                continue
            X_tgt.append(feats)
            tgt_ids_valid.append(mid)

        if not X_tgt:
            print(f"  [{task}] No target features available.")
            continue

        X_tgt = np.array(X_tgt, dtype=np.float32)
        probs_matrix = model.predict_proba(X_tgt)

        # Build picks
        for mid, probs in zip(tgt_ids_valid, probs_matrix):
            meta = meta_by_id.get(mid, {})
            pf = prematch_by_id.get(mid, {})

            # Odds dict for this task
            odds_dict = {}
            for mkt in market_by_id.get(mid, []):
                if mkt.get("play_type") == play_type:
                    odds_dict[0] = mkt.get("home_odds")
                    odds_dict[1] = mkt.get("draw_odds")
                    odds_dict[2] = mkt.get("away_odds")
                    break

            pick = pick_from_probs(probs, odds_dict, ev_threshold, min_confidence)
            if pick is None:
                continue

            # Ground truth (will be -1 for future / unknown matches)
            true_label = label_for_task(task, meta)

            # Handicap line for display
            handicap_line = None
            if task == "handicap_label":
                for mkt in market_by_id.get(mid, []):
                    if mkt.get("play_type") == "handicap_1x2":
                        handicap_line = mkt.get("handicap_line")
                        break

            all_picks.append({
                "match_id": mid,
                "date": meta.get("datetime_utc", "")[:10],
                "league": meta.get("league_code", ""),
                "home_team": pf.get("home", {}).get("team_name", meta.get("home_team_id", "?")),
                "away_team": pf.get("away", {}).get("team_name", meta.get("away_team_id", "?")),
                "task": task,
                "play_type": play_type,
                "handicap_line": handicap_line,
                "pick_label": pick["label"],
                "pick_name": pick["label_name"],
                "odds": pick["odds"],
                "ev": pick["ev"],
                "confidence": pick["prob"],
                "all_probs": [round(float(p), 4) for p in probs],
                "stake_yuan": STAKE_PER_BET,
                "true_label": true_label,
                "hit": (pick["label"] == true_label) if true_label >= 0 else None,
            })

    all_picks.sort(key=lambda x: -x["ev"])
    return all_picks


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_TASK_ABBREV = {"fulltime_label": "FT", "handicap_label": "HC"}
_PICK_ABBREV = {"home_win": "HOME", "draw": "DRAW", "away_win": "AWAY"}


def format_pick_table(picks, show_result=True):
    """Return a formatted text table of picks."""
    if not picks:
        return "  (no qualifying picks)\n"

    lines = []
    header = (
        f"  {'#':<3} {'Match':<32} {'Lg':<5} {'Task':<3} "
        f"{'Pick':<5} {'HC':>5} {'Odds':>5} {'EV':>6} {'Conf':>5}"
    )
    if show_result:
        header += f"  {'Result':<6}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for i, p in enumerate(picks, 1):
        home = p["home_team"][:14]
        away = p["away_team"][:14]
        matchup = f"{home} v {away}"
        hc = f"{p['handicap_line']:+.2f}" if p["handicap_line"] is not None else "  --"
        hit_str = ""
        if show_result and p.get("hit") is not None:
            hit_str = "  WIN " if p["hit"] else "  LOSS"
        elif show_result:
            hit_str = "  ?   "

        row = (
            f"  {i:<3} {matchup:<32} {p['league']:<5} {_TASK_ABBREV.get(p['task'], '??'):<3} "
            f"{_PICK_ABBREV.get(p['pick_name'], p['pick_name']):<5} {hc:>5} "
            f"{p['odds']:>5.2f} {p['ev']:>+6.3f} {p['confidence']:>5.3f}"
        )
        row += hit_str
        lines.append(row)

    return "\n".join(lines) + "\n"


def print_summary(picks, pred_date_str, ev_threshold, min_confidence):
    """Print a formatted pick slip to stdout."""
    print()
    print("=" * 72)
    print(f"  FOOTBALL LOTTERY PICK SLIP — {pred_date_str}")
    print(f"  EV threshold: {ev_threshold:+.2f}  |  Min confidence: {min_confidence:.2f}")
    print(f"  Total picks: {len(picks)}  |  Total stake: ¥{len(picks) * STAKE_PER_BET:.0f}")
    print("=" * 72)

    # Split by task
    hc_picks = [p for p in picks if p["task"] == "handicap_label"]
    ft_picks = [p for p in picks if p["task"] == "fulltime_label"]

    if hc_picks:
        known = [p for p in hc_picks if p.get("hit") is not None]
        result_str = ""
        if known:
            wins = sum(1 for p in known if p["hit"])
            profit = sum((p["odds"] - 1) * STAKE_PER_BET if p["hit"] else -STAKE_PER_BET for p in known)
            result_str = f"  [{wins}/{len(known)} wins, profit ¥{profit:+.2f}]"
        print(f"\n  HANDICAP 1X2 ({len(hc_picks)} picks){result_str}")
        show = any(p.get("hit") is not None for p in hc_picks)
        print(format_pick_table(hc_picks, show_result=show), end="")

    if ft_picks:
        known = [p for p in ft_picks if p.get("hit") is not None]
        result_str = ""
        if known:
            wins = sum(1 for p in known if p["hit"])
            profit = sum((p["odds"] - 1) * STAKE_PER_BET if p["hit"] else -STAKE_PER_BET for p in known)
            result_str = f"  [{wins}/{len(known)} wins, profit ¥{profit:+.2f}]"
        print(f"\n  FULLTIME 1X2 ({len(ft_picks)} picks){result_str}")
        show = any(p.get("hit") is not None for p in ft_picks)
        print(format_pick_table(ft_picks, show_result=show), end="")

    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    parser = argparse.ArgumentParser(
        description="Issue-based pre-match pick generation for football lottery."
    )
    parser.add_argument(
        "--date",
        default=today_str,
        help="Target date YYYY-MM-DD (default: today UTC). "
             "Trains on all data strictly before this date.",
    )
    parser.add_argument(
        "--tasks",
        default="handicap_label,fulltime_label",
        help="Comma-separated task list (default: handicap_label,fulltime_label)",
    )
    parser.add_argument("--ev-threshold", type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument(
        "--output",
        default=None,
        help="Save picks JSON to this path (default: output/picks/<date>.json)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Print only top-N picks per task in CLI (0 = all)",
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip() in TASK_NUM_CLASSES]
    if not tasks:
        print("No valid tasks specified.")
        sys.exit(1)

    picks = run_issue_predict(
        pred_date_str=args.date,
        tasks=tasks,
        ev_threshold=args.ev_threshold,
        min_confidence=args.min_confidence,
    )

    # Optionally truncate for display
    display_picks = picks
    if args.top_n > 0:
        hc = [p for p in picks if p["task"] == "handicap_label"][: args.top_n]
        ft = [p for p in picks if p["task"] == "fulltime_label"][: args.top_n]
        display_picks = hc + ft

    print_summary(display_picks, args.date, args.ev_threshold, args.min_confidence)

    # Save JSON
    output_path = os.path.abspath(args.output or os.path.join(ROOT, "output", "picks", f"{args.date}.json"))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "prediction_date": args.date,
                "ev_threshold": args.ev_threshold,
                "min_confidence": args.min_confidence,
                "total_picks": len(picks),
                "total_stake_yuan": len(picks) * STAKE_PER_BET,
                "picks": picks,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  Picks saved → {output_path}")


if __name__ == "__main__":
    main()
