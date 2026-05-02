"""EV-based betting backtest for football lottery prediction.

Takes per-match probability predictions (from scripts/predict.py) and market
odds (from data/processed/lottery_market.jsonl or lottery_market_cn.jsonl),
and simulates a bet-selection strategy to report risk-adjusted performance.

Expected value formula for class i:
    EV_i = probs[i] * (odds_i - 1) - (1 - probs[i])
         = probs[i] * odds_i - 1

A bet on outcome i is placed when:
    1. EV_i >= --ev-threshold
    2. probs[i] >= --min-confidence
    3. (optional) at most one bet per match: the highest-EV qualifying outcome

Label-to-odds slot mapping:
    fulltime_label  : 0=home_win → home_odds,  1=draw → draw_odds,   2=away_win → away_odds
    handicap_label  : 0=home_win → home_odds,  1=draw → draw_odds*,  2=away_win → away_odds
    htft_label      : no market odds in current data — skipped

    * handicap draw odds are NULL in european-source data; those bets are excluded.

Usage (against European odds already in lottery_market.jsonl):
    python scripts/backtest_ev.py \\
        --predictions output/predictions/fulltime_label_test.jsonl \\
        --market     data/processed/lottery_market.jsonl \\
        --task       fulltime_label \\
        --output     output/backtest/fulltime_test.json

Usage (against China Lottery odds if enriched):
    python scripts/backtest_ev.py \\
        --predictions output/predictions/fulltime_label_test.jsonl \\
        --market     data/processed/lottery_market_cn.jsonl \\
        --task       fulltime_label \\
        --ev-threshold 0.10 \\
        --max-one-bet-per-match \\
        --output     output/backtest/fulltime_cn_test.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Label index → market odds key, per task
TASK_ODDS_SLOTS = {
    "fulltime_label": {
        "play_type": "fulltime_1x2",
        "class_to_key": {0: "home_odds", 1: "draw_odds", 2: "away_odds"},
    },
    "handicap_label": {
        "play_type": "handicap_1x2",
        "class_to_key": {0: "home_odds", 1: "draw_odds", 2: "away_odds"},
    },
}

STAKE_PER_BET = 2.0  # 2 yuan per note (minimum ticket), matches official rule


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    """Load all records from a JSONL file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Market helpers
# ---------------------------------------------------------------------------

def build_market_index(market_records, play_type):
    """Return {match_id: market_dict} for the given play_type.

    If a match has multiple records for the same play_type, keep the one with
    the highest odds_capture_time (i.e. the most recent snapshot).  When no
    capture-time is available, keep the last record encountered.

    Parameters
    ----------
    market_records : list[dict]
    play_type : str  e.g. "fulltime_1x2"

    Returns
    -------
    dict[str, dict]
    """
    index = {}
    for rec in market_records:
        if rec.get("play_type") != play_type:
            continue
        mid = rec["match_id"]
        if mid not in index:
            index[mid] = rec
        else:
            existing_ts = index[mid].get("odds_capture_time") or ""
            new_ts = rec.get("odds_capture_time") or ""
            if new_ts >= existing_ts:
                index[mid] = rec
    return index


def get_odds_for_class(market_rec, class_idx, class_to_key):
    """Return (float odds, missing_reason | None) for a given class index.

    Parameters
    ----------
    market_rec : dict
    class_idx : int
    class_to_key : dict[int, str]  maps class index → odds field name

    Returns
    -------
    tuple(float | None, str | None)
    """
    key = class_to_key.get(class_idx)
    if key is None:
        return None, f"no_odds_key_for_class_{class_idx}"
    value = market_rec.get(key)
    if value is None:
        return None, f"null_{key}"
    try:
        odds = float(value)
    except (TypeError, ValueError):
        return None, f"invalid_{key}={value}"
    if odds <= 1.0:
        return None, f"odds_le_1_{key}={odds}"
    return odds, None


# ---------------------------------------------------------------------------
# EV computation
# ---------------------------------------------------------------------------

def compute_ev(prob, odds):
    """Expected value per unit stake.

    EV = prob * (odds - 1) - (1 - prob)
       = prob * odds - 1

    Parameters
    ----------
    prob : float   model probability for this outcome
    odds : float   decimal odds (> 1.0)

    Returns
    -------
    float
    """
    return prob * odds - 1.0


def implied_probability(odds):
    """Raw implied probability from decimal odds (no margin removal)."""
    return 1.0 / odds if odds > 0 else None


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

def compute_max_drawdown(cumulative_profits):
    """Maximum peak-to-trough decline in cumulative P&L.

    Parameters
    ----------
    cumulative_profits : list[float]

    Returns
    -------
    float  (negative number or 0)
    """
    if not cumulative_profits:
        return 0.0
    peak = cumulative_profits[0]
    max_dd = 0.0
    for val in cumulative_profits:
        if val > peak:
            peak = val
        dd = val - peak
        if dd < max_dd:
            max_dd = dd
    return round(max_dd, 4)


def compute_longest_losing_streak(results):
    """Number of consecutive losses in a chronologically-ordered results list.

    Parameters
    ----------
    results : list[bool]  True = win, False = loss

    Returns
    -------
    int
    """
    max_streak = 0
    current = 0
    for won in results:
        if not won:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def odds_bucket(odds):
    """Return a human-readable odds range label."""
    if odds < 1.5:
        return "1.00-1.50"
    if odds < 2.0:
        return "1.50-2.00"
    if odds < 3.0:
        return "2.00-3.00"
    if odds < 5.0:
        return "3.00-5.00"
    return "5.00+"


# ---------------------------------------------------------------------------
# Core backtest loop
# ---------------------------------------------------------------------------

def run_backtest(predictions, market_index, task, ev_threshold, min_confidence, max_one_bet_per_match):
    """Simulate betting strategy for a list of matches.

    Parameters
    ----------
    predictions : list[dict]
        Per-match records from predict.py JSONL output.
        Required fields: match_id, probs (list[float]), true_label (int).
    market_index : dict[str, dict]
        {match_id: market_record}, for the correct play_type.
    task : str
        "fulltime_label" or "handicap_label".
    ev_threshold : float
        Minimum EV to place a bet.
    min_confidence : float
        Minimum model probability to place a bet.
    max_one_bet_per_match : bool
        If True, place at most one bet per match (highest EV qualifying outcome).

    Returns
    -------
    dict  raw backtest results with all individual bet records.
    """
    slot_cfg = TASK_ODDS_SLOTS[task]
    class_to_key = slot_cfg["class_to_key"]
    num_classes = len(predictions[0]["probs"]) if predictions else 3

    bets = []          # individual placed bets
    skipped = []       # matches with no qualifying bets
    no_market = 0      # matches with no market record

    for pred in predictions:
        mid = pred["match_id"]
        probs = pred["probs"]
        true_label = pred["true_label"]

        market_rec = market_index.get(mid)
        if market_rec is None:
            no_market += 1
            continue

        candidates = []
        for cls in range(num_classes):
            p = probs[cls]
            odds, missing_reason = get_odds_for_class(market_rec, cls, class_to_key)
            if odds is None:
                continue
            ev = compute_ev(p, odds)
            imp_prob = implied_probability(odds)
            if ev >= ev_threshold and p >= min_confidence:
                candidates.append(
                    {
                        "match_id": mid,
                        "class": cls,
                        "prob": round(p, 6),
                        "odds": round(odds, 4),
                        "ev": round(ev, 6),
                        "implied_prob": round(imp_prob, 6),
                        "true_label": true_label,
                        "won": int(cls == true_label),
                        "profit": round((odds - 1) * STAKE_PER_BET if cls == true_label else -STAKE_PER_BET, 4),
                    }
                )

        if not candidates:
            skipped.append(mid)
            continue

        if max_one_bet_per_match:
            candidates = [max(candidates, key=lambda c: c["ev"])]

        bets.extend(candidates)

    return {
        "bets": bets,
        "no_market_count": no_market,
        "skipped_count": len(skipped),
    }


def aggregate_results(raw, task, ev_threshold, min_confidence, max_one_bet_per_match):
    """Compute summary statistics from raw backtest bets.

    Parameters
    ----------
    raw : dict  output from run_backtest()

    Returns
    -------
    dict  full backtest report
    """
    bets = raw["bets"]
    if not bets:
        return {
            "task": task,
            "ev_threshold": ev_threshold,
            "min_confidence": min_confidence,
            "max_one_bet_per_match": max_one_bet_per_match,
            "total_bets": 0,
            "no_market_matches": raw["no_market_count"],
            "skipped_matches": raw["skipped_count"],
            "message": "No bets placed — try lowering ev_threshold or min_confidence.",
        }

    total_bets = len(bets)
    total_stake = round(total_bets * STAKE_PER_BET, 2)
    wins = sum(b["won"] for b in bets)
    total_profit = round(sum(b["profit"] for b in bets), 4)
    roi = round(total_profit / total_stake, 6) if total_stake > 0 else 0.0
    yield_pct = round(roi * 100, 4)
    hit_rate = round(wins / total_bets, 6)

    # chronological cumulative P&L
    cum = []
    running = 0.0
    for b in bets:
        running += b["profit"]
        cum.append(round(running, 4))

    max_dd = compute_max_drawdown(cum)
    longest_loss = compute_longest_losing_streak([bool(b["won"]) for b in bets])

    # odds bucket breakdown
    bucket_stats = defaultdict(lambda: {"bets": 0, "wins": 0, "stake": 0.0, "profit": 0.0})
    for b in bets:
        key = odds_bucket(b["odds"])
        bucket_stats[key]["bets"] += 1
        bucket_stats[key]["wins"] += b["won"]
        bucket_stats[key]["stake"] += STAKE_PER_BET
        bucket_stats[key]["profit"] += b["profit"]

    odds_buckets = {}
    for key, s in sorted(bucket_stats.items()):
        odds_buckets[key] = {
            "bets": s["bets"],
            "hit_rate": round(s["wins"] / s["bets"], 4),
            "roi": round(s["profit"] / s["stake"], 4) if s["stake"] > 0 else 0.0,
        }

    # class breakdown
    class_stats = defaultdict(lambda: {"bets": 0, "wins": 0, "stake": 0.0, "profit": 0.0})
    for b in bets:
        key = b["class"]
        class_stats[key]["bets"] += 1
        class_stats[key]["wins"] += b["won"]
        class_stats[key]["stake"] += STAKE_PER_BET
        class_stats[key]["profit"] += b["profit"]

    class_breakdown = {}
    for k in sorted(class_stats.keys()):
        s = class_stats[k]
        class_breakdown[str(k)] = {
            "bets": s["bets"],
            "hit_rate": round(s["wins"] / s["bets"], 4),
            "roi": round(s["profit"] / s["stake"], 4) if s["stake"] > 0 else 0.0,
        }

    # EV distribution summary
    evs = [b["ev"] for b in bets]
    ev_summary = {
        "mean": round(sum(evs) / len(evs), 6),
        "min": round(min(evs), 6),
        "max": round(max(evs), 6),
    }

    return {
        "task": task,
        "ev_threshold": ev_threshold,
        "min_confidence": min_confidence,
        "max_one_bet_per_match": max_one_bet_per_match,
        "stake_per_bet_yuan": STAKE_PER_BET,
        "total_bets": total_bets,
        "total_stake_yuan": total_stake,
        "total_profit_yuan": total_profit,
        "roi": roi,
        "yield_pct": yield_pct,
        "hit_rate": hit_rate,
        "wins": wins,
        "losses": total_bets - wins,
        "max_drawdown_yuan": max_dd,
        "longest_losing_streak": longest_loss,
        "no_market_matches": raw["no_market_count"],
        "skipped_matches": raw["skipped_count"],
        "ev_summary": ev_summary,
        "odds_bucket_breakdown": odds_buckets,
        "class_breakdown": class_breakdown,
        "cumulative_profit_curve": cum,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="EV-based betting backtest for football lottery prediction models."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to per-match predictions JSONL (output of scripts/predict.py).",
    )
    parser.add_argument(
        "--market",
        required=True,
        help="Path to market odds JSONL (lottery_market.jsonl or lottery_market_cn.jsonl).",
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list(TASK_ODDS_SLOTS.keys()),
        help="Prediction task. Must match the task used in --predictions.",
    )
    parser.add_argument(
        "--ev-threshold",
        type=float,
        default=0.0,
        help="Minimum EV to place a bet. 0.0 = any positive EV. 0.05 = at least 5%% edge.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum model probability to place a bet.",
    )
    parser.add_argument(
        "--max-one-bet-per-match",
        action="store_true",
        default=False,
        help="Place at most one bet per match (highest EV outcome only).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON backtest report. Defaults to output/backtest/<task>.json",
    )
    parser.add_argument(
        "--save-bets",
        action="store_true",
        default=False,
        help="Include the full per-bet list in the output JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    slot_cfg = TASK_ODDS_SLOTS[args.task]

    print(f"Loading predictions: {args.predictions}")
    predictions = load_jsonl(args.predictions)
    print(f"  {len(predictions)} prediction records")

    print(f"Loading market odds: {args.market}")
    market_records = load_jsonl(args.market)
    market_index = build_market_index(market_records, slot_cfg["play_type"])
    print(f"  {len(market_index)} market records for play_type={slot_cfg['play_type']}")

    print(
        f"Running backtest: ev_threshold={args.ev_threshold}, "
        f"min_confidence={args.min_confidence}, "
        f"max_one_bet_per_match={args.max_one_bet_per_match}"
    )

    raw = run_backtest(
        predictions=predictions,
        market_index=market_index,
        task=args.task,
        ev_threshold=args.ev_threshold,
        min_confidence=args.min_confidence,
        max_one_bet_per_match=args.max_one_bet_per_match,
    )

    report = aggregate_results(
        raw=raw,
        task=args.task,
        ev_threshold=args.ev_threshold,
        min_confidence=args.min_confidence,
        max_one_bet_per_match=args.max_one_bet_per_match,
    )

    if not args.save_bets:
        report.pop("cumulative_profit_curve", None)
    else:
        report["bets"] = raw["bets"]

    output_path = args.output
    if output_path is None:
        tag = f"ev{args.ev_threshold}_conf{args.min_confidence}"
        output_path = os.path.join("output", "backtest", f"{args.task}_{tag}.json")

    write_json(output_path, report)

    # pretty-print key metrics
    print("\n--- Backtest Summary ---")
    for key in (
        "total_bets", "total_stake_yuan", "total_profit_yuan",
        "roi", "yield_pct", "hit_rate",
        "max_drawdown_yuan", "longest_losing_streak",
    ):
        if key in report:
            print(f"  {key}: {report[key]}")
    if "odds_bucket_breakdown" in report:
        print("\n  Odds bucket breakdown:")
        for bucket, stats in report["odds_bucket_breakdown"].items():
            print(f"    {bucket}: bets={stats['bets']}, hit_rate={stats['hit_rate']}, roi={stats['roi']}")
    print(f"\nWrote backtest report → {output_path}")


if __name__ == "__main__":
    main()
