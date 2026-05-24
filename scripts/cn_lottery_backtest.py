#!/usr/bin/env python3
"""
China Lottery backtest: compare European odds proxy vs actual lottery SP payouts.

Pipeline:
1. Load model predictions (from predict_lgbm.py JSONL output)
2. Load existing European odds (lottery_market.jsonl) — used for EV filtering
3. Load China Lottery SP data (from fetch_cn_odds.py + enrich_lottery_odds.py)
4. For each bet placed (based on European odds EV filter):
   a. If we have a matching China Lottery SP for the bet outcome → use lottery payout
   b. Otherwise → fall back to European odds payout
5. Report ROI and coverage for both columns

Usage:
    python scripts/cn_lottery_backtest.py \\
        --predictions output/predictions/handicap_label_test.jsonl \\
        --eu-market data/processed/lottery_market.jsonl \\
        --cn-market data/processed/lottery_market_cn.jsonl \\
        --task handicap_label \\
        --ev-threshold 0.05 \\
        --min-confidence 0.55

Output: Console table + output/backtest/handicap_test_cn_comparison.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


LABEL_TO_OUTCOME = {
    0: "home",
    1: "draw",
    2: "away",
}

OUTCOME_TO_ODDS_KEY = {
    "home": "home_odds",
    "draw": "draw_odds",
    "away": "away_odds",
}

TASK_PLAY_TYPE = {
    "handicap_label": "handicap_1x2",
    "fulltime_label": "fulltime_1x2",
}


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_market_index(market_rows: list[dict], play_type: str) -> dict:
    """Index market rows by match_id, filtering to a specific play_type."""
    index = {}
    for row in market_rows:
        if row.get("play_type") != play_type:
            continue
        mid = row["match_id"]
        if mid not in index:
            index[mid] = row
    return index


def get_odds_for_outcome(market_row: dict, outcome: str) -> float | None:
    """Extract the odds for a specific outcome from a market row."""
    mapping = {"home": "home_odds", "draw": "draw_odds", "away": "away_odds"}
    return market_row.get(mapping.get(outcome, ""))


def simulate(
    predictions: list[dict],
    eu_index: dict,
    cn_index: dict,
    task: str,
    ev_threshold: float,
    min_confidence: float,
    stake: float = 2.0,
) -> dict:
    """Run the simulation and return stats for both EU and CN odds."""
    play_type = TASK_PLAY_TYPE[task]

    n_total = 0
    n_ev_filtered = 0

    eu_bets = eu_profit = 0.0
    cn_bets = cn_profit = 0.0
    cn_covered = 0  # bets where CN SP was available

    for pred in predictions:
        match_id = pred.get("match_id")
        probs_list = pred.get("probs")  # list [home_prob, draw_prob, away_prob]
        actual = pred.get("true_label")

        if not match_id or probs_list is None or actual is None:
            continue

        n_total += 1
        eu_row = eu_index.get(match_id)
        if eu_row is None:
            continue

        # EV filtering: find best outcome by EU EV
        best_outcome = None
        best_ev = -999.0
        best_prob = 0.0

        for class_idx, prob in enumerate(probs_list):
            outcome = LABEL_TO_OUTCOME.get(class_idx)
            if outcome is None:
                continue
            eu_odds = get_odds_for_outcome(eu_row, outcome)
            if eu_odds is None:
                continue
            ev = prob * eu_odds - 1.0
            if ev > best_ev:
                best_ev = ev
                best_outcome = outcome
                best_prob = prob

        if best_outcome is None:
            continue
        if best_ev < ev_threshold or best_prob < min_confidence:
            continue

        n_ev_filtered += 1
        eu_odds_bet = get_odds_for_outcome(eu_row, best_outcome)

        # Convert actual label to outcome string
        actual_outcome = LABEL_TO_OUTCOME.get(actual)

        # ---- EU payout ----
        eu_bets += 1
        if actual_outcome == best_outcome:
            eu_profit += eu_odds_bet * stake - stake
        else:
            eu_profit -= stake

        # ---- CN payout ----
        cn_row = cn_index.get(match_id)
        cn_sp = None
        if cn_row is not None:
            cn_sp = get_odds_for_outcome(cn_row, best_outcome)

        cn_bets += 1
        if cn_sp is not None:
            cn_covered += 1
            if actual_outcome == best_outcome:
                cn_profit += cn_sp * stake - stake
            else:
                cn_profit -= stake
        else:
            # Fall back to EU odds
            if actual_outcome == best_outcome:
                cn_profit += eu_odds_bet * stake - stake
            else:
                cn_profit -= stake

    eu_staked = eu_bets * stake
    cn_staked = cn_bets * stake

    return {
        "total_predictions": n_total,
        "eu_bets": int(eu_bets),
        "eu_profit": round(eu_profit, 2),
        "eu_staked": round(eu_staked, 2),
        "eu_roi_pct": round(eu_profit / eu_staked * 100, 2) if eu_staked > 0 else 0,
        "cn_bets": int(cn_bets),
        "cn_covered": cn_covered,
        "cn_coverage_pct": round(cn_covered / cn_bets * 100, 1) if cn_bets > 0 else 0,
        "cn_profit": round(cn_profit, 2),
        "cn_staked": round(cn_staked, 2),
        "cn_roi_pct": round(cn_profit / cn_staked * 100, 2) if cn_staked > 0 else 0,
        "ev_threshold": ev_threshold,
        "min_confidence": min_confidence,
        "stake_per_bet": stake,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare EU odds vs CN Lottery SP in backtest")
    parser.add_argument(
        "--predictions",
        default=None,
        help="Predictions JSONL from predict_lgbm.py (default: auto from --task)",
    )
    parser.add_argument(
        "--eu-market",
        default="data/processed/lottery_market.jsonl",
        help="European odds market JSONL",
    )
    parser.add_argument(
        "--cn-market",
        default="data/processed/lottery_market_cn.jsonl",
        help="China Lottery SP market JSONL (from enrich_lottery_odds.py)",
    )
    parser.add_argument("--task", default="handicap_label", choices=list(TASK_PLAY_TYPE))
    parser.add_argument("--ev-threshold", type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--stake", type=float, default=2.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.predictions is None:
        args.predictions = f"output/predictions/{args.task}_test.jsonl"
    pred_path = root / args.predictions if not Path(args.predictions).is_absolute() else Path(args.predictions)
    eu_path = root / args.eu_market if not Path(args.eu_market).is_absolute() else Path(args.eu_market)
    cn_path = root / args.cn_market if not Path(args.cn_market).is_absolute() else Path(args.cn_market)

    print(f"Loading predictions: {pred_path}")
    predictions = load_jsonl(str(pred_path))
    print(f"  {len(predictions)} records")

    print(f"Loading EU market: {eu_path}")
    eu_rows = load_jsonl(str(eu_path))
    eu_index = build_market_index(eu_rows, TASK_PLAY_TYPE[args.task])
    print(f"  {len(eu_index)} matches indexed")

    cn_index = {}
    if cn_path.exists():
        print(f"Loading CN market: {cn_path}")
        cn_rows = load_jsonl(str(cn_path))
        cn_index = build_market_index(cn_rows, TASK_PLAY_TYPE[args.task])
        print(f"  {len(cn_index)} CN matches indexed")
    else:
        print(f"  CN market not found at {cn_path}, will use EU odds for all bets")

    results = simulate(
        predictions,
        eu_index,
        cn_index,
        task=args.task,
        ev_threshold=args.ev_threshold,
        min_confidence=args.min_confidence,
        stake=args.stake,
    )

    print("\n" + "=" * 60)
    print(f"Task: {args.task}  |  EV≥{args.ev_threshold}  conf≥{args.min_confidence}")
    print("=" * 60)
    print(f"{'Metric':<30} {'EU Odds':>12} {'CN Lottery':>12}")
    print("-" * 60)
    print(f"{'Bets placed':<30} {results['eu_bets']:>12} {results['cn_bets']:>12}")
    print(f"{'CN SP coverage':<30} {'':>12} {results['cn_coverage_pct']:>11.1f}%")
    print(f"{'Staked':<30} {'¥'+str(results['eu_staked']):>12} {'¥'+str(results['cn_staked']):>12}")
    print(f"{'Profit':<30} {'¥'+str(results['eu_profit']):>12} {'¥'+str(results['cn_profit']):>12}")
    print(f"{'ROI':<30} {str(results['eu_roi_pct'])+'%':>12} {str(results['cn_roi_pct'])+'%':>12}")
    print("=" * 60)

    if args.output:
        out = Path(args.output)
    else:
        out = root / "output" / "backtest" / f"{args.task}_cn_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
