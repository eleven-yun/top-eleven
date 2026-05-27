#!/usr/bin/env python3
"""Optimize daily exposure caps for issue-style betting.

Evaluates historical pick concentration controls on the test predictions:
- max picks per day
- max picks per league per day

Selection still uses EU EV + confidence filters; payout uses pre-match CN odds
when available, else EU odds fallback.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from cn_lottery_backtest import (
    LABEL_TO_OUTCOME,
    TASK_PLAY_TYPE,
    build_market_index,
    build_prematch_index,
    get_odds_for_outcome,
    get_prematch_odds_for_outcome,
    load_jsonl,
)


def int_list(arg: str) -> list[int]:
    return [int(x.strip()) for x in arg.split(",") if x.strip()]


def load_meta_maps(meta_rows: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    match_to_date = {}
    match_to_league = {}
    for row in meta_rows:
        mid = row.get("match_id")
        if not mid:
            continue
        match_to_date[mid] = (row.get("datetime_utc") or "")[:10]
        match_to_league[mid] = row.get("league_code") or "UNK"
    return match_to_date, match_to_league


def build_candidates(
    predictions: list[dict],
    eu_index: dict,
    prematch_index: dict,
    match_to_date: dict[str, str],
    match_to_league: dict[str, str],
    ev_threshold: float,
    min_confidence: float,
) -> list[dict]:
    candidates = []
    for pred in predictions:
        mid = pred.get("match_id")
        probs = pred.get("probs")
        true_label = pred.get("true_label")
        if not mid or probs is None or true_label is None:
            continue

        eu_row = eu_index.get(mid)
        if eu_row is None:
            continue

        best = None
        for cls, prob in enumerate(probs):
            outcome = LABEL_TO_OUTCOME.get(cls)
            if outcome is None:
                continue
            eu_odds = get_odds_for_outcome(eu_row, outcome)
            if eu_odds is None:
                continue
            ev = prob * eu_odds - 1.0
            if best is None or ev > best["ev"]:
                best = {
                    "cls": cls,
                    "outcome": outcome,
                    "prob": float(prob),
                    "ev": float(ev),
                    "eu_odds": float(eu_odds),
                }

        if best is None:
            continue
        if best["ev"] < ev_threshold or best["prob"] < min_confidence:
            continue

        p_row = prematch_index.get(mid)
        p_odds = None
        if p_row is not None:
            p_odds = get_prematch_odds_for_outcome(p_row, best["outcome"])

        payout_odds = float(p_odds) if p_odds is not None else best["eu_odds"]
        won = int(true_label) == int(best["cls"])
        profit = payout_odds * 2.0 - 2.0 if won else -2.0

        candidates.append(
            {
                "match_id": mid,
                "date": match_to_date.get(mid, ""),
                "league": match_to_league.get(mid, "UNK"),
                "ev": round(best["ev"], 6),
                "prob": round(best["prob"], 6),
                "payout_odds": payout_odds,
                "profit": round(profit, 4),
                "covered_by_prematch": p_odds is not None,
            }
        )

    return candidates


def apply_caps(cands: list[dict], max_picks: int, max_per_league: int) -> list[dict]:
    by_date = defaultdict(list)
    for c in cands:
        by_date[c["date"]].append(c)

    chosen = []
    for d in sorted(by_date):
        day = sorted(by_date[d], key=lambda x: x["ev"], reverse=True)
        league_count = defaultdict(int)
        for c in day:
            if max_picks > 0 and len([x for x in chosen if x["date"] == d]) >= max_picks:
                break
            if max_per_league > 0 and league_count[c["league"]] >= max_per_league:
                continue
            chosen.append(c)
            league_count[c["league"]] += 1
    return chosen


def evaluate(chosen: list[dict], max_picks: int, max_per_league: int) -> dict:
    bets = len(chosen)
    staked = bets * 2.0
    profit = round(sum(c["profit"] for c in chosen), 2)
    roi = round(profit / staked * 100, 2) if staked > 0 else 0.0
    coverage = round(sum(1 for c in chosen if c["covered_by_prematch"]) / bets * 100, 1) if bets else 0.0
    return {
        "max_picks": max_picks,
        "max_picks_per_league": max_per_league,
        "bets": bets,
        "staked": staked,
        "profit": profit,
        "roi_pct": roi,
        "prematch_coverage_pct": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize exposure caps for issue-style picks")
    parser.add_argument("--task", default="handicap_label", choices=list(TASK_PLAY_TYPE))
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--meta-file", default="data/processed/match_meta.jsonl")
    parser.add_argument("--eu-market", default="data/processed/lottery_market.jsonl")
    parser.add_argument("--prematch-market", default="data/processed/lottery_market_prematch_cn.jsonl")
    parser.add_argument("--ev-threshold", type=float, default=0.02)
    parser.add_argument("--min-confidence", type=float, default=0.51)
    parser.add_argument("--max-picks-grid", default="0,6,8,10,12")
    parser.add_argument("--max-per-league-grid", default="0,1,2,3")
    parser.add_argument("--min-bets", type=int, default=200)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    pred_path = (
        root / f"output/predictions/{args.task}_test.jsonl"
        if args.predictions is None
        else (root / args.predictions if not Path(args.predictions).is_absolute() else Path(args.predictions))
    )
    meta_path = root / args.meta_file if not Path(args.meta_file).is_absolute() else Path(args.meta_file)
    eu_path = root / args.eu_market if not Path(args.eu_market).is_absolute() else Path(args.eu_market)
    pre_path = root / args.prematch_market if not Path(args.prematch_market).is_absolute() else Path(args.prematch_market)

    predictions = load_jsonl(str(pred_path))
    meta_rows = load_jsonl(str(meta_path))
    eu_index = build_market_index(load_jsonl(str(eu_path)), TASK_PLAY_TYPE[args.task])
    pre_index = build_prematch_index(load_jsonl(str(pre_path)), TASK_PLAY_TYPE[args.task])
    match_to_date, match_to_league = load_meta_maps(meta_rows)

    cands = build_candidates(
        predictions=predictions,
        eu_index=eu_index,
        prematch_index=pre_index,
        match_to_date=match_to_date,
        match_to_league=match_to_league,
        ev_threshold=args.ev_threshold,
        min_confidence=args.min_confidence,
    )

    max_picks_vals = int_list(args.max_picks_grid)
    max_per_lg_vals = int_list(args.max_per_league_grid)

    rows = []
    for mp in max_picks_vals:
        for mpl in max_per_lg_vals:
            chosen = apply_caps(cands, mp, mpl)
            metrics = evaluate(chosen, mp, mpl)
            if metrics["bets"] >= args.min_bets:
                rows.append(metrics)

    rows_by_profit = sorted(rows, key=lambda r: (r["profit"], r["roi_pct"]), reverse=True)
    rows_by_roi = sorted(rows, key=lambda r: (r["roi_pct"], r["profit"]), reverse=True)

    print("Top cap settings by profit:")
    for r in rows_by_profit[:5]:
        print(
            f"  max_picks={r['max_picks']} max_per_league={r['max_picks_per_league']} | "
            f"bets={r['bets']} roi={r['roi_pct']:+.2f}% profit={r['profit']:+.2f}"
        )

    print("\nTop cap settings by ROI:")
    for r in rows_by_roi[:5]:
        print(
            f"  max_picks={r['max_picks']} max_per_league={r['max_picks_per_league']} | "
            f"bets={r['bets']} roi={r['roi_pct']:+.2f}% profit={r['profit']:+.2f}"
        )

    payload = {
        "task": args.task,
        "ev_threshold": args.ev_threshold,
        "min_confidence": args.min_confidence,
        "candidate_bets_before_caps": len(cands),
        "min_bets": args.min_bets,
        "top_by_profit": rows_by_profit[:20],
        "top_by_roi": rows_by_roi[:20],
    }

    out = (
        Path(args.output)
        if args.output
        else (root / "output" / "backtest" / f"{args.task}_exposure_caps_grid.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved exposure-cap optimization to {out}")


if __name__ == "__main__":
    main()
