#!/usr/bin/env python3
"""Monthly robustness checks for CN lottery strategy profiles.

Evaluates profile stability across month buckets to reduce overfitting risk from
single aggregate test metrics.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from cn_lottery_backtest import (
    TASK_PLAY_TYPE,
    build_market_index,
    build_prematch_index,
    load_jsonl,
    simulate,
)


def month_key(dt: str) -> str:
    return (dt or "")[:7]


def load_match_months(meta_path: Path) -> dict[str, str]:
    mapping = {}
    for row in load_jsonl(str(meta_path)):
        mapping[row.get("match_id")] = month_key(row.get("datetime_utc", ""))
    return mapping


def filter_predictions(predictions: list[dict], match_to_month: dict[str, str], month: str) -> list[dict]:
    out = []
    for p in predictions:
        mid = p.get("match_id")
        if mid and match_to_month.get(mid) == month:
            out.append(p)
    return out


def summarize_monthly(month_results: list[dict]) -> dict:
    if not month_results:
        return {
            "months": 0,
            "positive_months": 0,
            "negative_months": 0,
            "avg_monthly_roi_pct": 0.0,
            "worst_month_roi_pct": 0.0,
            "best_month_roi_pct": 0.0,
        }
    rois = [r["prematch_roi_pct"] for r in month_results if r["prematch_bets"] > 0]
    positive = sum(1 for r in rois if r > 0)
    negative = sum(1 for r in rois if r < 0)
    return {
        "months": len(rois),
        "positive_months": positive,
        "negative_months": negative,
        "avg_monthly_roi_pct": round(sum(rois) / len(rois), 2) if rois else 0.0,
        "worst_month_roi_pct": round(min(rois), 2) if rois else 0.0,
        "best_month_roi_pct": round(max(rois), 2) if rois else 0.0,
    }


def evaluate_profile(
    name: str,
    predictions: list[dict],
    match_to_month: dict[str, str],
    months: list[str],
    eu_index: dict,
    cn_index: dict,
    prematch_index: dict,
    task: str,
    ev_threshold: float,
    min_confidence: float,
    stake: float,
) -> dict:
    month_results = []
    for m in months:
        preds_m = filter_predictions(predictions, match_to_month, m)
        if not preds_m:
            continue
        res = simulate(
            predictions=preds_m,
            eu_index=eu_index,
            cn_index=cn_index,
            prematch_index=prematch_index,
            task=task,
            ev_threshold=ev_threshold,
            min_confidence=min_confidence,
            stake=stake,
        )
        res["month"] = m
        month_results.append(res)

    agg = summarize_monthly(month_results)
    return {
        "profile": {
            "name": name,
            "ev_threshold": ev_threshold,
            "min_confidence": min_confidence,
        },
        "summary": agg,
        "months": month_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly robustness checks for strategy profiles")
    parser.add_argument("--task", default="handicap_label", choices=list(TASK_PLAY_TYPE))
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--eu-market", default="data/processed/lottery_market.jsonl")
    parser.add_argument("--cn-market", default="data/processed/lottery_market_cn.jsonl")
    parser.add_argument("--prematch-market", default="data/processed/lottery_market_prematch_cn.jsonl")
    parser.add_argument("--meta-file", default="data/processed/match_meta.jsonl")
    parser.add_argument("--stake", type=float, default=2.0)
    parser.add_argument("--baseline-ev", type=float, default=0.05)
    parser.add_argument("--baseline-conf", type=float, default=0.55)
    parser.add_argument("--profit-ev", type=float, default=0.02)
    parser.add_argument("--profit-conf", type=float, default=0.51)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    pred_path = (
        root / f"output/predictions/{args.task}_test.jsonl"
        if args.predictions is None
        else (root / args.predictions if not Path(args.predictions).is_absolute() else Path(args.predictions))
    )
    eu_path = root / args.eu_market if not Path(args.eu_market).is_absolute() else Path(args.eu_market)
    cn_path = root / args.cn_market if not Path(args.cn_market).is_absolute() else Path(args.cn_market)
    prematch_path = root / args.prematch_market if not Path(args.prematch_market).is_absolute() else Path(args.prematch_market)
    meta_path = root / args.meta_file if not Path(args.meta_file).is_absolute() else Path(args.meta_file)

    predictions = load_jsonl(str(pred_path))
    eu_index = build_market_index(load_jsonl(str(eu_path)), TASK_PLAY_TYPE[args.task])
    cn_index = build_market_index(load_jsonl(str(cn_path)), TASK_PLAY_TYPE[args.task])
    prematch_index = build_prematch_index(load_jsonl(str(prematch_path)), TASK_PLAY_TYPE[args.task])

    match_to_month = load_match_months(meta_path)
    months = sorted({match_to_month.get(p.get("match_id", ""), "") for p in predictions if p.get("match_id")})
    months = [m for m in months if m]

    baseline = evaluate_profile(
        name="baseline",
        predictions=predictions,
        match_to_month=match_to_month,
        months=months,
        eu_index=eu_index,
        cn_index=cn_index,
        prematch_index=prematch_index,
        task=args.task,
        ev_threshold=args.baseline_ev,
        min_confidence=args.baseline_conf,
        stake=args.stake,
    )

    profit = evaluate_profile(
        name="profit_max",
        predictions=predictions,
        match_to_month=match_to_month,
        months=months,
        eu_index=eu_index,
        cn_index=cn_index,
        prematch_index=prematch_index,
        task=args.task,
        ev_threshold=args.profit_ev,
        min_confidence=args.profit_conf,
        stake=args.stake,
    )

    print("Monthly robustness summary:")
    for report in (baseline, profit):
        s = report["summary"]
        p = report["profile"]
        print(
            f"  {p['name']}: EV>={p['ev_threshold']:.2f}, conf>={p['min_confidence']:.2f} | "
            f"months={s['months']} pos={s['positive_months']} neg={s['negative_months']} "
            f"avgROI={s['avg_monthly_roi_pct']:+.2f}% worst={s['worst_month_roi_pct']:+.2f}%"
        )

    payload = {
        "task": args.task,
        "months": months,
        "baseline": baseline,
        "profit_max": profit,
    }

    out = (
        Path(args.output)
        if args.output
        else (root / "output" / "backtest" / f"{args.task}_robustness_monthly.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved robustness report to {out}")


if __name__ == "__main__":
    main()
