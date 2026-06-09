#!/usr/bin/env python3
"""Grid-search EV/confidence thresholds for CN lottery strategy tuning.

Optimizes threshold settings using 2025/26 test predictions and reports the
best combinations for EU proxy, CN SP payout, and CN pre-match closing payout.
"""

import argparse
import json
from pathlib import Path

from cn_lottery_backtest import (
    TASK_PLAY_TYPE,
    build_market_index,
    build_prematch_index,
    load_jsonl,
    simulate,
)


def frange(start: float, stop: float, step: float) -> list[float]:
    vals = []
    x = start
    while x <= stop + 1e-9:
        vals.append(round(x, 4))
        x += step
    return vals


def run_grid(
    predictions: list[dict],
    eu_index: dict,
    cn_index: dict,
    prematch_index: dict,
    match_to_league_code: dict,
    cn_supported_leagues: set,
    prematch_supported_leagues: set,
    task: str,
    ev_values: list[float],
    conf_values: list[float],
    min_bets: int,
    stake: float,
) -> list[dict]:
    rows = []
    for ev in ev_values:
        for conf in conf_values:
            res = simulate(
                predictions=predictions,
                eu_index=eu_index,
                cn_index=cn_index,
                prematch_index=prematch_index,
                match_to_league_code=match_to_league_code,
                cn_supported_leagues=cn_supported_leagues,
                prematch_supported_leagues=prematch_supported_leagues,
                task=task,
                ev_threshold=ev,
                min_confidence=conf,
                stake=stake,
            )
            if res["prematch_bets"] < min_bets:
                continue
            # A simple risk-adjusted score favoring higher ROI and sufficient sample size.
            score = res["prematch_roi_pct"] * ((res["prematch_bets"] / 100.0) ** 0.5)
            rows.append(
                {
                    "ev_threshold": ev,
                    "min_confidence": conf,
                    "prematch_bets": res["prematch_bets"],
                    "prematch_roi_pct": res["prematch_roi_pct"],
                    "prematch_profit": res["prematch_profit"],
                    "prematch_coverage_pct": res["prematch_coverage_pct"],
                    "prematch_supported_universe_coverage_pct": res[
                        "prematch_supported_universe_coverage_pct"
                    ],
                    "cn_roi_pct": res["cn_roi_pct"],
                    "eu_roi_pct": res["eu_roi_pct"],
                    "risk_adjusted_score": round(score, 4),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize CN strategy thresholds")
    parser.add_argument("--task", default="handicap_label", choices=list(TASK_PLAY_TYPE))
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--eu-market", default="data/processed/lottery_market.jsonl")
    parser.add_argument("--cn-market", default="data/processed/lottery_market_cn.jsonl")
    parser.add_argument("--prematch-market", default="data/processed/lottery_market_prematch_cn.jsonl")
    parser.add_argument("--match-meta", default="data/processed/match_meta.jsonl")
    parser.add_argument("--ev-start", type=float, default=0.02)
    parser.add_argument("--ev-stop", type=float, default=0.16)
    parser.add_argument("--ev-step", type=float, default=0.01)
    parser.add_argument("--conf-start", type=float, default=0.50)
    parser.add_argument("--conf-stop", type=float, default=0.70)
    parser.add_argument("--conf-step", type=float, default=0.01)
    parser.add_argument("--min-bets", type=int, default=300)
    parser.add_argument("--stake", type=float, default=2.0)
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: output/backtest/<task>_strategy_grid.json)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    pred_path = (
        root / f"output/predictions/{args.task}_test.jsonl"
        if args.predictions is None
        else (root / args.predictions if not Path(args.predictions).is_absolute() else Path(args.predictions))
    )
    eu_path = root / args.eu_market if not Path(args.eu_market).is_absolute() else Path(args.eu_market)
    cn_path = root / args.cn_market if not Path(args.cn_market).is_absolute() else Path(args.cn_market)
    prematch_path = (
        root / args.prematch_market
        if not Path(args.prematch_market).is_absolute()
        else Path(args.prematch_market)
    )
    meta_path = root / args.match_meta if not Path(args.match_meta).is_absolute() else Path(args.match_meta)

    predictions = load_jsonl(str(pred_path))
    eu_index = build_market_index(load_jsonl(str(eu_path)), TASK_PLAY_TYPE[args.task])
    cn_index = build_market_index(load_jsonl(str(cn_path)), TASK_PLAY_TYPE[args.task])
    prematch_index = build_prematch_index(load_jsonl(str(prematch_path)), TASK_PLAY_TYPE[args.task])

    match_to_league_code = {}
    cn_supported_leagues = set()
    prematch_supported_leagues = set()
    if meta_path.exists():
        meta_rows = load_jsonl(str(meta_path))
        match_to_league_code = {
            row.get("match_id"): (row.get("league_code") or "").lower()
            for row in meta_rows
            if row.get("match_id")
        }
        cn_supported_leagues = {
            match_to_league_code[mid] for mid in cn_index if match_to_league_code.get(mid)
        }
        prematch_supported_leagues = {
            match_to_league_code[mid] for mid in prematch_index if match_to_league_code.get(mid)
        }

    ev_values = frange(args.ev_start, args.ev_stop, args.ev_step)
    conf_values = frange(args.conf_start, args.conf_stop, args.conf_step)

    rows = run_grid(
        predictions=predictions,
        eu_index=eu_index,
        cn_index=cn_index,
        prematch_index=prematch_index,
        match_to_league_code=match_to_league_code,
        cn_supported_leagues=cn_supported_leagues,
        prematch_supported_leagues=prematch_supported_leagues,
        task=args.task,
        ev_values=ev_values,
        conf_values=conf_values,
        min_bets=args.min_bets,
        stake=args.stake,
    )

    rows.sort(key=lambda r: (r["prematch_profit"], r["prematch_roi_pct"]), reverse=True)
    best_profit = rows[:10]
    best_roi = sorted(rows, key=lambda r: (r["prematch_roi_pct"], r["prematch_profit"]), reverse=True)[:10]
    best_score = sorted(rows, key=lambda r: r["risk_adjusted_score"], reverse=True)[:10]

    print("Top 5 by pre-match profit:")
    for r in best_profit[:5]:
        print(
            f"  EV>={r['ev_threshold']:.2f}, conf>={r['min_confidence']:.2f} | "
            f"bets={r['prematch_bets']} roi={r['prematch_roi_pct']:+.2f}% "
            f"profit={r['prematch_profit']:+.2f} coverage={r['prematch_coverage_pct']:.1f}% "
            f"supported={r['prematch_supported_universe_coverage_pct']:.1f}%"
        )

    print("\nTop 5 by pre-match ROI:")
    for r in best_roi[:5]:
        print(
            f"  EV>={r['ev_threshold']:.2f}, conf>={r['min_confidence']:.2f} | "
            f"bets={r['prematch_bets']} roi={r['prematch_roi_pct']:+.2f}% "
            f"profit={r['prematch_profit']:+.2f} coverage={r['prematch_coverage_pct']:.1f}%"
        )

    print("\nTop 5 by risk-adjusted score:")
    for r in best_score[:5]:
        print(
            f"  EV>={r['ev_threshold']:.2f}, conf>={r['min_confidence']:.2f} | "
            f"score={r['risk_adjusted_score']:.2f} bets={r['prematch_bets']} "
            f"roi={r['prematch_roi_pct']:+.2f}% profit={r['prematch_profit']:+.2f}"
        )

    payload = {
        "task": args.task,
        "search_space": {
            "ev": [args.ev_start, args.ev_stop, args.ev_step],
            "confidence": [args.conf_start, args.conf_stop, args.conf_step],
            "min_bets": args.min_bets,
            "stake": args.stake,
        },
        "num_candidates": len(rows),
        "top_by_profit": best_profit,
        "top_by_roi": best_roi,
        "top_by_risk_adjusted": best_score,
    }

    if args.output:
        out = Path(args.output)
    else:
        out = root / "output" / "backtest" / f"{args.task}_strategy_grid.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved grid results to {out}")


if __name__ == "__main__":
    main()
