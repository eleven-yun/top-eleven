#!/usr/bin/env python3
"""Walk-forward monthly retuning for CN lottery strategy thresholds.

For each month m (after a warmup period):
1. Tune EV/conf thresholds on all months < m.
2. Evaluate selected thresholds on month m only.

This provides a more deployment-like estimate than tuning on the full period.
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


def month_key(dt: str) -> str:
    return (dt or "")[:7]


def load_match_months(meta_path: Path) -> dict[str, str]:
    mapping = {}
    for row in load_jsonl(str(meta_path)):
        mapping[row.get("match_id")] = month_key(row.get("datetime_utc", ""))
    return mapping


def select_best_thresholds(
    train_predictions: list[dict],
    eu_index: dict,
    cn_index: dict,
    prematch_index: dict,
    task: str,
    ev_values: list[float],
    conf_values: list[float],
    min_train_bets: int,
    stake: float,
    objective: str,
) -> tuple[dict | None, dict | None]:
    best_cfg = None
    best_res = None
    best_score = -1e18

    for ev in ev_values:
        for conf in conf_values:
            res = simulate(
                predictions=train_predictions,
                eu_index=eu_index,
                cn_index=cn_index,
                prematch_index=prematch_index,
                task=task,
                ev_threshold=ev,
                min_confidence=conf,
                stake=stake,
            )
            bets = res["prematch_bets"]
            if bets < min_train_bets:
                continue

            if objective == "profit":
                score = res["prematch_profit"]
            elif objective == "roi":
                score = res["prematch_roi_pct"]
            else:
                score = res["prematch_roi_pct"] * ((bets / 100.0) ** 0.5)

            if score > best_score:
                best_score = score
                best_cfg = {"ev_threshold": ev, "min_confidence": conf, "score": round(score, 4)}
                best_res = res

    return best_cfg, best_res


def summarize_monthly_results(monthly_rows: list[dict], stake: float) -> dict:
    if not monthly_rows:
        return {
            "months_evaluated": 0,
            "positive_months": 0,
            "negative_months": 0,
            "total_bets": 0,
            "total_staked": 0.0,
            "total_profit": 0.0,
            "roi_pct": 0.0,
            "avg_monthly_roi_pct": 0.0,
            "worst_month_roi_pct": 0.0,
            "best_month_roi_pct": 0.0,
        }

    rois = [r["test"]["prematch_roi_pct"] for r in monthly_rows if r["test"]["prematch_bets"] > 0]
    profits = [r["test"]["prematch_profit"] for r in monthly_rows]
    bets = [r["test"]["prematch_bets"] for r in monthly_rows]
    total_bets = sum(bets)
    total_staked = total_bets * stake
    total_profit = sum(profits)

    return {
        "months_evaluated": len(monthly_rows),
        "positive_months": sum(1 for x in rois if x > 0),
        "negative_months": sum(1 for x in rois if x < 0),
        "total_bets": int(total_bets),
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi_pct": round(total_profit / total_staked * 100, 2) if total_staked > 0 else 0.0,
        "avg_monthly_roi_pct": round(sum(rois) / len(rois), 2) if rois else 0.0,
        "worst_month_roi_pct": round(min(rois), 2) if rois else 0.0,
        "best_month_roi_pct": round(max(rois), 2) if rois else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward monthly retuning for EV/conf thresholds")
    parser.add_argument("--task", default="handicap_label", choices=list(TASK_PLAY_TYPE))
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--eu-market", default="data/processed/lottery_market.jsonl")
    parser.add_argument("--cn-market", default="data/processed/lottery_market_cn.jsonl")
    parser.add_argument("--prematch-market", default="data/processed/lottery_market_prematch_cn.jsonl")
    parser.add_argument("--meta-file", default="data/processed/match_meta.jsonl")
    parser.add_argument("--ev-start", type=float, default=0.02)
    parser.add_argument("--ev-stop", type=float, default=0.16)
    parser.add_argument("--ev-step", type=float, default=0.01)
    parser.add_argument("--conf-start", type=float, default=0.50)
    parser.add_argument("--conf-stop", type=float, default=0.70)
    parser.add_argument("--conf-step", type=float, default=0.01)
    parser.add_argument("--min-train-months", type=int, default=3)
    parser.add_argument("--min-train-bets", type=int, default=150)
    parser.add_argument("--stake", type=float, default=2.0)
    parser.add_argument("--objective", choices=["profit", "roi", "risk_adjusted"], default="profit")
    parser.add_argument("--static-ev", type=float, default=0.02)
    parser.add_argument("--static-conf", type=float, default=0.51)
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

    predictions_by_month: dict[str, list[dict]] = {}
    for p in predictions:
        mid = p.get("match_id")
        if not mid:
            continue
        m = match_to_month.get(mid)
        if not m:
            continue
        predictions_by_month.setdefault(m, []).append(p)

    months = sorted(predictions_by_month)
    ev_values = frange(args.ev_start, args.ev_stop, args.ev_step)
    conf_values = frange(args.conf_start, args.conf_stop, args.conf_step)

    monthly_rows = []
    static_rows = []
    for i, month in enumerate(months):
        if i < args.min_train_months:
            continue

        train_months = months[:i]
        train_predictions = []
        for tm in train_months:
            train_predictions.extend(predictions_by_month[tm])

        best_cfg, train_best_res = select_best_thresholds(
            train_predictions=train_predictions,
            eu_index=eu_index,
            cn_index=cn_index,
            prematch_index=prematch_index,
            task=args.task,
            ev_values=ev_values,
            conf_values=conf_values,
            min_train_bets=args.min_train_bets,
            stake=args.stake,
            objective=args.objective,
        )
        if best_cfg is None or train_best_res is None:
            continue

        test_predictions = predictions_by_month[month]
        test_res = simulate(
            predictions=test_predictions,
            eu_index=eu_index,
            cn_index=cn_index,
            prematch_index=prematch_index,
            task=args.task,
            ev_threshold=best_cfg["ev_threshold"],
            min_confidence=best_cfg["min_confidence"],
            stake=args.stake,
        )

        static_res = simulate(
            predictions=test_predictions,
            eu_index=eu_index,
            cn_index=cn_index,
            prematch_index=prematch_index,
            task=args.task,
            ev_threshold=args.static_ev,
            min_confidence=args.static_conf,
            stake=args.stake,
        )

        row = {
            "month": month,
            "train_months": train_months,
            "selected": best_cfg,
            "train_best": {
                "prematch_bets": train_best_res["prematch_bets"],
                "prematch_roi_pct": train_best_res["prematch_roi_pct"],
                "prematch_profit": train_best_res["prematch_profit"],
            },
            "test": {
                "prematch_bets": test_res["prematch_bets"],
                "prematch_roi_pct": test_res["prematch_roi_pct"],
                "prematch_profit": test_res["prematch_profit"],
                "prematch_coverage_pct": test_res["prematch_coverage_pct"],
            },
            "static_test": {
                "ev_threshold": args.static_ev,
                "min_confidence": args.static_conf,
                "prematch_bets": static_res["prematch_bets"],
                "prematch_roi_pct": static_res["prematch_roi_pct"],
                "prematch_profit": static_res["prematch_profit"],
            },
        }
        monthly_rows.append(row)
        static_rows.append({"test": row["static_test"]})

    dynamic_summary = summarize_monthly_results(monthly_rows, args.stake)
    static_summary = summarize_monthly_results(static_rows, args.stake)

    print("Walk-forward monthly summary:")
    print(
        "  dynamic retune | "
        f"months={dynamic_summary['months_evaluated']} bets={dynamic_summary['total_bets']} "
        f"profit={dynamic_summary['total_profit']:+.2f} roi={dynamic_summary['roi_pct']:+.2f}% "
        f"pos={dynamic_summary['positive_months']} neg={dynamic_summary['negative_months']}"
    )
    print(
        "  static profile | "
        f"months={static_summary['months_evaluated']} bets={static_summary['total_bets']} "
        f"profit={static_summary['total_profit']:+.2f} roi={static_summary['roi_pct']:+.2f}% "
        f"pos={static_summary['positive_months']} neg={static_summary['negative_months']}"
    )

    payload = {
        "task": args.task,
        "search_space": {
            "ev": [args.ev_start, args.ev_stop, args.ev_step],
            "confidence": [args.conf_start, args.conf_stop, args.conf_step],
            "objective": args.objective,
            "min_train_months": args.min_train_months,
            "min_train_bets": args.min_train_bets,
            "stake": args.stake,
        },
        "baseline_static": {
            "ev_threshold": args.static_ev,
            "min_confidence": args.static_conf,
            "summary": static_summary,
        },
        "dynamic_walk_forward": {
            "summary": dynamic_summary,
            "months": monthly_rows,
        },
    }

    out = (
        Path(args.output)
        if args.output
        else (root / "output" / "backtest" / f"{args.task}_walkforward_monthly.json")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved walk-forward report to {out}")


if __name__ == "__main__":
    main()
