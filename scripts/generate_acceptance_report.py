#!/usr/bin/env python3
"""Generate a consolidated Phase 9 acceptance report across tasks.

This script reuses cn_lottery_backtest simulation logic and writes one JSON
artifact with per-task results plus an overall acceptance verdict.
"""

import argparse
import json
import sys
from pathlib import Path

from cn_lottery_backtest import (
    TASK_PLAY_TYPE,
    build_market_index,
    build_prematch_index,
    load_jsonl,
    simulate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate consolidated Phase 9 acceptance report")
    parser.add_argument(
        "--tasks",
        default="handicap_label,fulltime_label",
        help="Comma-separated task list (subset of handicap_label,fulltime_label)",
    )
    parser.add_argument("--predictions-dir", default="output/predictions")
    parser.add_argument("--eu-market", default="data/processed/lottery_market.jsonl")
    parser.add_argument("--cn-market", default="data/processed/lottery_market_cn.jsonl")
    parser.add_argument("--prematch-market", default="data/processed/lottery_market_prematch_cn.jsonl")
    parser.add_argument("--match-meta", default="data/processed/match_meta.jsonl")
    parser.add_argument("--ev-threshold", type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--stake", type=float, default=2.0)
    parser.add_argument("--acceptance-mode", choices=["raw", "supported_universe"], default="raw")
    parser.add_argument("--acceptance-target", type=float, default=70.0)
    parser.add_argument(
        "--fail-on-overall-fail",
        action="store_true",
        help="Exit with code 1 when consolidated overall_pass is false",
    )
    parser.add_argument("--output", default="output/backtest/acceptance_report.json")
    return parser.parse_args()


def resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else root / p


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    invalid = [t for t in tasks if t not in TASK_PLAY_TYPE]
    if invalid:
        raise ValueError(f"Unsupported tasks: {invalid}")

    eu_path = resolve_path(root, args.eu_market)
    cn_path = resolve_path(root, args.cn_market)
    prematch_path = resolve_path(root, args.prematch_market)
    meta_path = resolve_path(root, args.match_meta)
    predictions_dir = resolve_path(root, args.predictions_dir)

    eu_rows = load_jsonl(str(eu_path))
    cn_rows = load_jsonl(str(cn_path)) if cn_path.exists() else []
    prematch_rows = load_jsonl(str(prematch_path)) if prematch_path.exists() else []

    meta_rows = load_jsonl(str(meta_path)) if meta_path.exists() else []
    match_to_league_code = {
        row.get("match_id"): (row.get("league_code") or "").lower()
        for row in meta_rows
        if row.get("match_id")
    }

    report = {
        "acceptance_mode": args.acceptance_mode,
        "acceptance_target_coverage_pct": args.acceptance_target,
        "ev_threshold": args.ev_threshold,
        "min_confidence": args.min_confidence,
        "stake": args.stake,
        "tasks": {},
    }

    overall_pass = True
    for task in tasks:
        play_type = TASK_PLAY_TYPE[task]
        pred_path = predictions_dir / f"{task}_test.jsonl"
        predictions = load_jsonl(str(pred_path))

        eu_index = build_market_index(eu_rows, play_type)
        cn_index = build_market_index(cn_rows, play_type)
        prematch_index = build_prematch_index(prematch_rows, play_type)

        cn_supported_leagues = {match_to_league_code[mid] for mid in cn_index if match_to_league_code.get(mid)}
        prematch_supported_leagues = {
            match_to_league_code[mid] for mid in prematch_index if match_to_league_code.get(mid)
        }

        result = simulate(
            predictions=predictions,
            eu_index=eu_index,
            cn_index=cn_index,
            prematch_index=prematch_index,
            match_to_league_code=match_to_league_code,
            cn_supported_leagues=cn_supported_leagues,
            prematch_supported_leagues=prematch_supported_leagues,
            task=task,
            ev_threshold=args.ev_threshold,
            min_confidence=args.min_confidence,
            stake=args.stake,
            acceptance_mode=args.acceptance_mode,
            acceptance_target=args.acceptance_target,
        )
        report["tasks"][task] = {
            "acceptance_selected_coverage_pct": result.get("acceptance_selected_coverage_pct", 0.0),
            "acceptance_selected_pass": result.get("acceptance_selected_pass", False),
            "acceptance_coverage": result.get("acceptance_coverage", {}),
            "prematch_roi_pct": result.get("prematch_roi_pct", 0.0),
            "eu_roi_pct": result.get("eu_roi_pct", 0.0),
            "bets": result.get("eu_bets", 0),
        }
        overall_pass = overall_pass and bool(result.get("acceptance_selected_pass", False))

    report["overall_pass"] = overall_pass

    out_path = resolve_path(root, args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote acceptance report to {out_path}")
    print(f"Overall acceptance: {'PASS' if overall_pass else 'FAIL'}")
    if args.fail_on_overall_fail and not overall_pass:
        print("Exiting with code 1 because overall acceptance is FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
