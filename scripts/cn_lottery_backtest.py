#!/usr/bin/env python3
"""
China Lottery backtest: compare EU proxy, CN SP, and pre-match closing payouts.

Pipeline:
1. Load model predictions (from predict_lgbm.py JSONL output)
2. Load existing European odds (lottery_market.jsonl) — used for EV filtering
3. Load China Lottery SP data (from fetch_cn_odds.py + enrich_lottery_odds.py)
4. Optionally load pre-match CN closing odds (from fetch_cn_prematch_odds.py + enrich_prematch_odds.py)
5. For each bet placed (based on European odds EV filter):
   a. Compute pure CN-covered ROI only when payout can be evaluated from matched CN settlement rows
   b. Compute a separate CN+EU fallback ROI for all bets (fallback to EU when CN payout is unavailable)
   c. Compute a pre-match closing ROI column (fallback to EU when pre-match odds are unavailable)
6. Report ROI and coverage for all columns

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


def build_prematch_index(market_rows: list[dict], play_type: str) -> dict:
    """Index pre-match market rows by match_id, filtering to a specific play_type."""
    index = {}
    for row in market_rows:
        if row.get("play_type") != play_type:
            continue
        mid = row.get("match_id")
        if mid and mid not in index:
            index[mid] = row
    return index


def get_odds_for_outcome(market_row: dict, outcome: str) -> float | None:
    """Extract the odds for a specific outcome from a market row."""
    odds_key = OUTCOME_TO_ODDS_KEY.get(outcome)
    if odds_key is None:
        return None
    return market_row.get(odds_key)


def get_prematch_odds_for_outcome(market_row: dict, outcome: str) -> float | None:
    """Extract closing odds for a specific outcome from a pre-match market row."""
    mapping = {"home": "home_odds", "draw": "draw_odds", "away": "away_odds"}
    closing = market_row.get("closing_odds") or {}
    return closing.get(mapping.get(outcome, ""))


def simulate(
    predictions: list[dict],
    eu_index: dict,
    cn_index: dict,
    prematch_index: dict,
    match_to_league_code: dict,
    cn_supported_leagues: set,
    prematch_supported_leagues: set,
    task: str,
    ev_threshold: float,
    min_confidence: float,
    stake: float = 2.0,
    kelly_fraction: float = 0.0,
    bankroll: float = 10000.0,
    kelly_min_stake: float = 2.0,
    kelly_max_stake_pct: float = 0.05,
) -> dict:
    """Run the simulation and return stats for EU, CN SP, CN+EU fallback, and pre-match odds."""
    play_type = TASK_PLAY_TYPE[task]

    n_total = 0
    n_ev_filtered = 0

    eu_bets = eu_profit = 0.0
    cn_bets = cn_profit = 0.0
    cn_fallback_bets = cn_fallback_profit = 0.0
    cn_covered = 0  # bets where a CN market row matched by match_id
    cn_supported_universe_bets = 0
    cn_supported_universe_covered = 0
    prematch_bets = prematch_profit = 0.0
    prematch_covered = 0  # bets where pre-match closing odds were available
    prematch_supported_universe_bets = 0
    prematch_supported_universe_covered = 0

    # Kelly sizing (pre-match strategy only)
    kelly_bankroll = bankroll
    kelly_peak = bankroll
    kelly_max_drawdown_pct = 0.0

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
        cn_bet_covered = False
        cn_fallback_bets += 1
        if cn_row is not None:
            # For matched CN rows, missing selected-outcome SP means selection lost
            # (scraper stores settlement SP for realized outcome only).
            if actual_outcome == best_outcome:
                cn_actual_sp = get_odds_for_outcome(cn_row, actual_outcome)
                if cn_actual_sp is not None:
                    cn_covered += 1
                    cn_bet_covered = True
                    cn_bets += 1
                    cn_profit += cn_actual_sp * stake - stake
                    cn_fallback_profit += cn_actual_sp * stake - stake
                else:
                    # Defensive guard for malformed CN rows on winning outcomes.
                    cn_fallback_profit += eu_odds_bet * stake - stake
            else:
                cn_covered += 1
                cn_bet_covered = True
                cn_bets += 1
                cn_profit -= stake
                cn_fallback_profit -= stake
        else:
            # No matched CN row: mixed column falls back to EU odds.
            if actual_outcome == best_outcome:
                cn_fallback_profit += eu_odds_bet * stake - stake
            else:
                cn_fallback_profit -= stake

        # ---- Pre-match closing payout ----
        prematch_row = prematch_index.get(match_id)
        prematch_odds = None
        if prematch_row is not None:
            prematch_odds = get_prematch_odds_for_outcome(prematch_row, best_outcome)
        prematch_bet_covered = prematch_odds is not None

        league_code = (match_to_league_code.get(match_id) or "").lower()
        if league_code in cn_supported_leagues:
            cn_supported_universe_bets += 1
            if cn_bet_covered:
                cn_supported_universe_covered += 1
        if league_code in prematch_supported_leagues:
            prematch_supported_universe_bets += 1
            if prematch_bet_covered:
                prematch_supported_universe_covered += 1

        prematch_bets += 1
        if prematch_odds is not None:
            prematch_covered += 1
            if actual_outcome == best_outcome:
                prematch_profit += prematch_odds * stake - stake
            else:
                prematch_profit -= stake
        else:
            # Fall back to EU odds
            if actual_outcome == best_outcome:
                prematch_profit += eu_odds_bet * stake - stake
            else:
                prematch_profit -= stake

        # ---- Optional Kelly bankroll simulation (pre-match line when available) ----
        if kelly_fraction > 0 and kelly_bankroll > 0:
            kelly_odds = prematch_odds if prematch_odds is not None else eu_odds_bet
            b = kelly_odds - 1.0
            p = best_prob
            if b > 0:
                kelly_full = (p * kelly_odds - 1.0) / b
                kelly_full = max(0.0, min(1.0, kelly_full))
                kelly_f = kelly_fraction * kelly_full
                stake_kelly = max(kelly_min_stake, kelly_bankroll * kelly_f)
                if kelly_max_stake_pct > 0:
                    stake_kelly = min(stake_kelly, kelly_bankroll * kelly_max_stake_pct)
                stake_kelly = min(stake_kelly, kelly_bankroll)

                if actual_outcome == best_outcome:
                    kelly_bankroll += stake_kelly * b
                else:
                    kelly_bankroll -= stake_kelly

                if kelly_bankroll > kelly_peak:
                    kelly_peak = kelly_bankroll
                if kelly_peak > 0:
                    dd_pct = (kelly_peak - kelly_bankroll) / kelly_peak * 100.0
                    if dd_pct > kelly_max_drawdown_pct:
                        kelly_max_drawdown_pct = dd_pct

    eu_staked = eu_bets * stake
    cn_staked = cn_bets * stake
    cn_fallback_staked = cn_fallback_bets * stake
    prematch_staked = prematch_bets * stake

    return {
        "total_predictions": n_total,
        "ev_filtered_predictions": n_ev_filtered,
        "eu_bets": int(eu_bets),
        "eu_profit": round(eu_profit, 2),
        "eu_staked": round(eu_staked, 2),
        "eu_roi_pct": round(eu_profit / eu_staked * 100, 2) if eu_staked > 0 else 0,
        "cn_bets": int(cn_bets),
        "cn_covered": cn_covered,
        "cn_coverage_pct": round(cn_covered / cn_fallback_bets * 100, 1) if cn_fallback_bets > 0 else 0,
        "cn_supported_universe_bets": cn_supported_universe_bets,
        "cn_supported_universe_covered": cn_supported_universe_covered,
        "cn_supported_universe_coverage_pct": round(cn_supported_universe_covered / cn_supported_universe_bets * 100, 1)
        if cn_supported_universe_bets > 0
        else 0,
        "cn_profit": round(cn_profit, 2),
        "cn_staked": round(cn_staked, 2),
        "cn_roi_pct": round(cn_profit / cn_staked * 100, 2) if cn_staked > 0 else 0,
        "cn_fallback_bets": int(cn_fallback_bets),
        "cn_fallback_profit": round(cn_fallback_profit, 2),
        "cn_fallback_staked": round(cn_fallback_staked, 2),
        "cn_fallback_roi_pct": round(cn_fallback_profit / cn_fallback_staked * 100, 2)
        if cn_fallback_staked > 0
        else 0,
        "prematch_bets": int(prematch_bets),
        "prematch_covered": prematch_covered,
        "prematch_coverage_pct": round(prematch_covered / prematch_bets * 100, 1) if prematch_bets > 0 else 0,
        "prematch_supported_universe_bets": prematch_supported_universe_bets,
        "prematch_supported_universe_covered": prematch_supported_universe_covered,
        "prematch_supported_universe_coverage_pct": round(
            prematch_supported_universe_covered / prematch_supported_universe_bets * 100, 1
        )
        if prematch_supported_universe_bets > 0
        else 0,
        "prematch_profit": round(prematch_profit, 2),
        "prematch_staked": round(prematch_staked, 2),
        "prematch_roi_pct": round(prematch_profit / prematch_staked * 100, 2) if prematch_staked > 0 else 0,
        "cn_fallback_bets": int(cn_fallback_bets),
        "cn_fallback_profit": round(cn_fallback_profit, 2),
        "cn_fallback_staked": round(cn_fallback_staked, 2),
        "cn_fallback_roi_pct": round(cn_fallback_profit / cn_fallback_staked * 100, 2)
        if cn_fallback_staked > 0
        else 0,
        "prematch_bets": int(prematch_bets),
        "prematch_covered": prematch_covered,
        "prematch_coverage_pct": round(prematch_covered / prematch_bets * 100, 1) if prematch_bets > 0 else 0,
        "prematch_profit": round(prematch_profit, 2),
        "prematch_staked": round(prematch_staked, 2),
        "prematch_roi_pct": round(prematch_profit / prematch_staked * 100, 2) if prematch_staked > 0 else 0,
        "kelly_enabled": kelly_fraction > 0,
        "kelly_fraction": kelly_fraction,
        "kelly_max_stake_pct": kelly_max_stake_pct,
        "kelly_initial_bankroll": round(bankroll, 2),
        "kelly_final_bankroll": round(kelly_bankroll, 2),
        "kelly_profit": round(kelly_bankroll - bankroll, 2),
        "kelly_roi_pct": round((kelly_bankroll - bankroll) / bankroll * 100, 2) if bankroll > 0 else 0,
        "kelly_max_drawdown_pct": round(kelly_max_drawdown_pct, 2),
        "acceptance_target_coverage_pct": 70.0,
        "acceptance_coverage": {
            "cn": {
                "raw": {
                    "covered": cn_covered,
                    "bets": int(cn_fallback_bets),
                    "coverage_pct": round(cn_covered / cn_fallback_bets * 100, 1) if cn_fallback_bets > 0 else 0,
                },
                "supported_universe": {
                    "covered": cn_supported_universe_covered,
                    "bets": cn_supported_universe_bets,
                    "coverage_pct": round(cn_supported_universe_covered / cn_supported_universe_bets * 100, 1)
                    if cn_supported_universe_bets > 0
                    else 0,
                },
            },
            "prematch": {
                "raw": {
                    "covered": prematch_covered,
                    "bets": int(prematch_bets),
                    "coverage_pct": round(prematch_covered / prematch_bets * 100, 1) if prematch_bets > 0 else 0,
                },
                "supported_universe": {
                    "covered": prematch_supported_universe_covered,
                    "bets": prematch_supported_universe_bets,
                    "coverage_pct": round(
                        prematch_supported_universe_covered / prematch_supported_universe_bets * 100, 1
                    )
                    if prematch_supported_universe_bets > 0
                    else 0,
                },
            },
        },
        "ev_threshold": ev_threshold,
        "min_confidence": min_confidence,
        "stake_per_bet": stake,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare EU odds vs CN Lottery SP vs pre-match CN closing odds")
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
    parser.add_argument(
        "--prematch-market",
        default="data/processed/lottery_market_prematch_cn.jsonl",
        help="Pre-match CN closing market JSONL (from enrich_prematch_odds.py)",
    )
    parser.add_argument(
        "--match-meta",
        default="data/processed/match_meta.jsonl",
        help="Match metadata JSONL used to map match_id to league_code",
    )
    parser.add_argument("--task", default="handicap_label", choices=list(TASK_PLAY_TYPE))
    parser.add_argument("--ev-threshold", type=float, default=0.05)
    parser.add_argument("--min-confidence", type=float, default=0.55)
    parser.add_argument("--stake", type=float, default=2.0)
    parser.add_argument("--kelly-fraction", type=float, default=0.0, help="0 disables Kelly sizing")
    parser.add_argument("--bankroll", type=float, default=10000.0, help="Initial bankroll for Kelly simulation")
    parser.add_argument("--kelly-min-stake", type=float, default=2.0, help="Minimum Kelly stake per bet")
    parser.add_argument(
        "--kelly-max-stake-pct",
        type=float,
        default=0.05,
        help="Cap Kelly stake as fraction of bankroll per bet (e.g., 0.05 = 5%%)",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    if args.predictions is None:
        args.predictions = f"output/predictions/{args.task}_test.jsonl"
    pred_path = root / args.predictions if not Path(args.predictions).is_absolute() else Path(args.predictions)
    eu_path = root / args.eu_market if not Path(args.eu_market).is_absolute() else Path(args.eu_market)
    cn_path = root / args.cn_market if not Path(args.cn_market).is_absolute() else Path(args.cn_market)
    prematch_path = root / args.prematch_market if not Path(args.prematch_market).is_absolute() else Path(args.prematch_market)
    meta_path = root / args.match_meta if not Path(args.match_meta).is_absolute() else Path(args.match_meta)

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

    prematch_index = {}
    if prematch_path.exists():
        print(f"Loading pre-match market: {prematch_path}")
        prematch_rows = load_jsonl(str(prematch_path))
        prematch_index = build_prematch_index(prematch_rows, TASK_PLAY_TYPE[args.task])
        print(f"  {len(prematch_index)} pre-match matches indexed")
    else:
        print(f"  Pre-match market not found at {prematch_path}, will use EU odds for all bets")

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
    else:
        print(f"  Match meta not found at {meta_path}, supported-universe coverage will be 0")

    results = simulate(
        predictions,
        eu_index,
        cn_index,
        prematch_index,
        match_to_league_code,
        cn_supported_leagues,
        prematch_supported_leagues,
        task=args.task,
        ev_threshold=args.ev_threshold,
        min_confidence=args.min_confidence,
        stake=args.stake,
        kelly_fraction=args.kelly_fraction,
        bankroll=args.bankroll,
        kelly_min_stake=args.kelly_min_stake,
        kelly_max_stake_pct=args.kelly_max_stake_pct,
    )

    print("\n" + "=" * 60)
    print(f"Task: {args.task}  |  EV≥{args.ev_threshold}  conf≥{args.min_confidence}")
    print("=" * 60)
    print(f"{'Metric':<30} {'EU Odds':>12} {'CN Lottery':>12} {'CN+EU fallback':>16} {'Pre-match':>12}")
    print("-" * 60)
    print(
        f"{'Bets placed':<30} {results['eu_bets']:>12} {results['cn_bets']:>12} "
        f"{results['cn_fallback_bets']:>16} {results['prematch_bets']:>12}"
    )
    print(
        f"{'Coverage':<30} {'':>12} {results['cn_coverage_pct']:>11.1f}% "
        f"{'':>16} {results['prematch_coverage_pct']:>11.1f}%"
    )
    print(
        f"{'Supported coverage':<30} {'':>12} {results['cn_supported_universe_coverage_pct']:>11.1f}% "
        f"{'':>16} {results['prematch_supported_universe_coverage_pct']:>11.1f}%"
    )
    print(
        f"{'Staked':<30} {'¥'+str(results['eu_staked']):>12} {'¥'+str(results['cn_staked']):>12} "
        f"{'¥'+str(results['cn_fallback_staked']):>16} {'¥'+str(results['prematch_staked']):>12}"
    )
    print(
        f"{'Profit':<30} {'¥'+str(results['eu_profit']):>12} {'¥'+str(results['cn_profit']):>12} "
        f"{'¥'+str(results['cn_fallback_profit']):>16} {'¥'+str(results['prematch_profit']):>12}"
    )
    print(
        f"{'ROI':<30} {str(results['eu_roi_pct'])+'%':>12} {str(results['cn_roi_pct'])+'%':>12} "
        f"{str(results['cn_fallback_roi_pct'])+'%':>16} {str(results['prematch_roi_pct'])+'%':>12}"
    )
    if results.get("kelly_enabled"):
        print("-" * 60)
        print(
            f"Kelly ({results['kelly_fraction']:.2f}x): bankroll ¥{results['kelly_initial_bankroll']} "
            f"-> ¥{results['kelly_final_bankroll']} | ROI {results['kelly_roi_pct']}% | "
            f"Cap {results['kelly_max_stake_pct']*100:.1f}% | "
            f"MaxDD {results['kelly_max_drawdown_pct']}%"
        )
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
