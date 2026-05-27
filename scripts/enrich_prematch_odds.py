#!/usr/bin/env python3
"""Enrich pre-match closing odds with match metadata.

Reads raw pre-match closing odds (from fetch_cn_prematch_odds.py) and joins
them to processed match_meta.jsonl by fuzzy matching on team names, kickoff
time, and league. Writes lottery_market_prematch_cn.jsonl compatible with
the backtest script.

Input files:
    data/raw/china_lottery/prematch_odds_raw.jsonl  — raw pre-match odds
    data/processed/match_meta.jsonl                 — canonical match records
    config/team_alias_cn.json                       — team name alias map

Output files:
    data/processed/lottery_market_prematch_cn.jsonl — enriched pre-match odds
    data/processed/prematch_match_report.json       — match coverage report

Usage:
    python scripts/enrich_prematch_odds.py
    python scripts/enrich_prematch_odds.py --dry-run
"""

import argparse
import json
import os

from enrich_lottery_odds import build_alias_index, load_json_or_jsonl, match_odds_to_meta, write_json


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def enrich_meta_team_names(meta_rows, prematch_rows):
    team_id_to_name = {}
    for row in prematch_rows:
        for side in ("home", "away"):
            features = row.get(side) or {}
            team_id = features.get("team_id")
            team_name = features.get("team_name")
            if team_id and team_name and team_id not in team_id_to_name:
                team_id_to_name[team_id] = team_name

    for row in meta_rows:
        if not row.get("home_team_name"):
            row["home_team_name"] = team_id_to_name.get(row.get("home_team_id"), "")
        if not row.get("away_team_name"):
            row["away_team_name"] = team_id_to_name.get(row.get("away_team_id"), "")

    return team_id_to_name


def transform_prematch_rows(odds_rows):
    transformed = []
    opening_index = {}
    for row in odds_rows:
        key = (row.get("zid"), row.get("play_type"))
        opening_index[key] = {
            "home_odds": row.get("opening_win"),
            "draw_odds": row.get("opening_draw"),
            "away_odds": row.get("opening_lost"),
        }
        transformed.append(
            {
                "source_site": row.get("source_site", "500.com"),
                "source_match_id": row.get("zid"),
                "issue_no": row.get("zid"),
                "play_type_raw": row.get("play_type") or row.get("play_type_raw"),
                "league_name_raw": row.get("league_name_raw"),
                "kickoff_local": row.get("kickoff_local"),
                "home_team_raw": row.get("home_team_raw"),
                "away_team_raw": row.get("away_team_raw"),
                "handicap_line_raw": row.get("handicap_line_raw"),
                "odds_home": row.get("closing_win"),
                "odds_draw": row.get("closing_draw"),
                "odds_away": row.get("closing_lost"),
                "odds_capture_time": row.get("closing_time"),
                "odds_stage": "close",
                "date_local": row.get("date_local"),
            }
        )
    return transformed, opening_index


def build_enriched_records(raw_rows, matched_rows, opening_index):
    raw_index = {(row.get("zid"), row.get("play_type")): row for row in raw_rows}
    enriched = []
    for match in matched_rows:
        source_match_id = match.get("issue_id")
        raw = raw_index.get((source_match_id, match.get("play_type")))
        if raw is None:
            continue
        opening = opening_index.get((raw.get("zid"), raw.get("play_type")), {})
        enriched.append(
            {
                "source_site": raw.get("source_site", "500.com"),
                "zid": raw.get("zid"),
                "match_id": match.get("match_id"),
                "date_local": raw.get("date_local"),
                "play_type": match.get("play_type"),
                "odds_stage": "close",
                "closing_odds": {
                    "home_odds": match.get("home_odds"),
                    "draw_odds": match.get("draw_odds"),
                    "away_odds": match.get("away_odds"),
                },
                "opening_odds": opening,
                "odds_capture_time": match.get("odds_capture_time"),
                "match_confidence_score": match.get("match_confidence_score"),
            }
        )
    return enriched


def build_report(raw_rows, matched_rows, unresolved_rows):
    report = {
        "total_odds_rows": len(raw_rows),
        "matched": len(matched_rows),
        "skipped_low_score": sum(1 for row in unresolved_rows if str(row.get("reason", "")).startswith("low_")),
        "skipped_no_candidate": sum(
            1 for row in unresolved_rows if row.get("reason") in {"no_candidates", "no_scored_candidates"}
        ),
        "by_play_type": {},
    }
    for row in raw_rows:
        play_type = row.get("play_type")
        report["by_play_type"].setdefault(play_type, {"total": 0, "matched": 0})
        report["by_play_type"][play_type]["total"] += 1
    for row in matched_rows:
        play_type = row.get("play_type")
        report["by_play_type"].setdefault(play_type, {"total": 0, "matched": 0})
        report["by_play_type"][play_type]["matched"] += 1
    return report


def main():
    parser = argparse.ArgumentParser(description="Enrich pre-match closing odds with match metadata")
    parser.add_argument(
        "--odds-file",
        default="data/raw/china_lottery/prematch_odds_raw.jsonl",
        help="Raw pre-match odds JSONL",
    )
    parser.add_argument(
        "--meta-file",
        default="data/processed/match_meta.jsonl",
        help="Match metadata JSONL",
    )
    parser.add_argument(
        "--alias-file",
        default="config/team_alias_cn.json",
        help="Team alias config",
    )
    parser.add_argument(
        "--prematch-file",
        default="data/processed/prematch_features.jsonl",
        help="Prematch features JSONL used to map team IDs to team names",
    )
    parser.add_argument(
        "--output",
        default="data/processed/lottery_market_prematch_cn.jsonl",
        help="Output JSONL for enriched pre-match odds",
    )
    parser.add_argument(
        "--report",
        default="data/processed/prematch_match_report.json",
        help="Match coverage report JSON",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.50,
        help="Minimum matching score threshold",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't write output files",
    )

    args = parser.parse_args()

    print(f"Loading pre-match odds from {args.odds_file}...")
    odds_rows = load_json_or_jsonl(args.odds_file)
    print(f"  → {len(odds_rows)} rows")

    print(f"Loading match metadata from {args.meta_file}...")
    meta_rows = load_json_or_jsonl(args.meta_file)
    print(f"  → {len(meta_rows)} rows")

    print(f"Loading prematch features from {args.prematch_file}...")
    prematch_rows = load_json_or_jsonl(args.prematch_file)
    team_id_to_name = enrich_meta_team_names(meta_rows, prematch_rows)
    print(f"  → {len(team_id_to_name)} team name mappings")

    print(f"Loading team aliases from {args.alias_file}...")
    with open(args.alias_file, encoding="utf-8") as f:
        alias_config = json.load(f)
    alias_index = build_alias_index(alias_config)

    print(f"\nMatching odds to metadata (min_score={args.min_score})...")
    transformed_rows, opening_index = transform_prematch_rows(odds_rows)
    matched_rows, unresolved_rows = match_odds_to_meta(
        transformed_rows,
        meta_rows,
        alias_index,
        min_score=args.min_score,
        min_gap=0.10,
    )
    matched = build_enriched_records(odds_rows, matched_rows, opening_index)
    report = build_report(odds_rows, matched, unresolved_rows)

    print(f"\nReport:")
    print(f"  Total odds rows: {report['total_odds_rows']}")
    print(f"  Matched: {report['matched']} ({100*report['matched']/max(1, report['total_odds_rows']):.1f}%)")
    print(f"  Skipped (low score): {report['skipped_low_score']}")
    print(f"  Skipped (no candidate): {report['skipped_no_candidate']}")
    print(f"\nBy play type:")
    for pt, stats in report["by_play_type"].items():
        pct = 100 * stats["matched"] / max(1, stats["total"])
        print(f"  {pt}: {stats['matched']}/{stats['total']} ({pct:.1f}%)")

    if not args.dry_run:
        write_jsonl(args.output, matched)
        print(f"\nWrote {len(matched)} enriched records to {args.output}")

        write_json(args.report, report)
        print(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
