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
from collections import Counter

from enrich_lottery_odds import (
    LEAGUE_ALIAS_CN,
    build_alias_index,
    load_json_or_jsonl,
    match_odds_to_meta,
    write_json,
)


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


def build_scope_index_from_data_config(data_config):
    leagues = set()
    countries = set()
    league_codes = set()
    competitions = data_config.get("requested_competitions", [])
    for comp in competitions:
        league = (comp.get("competition_name") or "").strip().lower()
        country = (comp.get("country_name") or "").strip().lower()
        league_code = (comp.get("league_code") or "").strip().lower()
        if league:
            leagues.add(league)
        if country:
            countries.add(country)
        if league_code:
            league_codes.add(league_code)
    return {
        "leagues": leagues,
        "countries": countries,
        "league_codes": league_codes,
    }


def build_team_name_counter(meta_rows):
    counter = Counter()
    for row in meta_rows:
        home_name = (row.get("home_team_name") or "").strip()
        away_name = (row.get("away_team_name") or "").strip()
        if home_name:
            counter[home_name] += 1
        if away_name:
            counter[away_name] += 1
    return counter


def suggest_canonical_teams(raw_team_name, canonical_counter, top_n=5):
    from enrich_lottery_odds import team_match_score

    scored = []
    for name, freq in canonical_counter.items():
        score = team_match_score(raw_team_name, name)
        if score > 0:
            scored.append((score, freq, name))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [
        {"canonical_team": name, "score": score, "frequency": freq}
        for score, freq, name in scored[:top_n]
    ]


def build_unresolved_alias_suggestions(unresolved_rows, meta_rows, scope_index, alias_index):
    canonical_counter = build_team_name_counter(meta_rows)
    suggestions = []
    seen = set()
    for row in unresolved_rows:
        if not is_scope_eligible(row, scope_index):
            continue
        home_raw = (row.get("home_team_raw") or "").strip()
        away_raw = (row.get("away_team_raw") or "").strip()

        home_needed = bool(home_raw) and home_raw.lower() not in alias_index
        away_needed = bool(away_raw) and away_raw.lower() not in alias_index
        if not home_needed and not away_needed:
            continue

        dedup_key = (
            row.get("source_match_id"),
            row.get("kickoff_local"),
            row.get("league_name_raw"),
            home_raw if home_needed else "",
            away_raw if away_needed else "",
        )
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        rec = {
            "source_match_id": row.get("source_match_id"),
            "kickoff_local": row.get("kickoff_local"),
            "league_name_raw": row.get("league_name_raw"),
            "reason": row.get("reason"),
            "top_score": row.get("top_score"),
        }

        if home_needed:
            rec["home_team_raw"] = home_raw
            rec["home_team_suggestions"] = suggest_canonical_teams(home_raw, canonical_counter, top_n=5)
        if away_needed:
            rec["away_team_raw"] = away_raw
            rec["away_team_suggestions"] = suggest_canonical_teams(away_raw, canonical_counter, top_n=5)

        actionable = False
        non_actionable_reasons = []
        if home_needed:
            if rec.get("home_team_suggestions"):
                actionable = True
            else:
                non_actionable_reasons.append("home_no_canonical_candidate")
        if away_needed:
            if rec.get("away_team_suggestions"):
                actionable = True
            else:
                non_actionable_reasons.append("away_no_canonical_candidate")

        rec["actionable"] = actionable
        rec["non_actionable_reason"] = None if actionable else ",".join(non_actionable_reasons)

        suggestions.append(rec)

    return suggestions


def is_scope_eligible(row, scope_index):
    league_raw = (row.get("league_name_raw") or "").strip().lower()
    if not league_raw:
        return False
    league_resolved = LEAGUE_ALIAS_CN.get(league_raw, league_raw)
    for candidate in scope_index["leagues"]:
        if league_resolved in candidate or candidate in league_resolved:
            return True
    return league_resolved in scope_index["countries"] or league_resolved in scope_index["league_codes"]


def build_report(raw_rows, matched_rows, unresolved_rows, scope_index):
    eligible_rows = [row for row in raw_rows if is_scope_eligible(row, scope_index)]
    out_of_scope_counter = Counter(
        row.get("league_name_raw") or "unknown"
        for row in raw_rows
        if not is_scope_eligible(row, scope_index)
    )
    report = {
        "total_odds_rows": len(raw_rows),
        "eligible_odds_rows": len(eligible_rows),
        "out_of_scope_rows": len(raw_rows) - len(eligible_rows),
        "matched": len(matched_rows),
        "eligible_match_rate": round(len(matched_rows) / max(1, len(eligible_rows)), 4),
        "overall_match_rate": round(len(matched_rows) / max(1, len(raw_rows)), 4),
        "skipped_low_score": sum(1 for row in unresolved_rows if str(row.get("reason", "")).startswith("low_")),
        "skipped_no_candidate": sum(
            1 for row in unresolved_rows if row.get("reason") in {"no_candidates", "no_scored_candidates"}
        ),
        "top_out_of_scope_leagues": dict(out_of_scope_counter.most_common(10)),
        "by_play_type": {},
        "unresolved_alias_suggestion_sample": [],
        "unresolved_alias_actionable": 0,
        "unresolved_alias_non_actionable": 0,
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


def build_scope_gap_report(raw_rows, unresolved_rows, scope_index):
    out_of_scope_rows = [row for row in raw_rows if not is_scope_eligible(row, scope_index)]
    out_of_scope_counter = Counter((row.get("league_name_raw") or "unknown") for row in out_of_scope_rows)
    resolved_league_counter = Counter(
        LEAGUE_ALIAS_CN.get((row.get("league_name_raw") or "").strip().lower(), (row.get("league_name_raw") or "unknown"))
        for row in out_of_scope_rows
    )

    unresolved_by_league = Counter()
    for row in unresolved_rows:
        if not is_scope_eligible(row, scope_index):
            league = row.get("league_name_raw") or "unknown"
            unresolved_by_league[league] += 1

    examples = []
    seen = set()
    for row in out_of_scope_rows:
        key = (
            row.get("source_match_id"),
            row.get("league_name_raw"),
            row.get("home_team_raw"),
            row.get("away_team_raw"),
            row.get("kickoff_local"),
        )
        if key in seen:
            continue
        seen.add(key)
        examples.append(
            {
                "source_match_id": row.get("source_match_id"),
                "league_name_raw": row.get("league_name_raw"),
                "home_team_raw": row.get("home_team_raw"),
                "away_team_raw": row.get("away_team_raw"),
                "kickoff_local": row.get("kickoff_local"),
            }
        )
        if len(examples) >= 30:
            break

    return {
        "total_rows": len(raw_rows),
        "out_of_scope_rows": len(out_of_scope_rows),
        "out_of_scope_rate": round(len(out_of_scope_rows) / max(1, len(raw_rows)), 4),
        "top_out_of_scope_leagues": dict(out_of_scope_counter.most_common(30)),
        "top_out_of_scope_resolved_leagues": dict(resolved_league_counter.most_common(30)),
        "suggested_competition_priority": [
            {
                "competition_name": league,
                "rows": count,
                "share_of_out_of_scope": round(count / max(1, len(out_of_scope_rows)), 4),
            }
            for league, count in resolved_league_counter.most_common(20)
        ],
        "unresolved_out_of_scope_by_league": dict(unresolved_by_league.most_common(30)),
        "sample_out_of_scope_rows": examples,
    }


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
        "--data-config",
        default="config/data_config.json",
        help="Dataset scope config JSON (requested_competitions)",
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
        "--suggestions-output",
        default="data/processed/prematch_unresolved_alias_suggestions.json",
        help="Output JSON for unresolved in-scope alias suggestions",
    )
    parser.add_argument(
        "--scope-gap-output",
        default="data/processed/prematch_scope_gap_report.json",
        help="Output JSON for out-of-scope league rows (dataset expansion candidates)",
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

    print(f"Loading dataset scope config from {args.data_config}...")
    with open(args.data_config, encoding="utf-8") as f:
        data_config = json.load(f)
    scope_index = build_scope_index_from_data_config(data_config)
    print(
        "  → "
        f"{len(scope_index['leagues'])} leagues, "
        f"{len(scope_index['countries'])} countries, "
        f"{len(scope_index['league_codes'])} league codes"
    )

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
    report = build_report(odds_rows, matched, unresolved_rows, scope_index)
    unresolved_alias_suggestions = build_unresolved_alias_suggestions(
        unresolved_rows,
        meta_rows,
        scope_index,
        alias_index,
    )
    scope_gap_report = build_scope_gap_report(odds_rows, unresolved_rows, scope_index)
    report["unresolved_alias_suggestion_sample"] = unresolved_alias_suggestions[:20]
    report["unresolved_alias_actionable"] = sum(
        1 for row in unresolved_alias_suggestions if row.get("actionable")
    )
    report["unresolved_alias_non_actionable"] = len(unresolved_alias_suggestions) - report["unresolved_alias_actionable"]

    print(f"\nReport:")
    print(f"  Total odds rows: {report['total_odds_rows']}")
    print(f"  In-scope rows: {report['eligible_odds_rows']}")
    print(f"  Out-of-scope rows: {report['out_of_scope_rows']}")
    print(f"  Matched: {report['matched']} ({100*report['overall_match_rate']:.1f}% of total)")
    print(f"  Eligible coverage: {100*report['eligible_match_rate']:.1f}%")
    print(f"  Skipped (low score): {report['skipped_low_score']}")
    print(f"  Skipped (no candidate): {report['skipped_no_candidate']}")
    print(f"  Alias suggestion candidates: {len(unresolved_alias_suggestions)}")
    print(f"    Actionable: {report['unresolved_alias_actionable']}")
    print(f"    Non-actionable: {report['unresolved_alias_non_actionable']}")
    print(f"  Scope-gap rows: {scope_gap_report['out_of_scope_rows']} ({100*scope_gap_report['out_of_scope_rate']:.1f}%)")
    if scope_gap_report["suggested_competition_priority"]:
        print(f"  Suggested expansion priorities (resolved leagues):")
        for item in scope_gap_report["suggested_competition_priority"][:10]:
            pct = item["share_of_out_of_scope"] * 100
            print(f"    {item['competition_name']}: {item['rows']} rows ({pct:.1f}%)")
    if report["top_out_of_scope_leagues"]:
        print(f"\nTop out-of-scope leagues:")
        for league, count in report["top_out_of_scope_leagues"].items():
            print(f"  {league}: {count}")
    print(f"\nBy play type:")
    for pt, stats in report["by_play_type"].items():
        pct = 100 * stats["matched"] / max(1, stats["total"])
        print(f"  {pt}: {stats['matched']}/{stats['total']} ({pct:.1f}%)")

    if not args.dry_run:
        write_jsonl(args.output, matched)
        print(f"\nWrote {len(matched)} enriched records to {args.output}")

        write_json(args.report, report)
        print(f"Wrote report to {args.report}")

        write_json(args.suggestions_output, unresolved_alias_suggestions)
        print(f"Wrote unresolved alias suggestions to {args.suggestions_output}")

        write_json(args.scope_gap_output, scope_gap_report)
        print(f"Wrote scope gap report to {args.scope_gap_output}")


if __name__ == "__main__":
    main()
