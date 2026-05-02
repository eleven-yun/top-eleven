"""Enrich lottery market data with external odds sources.

Reads raw odds rows (from China Lottery or any external source) and joins
them to processed match_meta.jsonl by fuzzy matching on team names, kickoff
time, and league.  Writes a clean lottery_market_cn.jsonl compatible with
the existing data loader and scripts/backtest_ev.py.

Input files:
    data/raw/china_lottery/odds_raw.jsonl   — one record per raw odds row
    data/processed/match_meta.jsonl         — canonical match records
    config/team_alias_cn.json               — team name alias map

Output files:
    data/processed/lottery_market_cn.jsonl  — enriched market odds, loader-compatible
    data/processed/odds_match_report.json   — match coverage and QA report

Raw odds record expected schema (all fields optional unless marked required):
    source_site        : str   — e.g. "500.com", "zucai.com", "manual"
    source_match_id    : str   — source's own match identifier
    issue_no           : str   — lottery issue number (竞彩期号) if available
    play_type_raw      : str   — "fulltime_1x2" | "htft_1x2" | "handicap_1x2"
    league_name_raw    : str   — e.g. "英超", "Premier League"
    kickoff_local      : str   — ISO-8601 local time, e.g. "2024-09-15T15:00:00"
    home_team_raw      : str * required
    away_team_raw      : str * required
    handicap_line_raw  : float — positive = home gives handicap, negative = home receives
    odds_home          : float
    odds_draw          : float
    odds_away          : float
    odds_capture_time  : str   — ISO-8601 when odds snapshot was taken
    odds_stage         : str   — "open" | "close" | "live"

Usage:
    python scripts/enrich_lottery_odds.py
    python scripts/enrich_lottery_odds.py --odds-file data/raw/my_odds.jsonl
    python scripts/enrich_lottery_odds.py --dry-run
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json_or_jsonl(path):
    if not os.path.exists(path):
        return []
    if path.endswith(".jsonl"):
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, list) else [payload]


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def safe_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Team name normalization and alias lookup
# ---------------------------------------------------------------------------

def build_alias_index(alias_config):
    """Build a flat dict: any_alias_lower → canonical_name.

    Parameters
    ----------
    alias_config : dict
        Loaded config/team_alias_cn.json

    Returns
    -------
    dict[str, str]
    """
    index = {}
    for canonical, aliases in alias_config.get("teams", {}).items():
        index[canonical.lower()] = canonical
        for alias in aliases:
            index[alias.lower()] = canonical
    return index


def normalize_team_name(raw_name, alias_index):
    """Return canonical team name or the normalized raw name if not in index.

    Parameters
    ----------
    raw_name : str
    alias_index : dict[str, str]

    Returns
    -------
    str
    """
    if not raw_name:
        return ""
    key = raw_name.strip().lower()
    return alias_index.get(key, raw_name.strip())


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

def parse_iso_dt(dt_str):
    """Parse an ISO-8601-like datetime string to a timezone-naive datetime.

    This project treats all timestamps as **UTC** for proximity scoring:
    - match_meta.jsonl stores kickoff as ``datetime_utc`` (UTC).
    - raw odds rows store kickoff as ``kickoff_local``; callers are expected
      to supply UTC-equivalent values in that field (or accept the resulting
      scoring imprecision if local and UTC times differ).
    Both fields are parsed without timezone conversion so they can be compared
    directly.  If the source odds data is truly local time, convert to UTC
    before passing to this pipeline.

    Parameters
    ----------
    dt_str : str | None

    Returns
    -------
    datetime | None
    """
    if not dt_str:
        return None
    # Each tuple is (format_string, expected_length_of_parsed_string).
    # We slice dt_str to the expected length before parsing so that extra
    # characters (e.g. timezone offset, milliseconds) are silently ignored.
    for fmt, expected_len in (
        ("%Y-%m-%dT%H:%M:%S", 19),
        ("%Y-%m-%dT%H:%M",    16),
        ("%Y-%m-%d %H:%M:%S", 19),
        ("%Y-%m-%d %H:%M",    16),
        ("%Y-%m-%d",          10),
    ):
        try:
            return datetime.strptime(dt_str[:expected_len], fmt)
        except (ValueError, TypeError):
            continue
    return None


def kickoff_distance_hours(dt1, dt2):
    """Return absolute hours between two datetime objects.

    Parameters
    ----------
    dt1, dt2 : datetime | None

    Returns
    -------
    float | None
    """
    if dt1 is None or dt2 is None:
        return None
    delta = abs((dt1 - dt2).total_seconds()) / 3600.0
    return delta


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

PLAY_TYPE_CANONICAL = {
    "fulltime_1x2": "fulltime_1x2",
    "胜平负": "fulltime_1x2",
    "1x2": "fulltime_1x2",
    "htft_1x2": "htft_1x2",
    "半全场": "htft_1x2",
    "htft": "htft_1x2",
    "handicap_1x2": "handicap_1x2",
    "让球胜平负": "handicap_1x2",
    "ah": "handicap_1x2",
    "asian_handicap": "handicap_1x2",
}


def normalize_play_type(raw):
    if not raw:
        return None
    return PLAY_TYPE_CANONICAL.get(raw.lower().strip())


def team_match_score(name_a, name_b):
    """Fuzzy score 0.0–1.0 for two team name strings.

    - Exact match after strip+lower : 1.0
    - One contains the other        : 0.7
    - No overlap                    : 0.0

    Parameters
    ----------
    name_a, name_b : str

    Returns
    -------
    float
    """
    a = name_a.strip().lower()
    b = name_b.strip().lower()
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.7
    # token overlap
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    overlap = tokens_a & tokens_b
    if not overlap:
        return 0.0
    jaccard = len(overlap) / len(tokens_a | tokens_b)
    return round(jaccard * 0.6, 4)


def score_candidate(odds_row, meta_row, alias_index, kickoff_dt_odds):
    """Compute a composite match confidence score [0.0, 1.0] between an odds row
    and a match_meta record.

    Weights:
        Team name match        : 0.45  (home + away each 0.225)
        Kickoff proximity      : 0.30
        League consistency     : 0.20
        Season consistency     : 0.05

    Parameters
    ----------
    odds_row : dict
    meta_row : dict
    alias_index : dict[str, str]
    kickoff_dt_odds : datetime | None

    Returns
    -------
    float  total score in [0.0, 1.0]
    """
    score = 0.0

    # --- team name component ---
    home_raw = normalize_team_name(odds_row.get("home_team_raw", ""), alias_index)
    away_raw = normalize_team_name(odds_row.get("away_team_raw", ""), alias_index)

    meta_home = meta_row.get("home_team_name", "") or meta_row.get("home_team_id", "")
    meta_away = meta_row.get("away_team_name", "") or meta_row.get("away_team_id", "")

    home_score = team_match_score(home_raw, meta_home)
    away_score = team_match_score(away_raw, meta_away)
    score += (home_score + away_score) * 0.225

    # --- kickoff proximity ---
    meta_dt = parse_iso_dt(meta_row.get("datetime_utc"))
    dist_h = kickoff_distance_hours(kickoff_dt_odds, meta_dt)
    if dist_h is not None:
        if dist_h <= 0.5:
            score += 0.30
        elif dist_h <= 2.0:
            score += 0.20
        elif dist_h <= 12.0:
            score += 0.10
        # else: no contribution

    # --- league consistency ---
    league_raw = (odds_row.get("league_name_raw") or "").lower()
    meta_league = (meta_row.get("league") or "").lower()
    meta_country = (meta_row.get("country_name") or "").lower()
    meta_league_code = (meta_row.get("league_code") or "").lower()

    if league_raw:
        for candidate in (meta_league, meta_country, meta_league_code):
            if candidate and (league_raw in candidate or candidate in league_raw):
                score += 0.20
                break

    # --- season consistency (derive season year from kickoff) ---
    # Only award the bonus when at least one stronger signal already fired,
    # otherwise almost every meta_row gets a non-zero score (year diff <= 1
    # is nearly always true across 23k rows).
    if score > 0.0 and kickoff_dt_odds is not None and meta_dt is not None:
        year_diff = abs(kickoff_dt_odds.year - meta_dt.year)
        if year_diff <= 1:
            score += 0.05

    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# Matching engine
# ---------------------------------------------------------------------------

def _build_meta_date_index(meta_rows):
    """Pre-index meta_rows by kickoff date for O(1) candidate narrowing.

    Returns
    -------
    tuple(dict[str, list[dict]], list[dict])
        date_index keyed by "YYYY-MM-DD" and a fallback list for rows with no
        parseable datetime.
    """
    date_index = {}
    no_date = []
    for meta_row in meta_rows:
        dt = parse_iso_dt(meta_row.get("datetime_utc"))
        if dt is None:
            no_date.append(meta_row)
        else:
            key = dt.strftime("%Y-%m-%d")
            date_index.setdefault(key, []).append(meta_row)
    return date_index, no_date


def _get_meta_candidates(kickoff_dt, date_index, no_date, window_days=1):
    """Return meta_rows whose kickoff falls within window_days of kickoff_dt.

    When kickoff_dt is None every meta_row is returned as a fallback.
    """
    candidates = list(no_date)
    if kickoff_dt is None:
        for rows in date_index.values():
            candidates.extend(rows)
        return candidates
    for delta in range(-window_days, window_days + 1):
        day = (kickoff_dt + timedelta(days=delta)).strftime("%Y-%m-%d")
        candidates.extend(date_index.get(day, []))
    return candidates


def match_odds_to_meta(odds_rows, meta_rows, alias_index, min_score=0.75, min_gap=0.10):
    """Attempt to match each odds row to a match_meta record.

    Parameters
    ----------
    odds_rows : list[dict]
    meta_rows : list[dict]
    alias_index : dict[str, str]
    min_score : float   minimum top score to accept a match
    min_gap : float     minimum gap between top-1 and top-2 scores

    Returns
    -------
    tuple(list[dict], list[dict])
        matched_records, unresolved_records
    """
    matched = []
    unresolved = []
    date_index, no_date = _build_meta_date_index(meta_rows)

    for odds_row in odds_rows:
        kickoff_dt = parse_iso_dt(odds_row.get("kickoff_local"))
        play_type = normalize_play_type(odds_row.get("play_type_raw"))
        if play_type is None:
            unresolved.append(
                {
                    "source_match_id": odds_row.get("source_match_id"),
                    "home_team_raw": odds_row.get("home_team_raw"),
                    "away_team_raw": odds_row.get("away_team_raw"),
                    "kickoff_local": odds_row.get("kickoff_local"),
                    "league_name_raw": odds_row.get("league_name_raw"),
                    "top_score": 0.0,
                    "reason": "unknown_play_type",
                }
            )
            continue

        candidates = _get_meta_candidates(kickoff_dt, date_index, no_date)
        num_candidates = len(candidates)
        scored = []
        for meta_row in candidates:
            s = score_candidate(odds_row, meta_row, alias_index, kickoff_dt)
            if s > 0:
                scored.append((s, meta_row))

        scored.sort(key=lambda x: -x[0])

        top_score = scored[0][0] if scored else 0.0
        second_score = scored[1][0] if len(scored) >= 2 else 0.0
        gap = top_score - second_score

        if top_score >= min_score and gap >= min_gap:
            best_meta = scored[0][1]
            match_id = best_meta["match_id"]

            matched.append(
                {
                    "match_id": match_id,
                    "issue_id": odds_row.get("issue_no") or f"ext-{match_id}",
                    "play_type": play_type,
                    "odds_stage": odds_row.get("odds_stage", "close"),
                    "handicap_line": safe_float(odds_row.get("handicap_line_raw")),
                    "home_odds": safe_float(odds_row.get("odds_home")),
                    "draw_odds": safe_float(odds_row.get("odds_draw")),
                    "away_odds": safe_float(odds_row.get("odds_away")),
                    "odds_capture_time": odds_row.get("odds_capture_time"),
                    "source": odds_row.get("source_site", "external"),
                    "match_confidence_score": top_score,
                    "unresolved_reason": None,
                }
            )
        else:
            reason = (
                "no_candidates" if num_candidates == 0
                else "no_scored_candidates" if not scored
                else f"low_score={top_score}" if top_score < min_score
                else f"low_gap={gap}"
            )
            unresolved.append(
                {
                    "source_match_id": odds_row.get("source_match_id"),
                    "home_team_raw": odds_row.get("home_team_raw"),
                    "away_team_raw": odds_row.get("away_team_raw"),
                    "kickoff_local": odds_row.get("kickoff_local"),
                    "league_name_raw": odds_row.get("league_name_raw"),
                    "top_score": top_score,
                    "reason": reason,
                }
            )

    return matched, unresolved


# ---------------------------------------------------------------------------
# Deduplication of matched records
# ---------------------------------------------------------------------------

def deduplicate_market_records(records):
    """Within same (match_id, play_type, odds_stage), keep latest by odds_capture_time.

    Parameters
    ----------
    records : list[dict]

    Returns
    -------
    list[dict]
    """
    def _parse_ts(ts_str):
        """Parse an odds_capture_time string to datetime; return None on failure."""
        if not ts_str:
            return None
        return parse_iso_dt(ts_str)

    index = {}
    for rec in records:
        key = (rec["match_id"], rec.get("play_type"), rec.get("odds_stage"))
        if key not in index:
            index[key] = rec
        else:
            existing_dt = _parse_ts(index[key].get("odds_capture_time"))
            new_dt = _parse_ts(rec.get("odds_capture_time"))
            if existing_dt is None and new_dt is None:
                # Neither parseable — fall back to string comparison
                if (rec.get("odds_capture_time") or "") >= (index[key].get("odds_capture_time") or ""):
                    index[key] = rec
            elif new_dt is not None and (existing_dt is None or new_dt >= existing_dt):
                index[key] = rec
    return list(index.values())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_root():
    """Return the project root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match raw external odds to processed match_meta and write enriched market JSONL."
    )
    parser.add_argument(
        "--odds-file",
        default=None,
        help="Path to raw odds JSONL. Defaults to data/raw/china_lottery/odds_raw.jsonl",
    )
    parser.add_argument(
        "--meta-file",
        default=None,
        help="Path to match_meta JSONL. Defaults to data/processed/match_meta.jsonl",
    )
    parser.add_argument(
        "--alias-file",
        default=None,
        help="Path to team_alias_cn.json. Defaults to config/team_alias_cn.json",
    )
    parser.add_argument(
        "--output-market",
        default=None,
        help="Output path for enriched market JSONL. Defaults to data/processed/lottery_market_cn.jsonl",
    )
    parser.add_argument(
        "--output-report",
        default=None,
        help="Output path for match coverage report JSON. Defaults to data/processed/odds_match_report.json",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.75,
        help="Minimum match confidence score to accept (default: 0.75)",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.10,
        help="Minimum score gap between top-1 and top-2 candidates (default: 0.10)",
    )
    parser.add_argument(
        "--prematch-file",
        default=None,
        help="Path to prematch_features.jsonl for team name lookup. "
             "Defaults to data/processed/prematch_features.jsonl",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show match coverage stats without writing output files.",
    )
    return parser.parse_args()


def main():
    root = resolve_root()
    args = parse_args()

    odds_path      = args.odds_file     or str(root / "data/raw/china_lottery/odds_raw.jsonl")
    meta_path      = args.meta_file     or str(root / "data/processed/match_meta.jsonl")
    alias_path     = args.alias_file    or str(root / "config/team_alias_cn.json")
    prematch_path  = args.prematch_file or str(root / "data/processed/prematch_features.jsonl")
    out_market     = args.output_market or str(root / "data/processed/lottery_market_cn.jsonl")
    out_report     = args.output_report or str(root / "data/processed/odds_match_report.json")

    # --- load ---
    odds_rows = load_json_or_jsonl(odds_path)
    if not odds_rows:
        print(f"No raw odds rows found at {odds_path}. Nothing to do.")
        print("To get started, place a JSONL file with raw odds there (see module docstring for schema).")
        return

    meta_rows = load_json_or_jsonl(meta_path)
    if not meta_rows:
        raise FileNotFoundError(f"match_meta not found at {meta_path}. Run data/build_dataset.py first.")

    alias_config = {}
    if os.path.exists(alias_path):
        with open(alias_path, "r", encoding="utf-8") as f:
            alias_config = json.load(f)
    alias_index = build_alias_index(alias_config)

    # Build team_id → team_name lookup from prematch_features so score_candidate
    # can compare external names against real team names rather than numeric IDs.
    team_id_to_name = {}
    prematch_rows = load_json_or_jsonl(prematch_path)
    for prow in prematch_rows:
        for side in ("home", "away"):
            feat = prow.get(side) or {}
            tid = feat.get("team_id")
            tname = feat.get("team_name")
            if tid and tname and tid not in team_id_to_name:
                team_id_to_name[tid] = tname

    # Enrich meta_rows in-place with team names for fuzzy matching.
    # Use falsy check so empty-string values are also overwritten.
    for meta_row in meta_rows:
        if not meta_row.get("home_team_name"):
            meta_row["home_team_name"] = team_id_to_name.get(meta_row.get("home_team_id"), "")
        if not meta_row.get("away_team_name"):
            meta_row["away_team_name"] = team_id_to_name.get(meta_row.get("away_team_id"), "")

    print(f"Loaded {len(odds_rows)} raw odds rows, {len(meta_rows)} match meta records, "
          f"{len(alias_index)} alias entries, {len(team_id_to_name)} team name mappings")

    # --- match ---
    matched, unresolved = match_odds_to_meta(
        odds_rows=odds_rows,
        meta_rows=meta_rows,
        alias_index=alias_index,
        min_score=args.min_score,
        min_gap=args.min_gap,
    )

    deduplicated = deduplicate_market_records(matched)

    total = len(odds_rows)
    match_rate = round(len(matched) / total, 4) if total else 0.0
    dedup_dropped = len(matched) - len(deduplicated)

    report = {
        "total_odds_rows": total,
        "matched": len(matched),
        "unresolved": len(unresolved),
        "match_rate": match_rate,
        "dedup_dropped": dedup_dropped,
        "final_market_records": len(deduplicated),
        "min_score": args.min_score,
        "min_gap": args.min_gap,
        "unresolved_sample": unresolved[:20],
    }

    # --- summary ---
    print(f"\n--- Match Coverage ---")
    print(f"  Total raw odds rows  : {total}")
    print(f"  Matched              : {len(matched)}  ({match_rate*100:.1f}%)")
    print(f"  Unresolved           : {len(unresolved)}")
    print(f"  Dedup dropped        : {dedup_dropped}")
    print(f"  Final market records : {len(deduplicated)}")

    if args.dry_run:
        print("\nDry-run mode: no files written.")
        if unresolved:
            print("\nSample unresolved rows (first 5):")
            for row in unresolved[:5]:
                print(f"  {row}")
        return

    write_jsonl(out_market, deduplicated)
    write_json(out_report, report)
    print(f"\nWrote enriched market → {out_market}")
    print(f"Wrote match report    → {out_report}")

    if unresolved:
        print(f"\nWarning: {len(unresolved)} unresolved rows. "
              "Add aliases to config/team_alias_cn.json to improve coverage.")


if __name__ == "__main__":
    main()
