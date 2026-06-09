#!/usr/bin/env python3
"""Convert Source-B pre-match odds export into canonical JSONL schema.

Output schema is compatible with scripts/enrich_prematch_odds.py input
(same shape as fetch_cn_prematch_odds.py output).

Examples:
  python scripts/convert_source_b_prematch_odds.py \
      --input data/raw/source_b/prematch.csv \
      --input-format csv \
      --id-col match_id \
      --kickoff-col kickoff_utc \
      --league-col league \
      --home-col home_team \
      --away-col away_team \
      --play-type-col market \
      --opening-win-col open_home --opening-draw-col open_draw --opening-lost-col open_away \
      --closing-win-col close_home --closing-draw-col close_draw --closing-lost-col close_away
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path


PLAY_TYPE_MAP = {
    "fulltime_1x2": "fulltime_1x2",
    "nspf": "fulltime_1x2",
    "胜平负": "fulltime_1x2",
    "handicap_1x2": "handicap_1x2",
    "spf": "handicap_1x2",
    "让球胜平负": "handicap_1x2",
}


def load_rows(path: Path, input_format: str) -> list[dict]:
    if input_format == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON input must be a list of objects")
        return payload

    if input_format == "jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if input_format == "csv":
        with path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    raise ValueError(f"Unsupported input format: {input_format}")


def to_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def infer_play_type(raw: str, default_mode: str) -> str:
    if default_mode == "fulltime_1x2":
        return "fulltime_1x2"
    if default_mode == "handicap_1x2":
        return "handicap_1x2"
    key = (raw or "").strip().lower()
    return PLAY_TYPE_MAP.get(key, "fulltime_1x2")


def build_zid(row: dict, id_col: str, play_type: str, kickoff_local: str, home: str, away: str) -> str:
    direct = (row.get(id_col) or "").strip() if id_col else ""
    if direct:
        return direct
    seed = f"{play_type}|{kickoff_local}|{home}|{away}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Source-B prematch odds to canonical JSONL")
    parser.add_argument("--input", required=True, help="Source-B input file (csv/json/jsonl)")
    parser.add_argument("--input-format", choices=["csv", "json", "jsonl"], default="csv")
    parser.add_argument(
        "--output",
        default="data/raw/china_lottery/prematch_odds_source_b_raw.jsonl",
        help="Canonical output JSONL path",
    )
    parser.add_argument("--source-site", default="source_b")

    parser.add_argument("--id-col", default="")
    parser.add_argument("--date-col", default="date_local")
    parser.add_argument("--kickoff-col", default="kickoff_local")
    parser.add_argument("--league-col", default="league_name_raw")
    parser.add_argument("--home-col", default="home_team_raw")
    parser.add_argument("--away-col", default="away_team_raw")
    parser.add_argument("--play-type-col", default="play_type")
    parser.add_argument("--handicap-col", default="handicap_line_raw")

    parser.add_argument("--opening-win-col", default="opening_win")
    parser.add_argument("--opening-draw-col", default="opening_draw")
    parser.add_argument("--opening-lost-col", default="opening_lost")
    parser.add_argument("--opening-time-col", default="opening_time")

    parser.add_argument("--closing-win-col", default="closing_win")
    parser.add_argument("--closing-draw-col", default="closing_draw")
    parser.add_argument("--closing-lost-col", default="closing_lost")
    parser.add_argument("--closing-time-col", default="closing_time")

    parser.add_argument(
        "--default-play-type",
        choices=["auto", "fulltime_1x2", "handicap_1x2"],
        default="auto",
        help="Fallback play type if source play_type cannot be mapped",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    rows = load_rows(in_path, args.input_format)
    out_rows = []

    for row in rows:
        kickoff_local = (row.get(args.kickoff_col) or "").strip()
        date_local = (row.get(args.date_col) or "").strip()
        if not date_local and kickoff_local:
            date_local = kickoff_local[:10]

        raw_play_type = (row.get(args.play_type_col) or "").strip()
        play_type = infer_play_type(raw_play_type, args.default_play_type)

        home = (row.get(args.home_col) or "").strip()
        away = (row.get(args.away_col) or "").strip()
        if not home or not away:
            continue

        zid = build_zid(row, args.id_col, play_type, kickoff_local, home, away)

        out_rows.append(
            {
                "source_site": args.source_site,
                "zid": zid,
                "issue_no": zid,
                "play_type_raw": raw_play_type or play_type,
                "play_type": play_type,
                "date_local": date_local,
                "league_name_raw": (row.get(args.league_col) or "").strip(),
                "kickoff_local": kickoff_local,
                "home_team_raw": home,
                "away_team_raw": away,
                "handicap_line_raw": (row.get(args.handicap_col) or "").strip(),
                "opening_win": to_float(row.get(args.opening_win_col)),
                "opening_draw": to_float(row.get(args.opening_draw_col)),
                "opening_lost": to_float(row.get(args.opening_lost_col)),
                "opening_time": (row.get(args.opening_time_col) or "").strip(),
                "closing_win": to_float(row.get(args.closing_win_col)),
                "closing_draw": to_float(row.get(args.closing_draw_col)),
                "closing_lost": to_float(row.get(args.closing_lost_col)),
                "closing_time": (row.get(args.closing_time_col) or "").strip(),
                "odds_history": [],
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in out_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Converted {len(out_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
