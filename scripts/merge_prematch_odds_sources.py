#!/usr/bin/env python3
"""Merge multiple canonical prematch odds JSONL files into one deduplicated file.

All inputs are expected to already follow the canonical schema used by
scripts/enrich_prematch_odds.py (same shape as fetch_cn_prematch_odds output).
"""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def norm(v) -> str:
    return str(v or "").strip().lower()


def record_key(r: dict) -> tuple[str, ...]:
    return (
        norm(r.get("play_type")),
        norm(r.get("date_local")),
        norm(r.get("kickoff_local")),
        norm(r.get("league_name_raw")),
        norm(r.get("home_team_raw")),
        norm(r.get("away_team_raw")),
        norm(r.get("handicap_line_raw")),
    )


def completeness_score(r: dict) -> int:
    score = 0
    for k in ("opening_win", "opening_draw", "opening_lost", "closing_win", "closing_draw", "closing_lost"):
        if r.get(k) is not None:
            score += 1
    return score


def is_better(new: dict, old: dict, source_rank: dict[str, int]) -> bool:
    new_rank = source_rank.get(norm(new.get("source_site")), 10_000)
    old_rank = source_rank.get(norm(old.get("source_site")), 10_000)
    if new_rank != old_rank:
        return new_rank < old_rank

    new_close = norm(new.get("closing_time"))
    old_close = norm(old.get("closing_time"))
    if new_close != old_close:
        return new_close > old_close

    return completeness_score(new) > completeness_score(old)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge canonical prematch odds JSONL files")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input JSONL path (repeat --input for multiple files)",
    )
    parser.add_argument(
        "--output",
        default="data/raw/china_lottery/prematch_odds_raw_merged.jsonl",
        help="Output merged JSONL path",
    )
    parser.add_argument(
        "--prefer-source",
        default="500.com,source_b",
        help="Preferred source order (comma-separated), first wins on tie",
    )
    args = parser.parse_args()

    source_order = [norm(x) for x in args.prefer_source.split(",") if x.strip()]
    source_rank = {s: i for i, s in enumerate(source_order)}

    all_rows = []
    for raw in args.input:
        p = Path(raw)
        if not p.exists():
            print(f"WARN missing input, skip: {p}")
            continue
        rows = load_jsonl(p)
        print(f"Loaded {len(rows)} rows from {p}")
        all_rows.extend(rows)

    dedup = {}
    for row in all_rows:
        k = record_key(row)
        if k not in dedup or is_better(row, dedup[k], source_rank):
            dedup[k] = row

    merged = list(dedup.values())
    merged.sort(
        key=lambda r: (
            norm(r.get("date_local")),
            norm(r.get("kickoff_local")),
            norm(r.get("play_type")),
            norm(r.get("home_team_raw")),
            norm(r.get("away_team_raw")),
        )
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Merged {len(all_rows)} -> {len(merged)} rows into {out_path}")


if __name__ == "__main__":
    main()
