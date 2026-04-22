import argparse
import json
from pathlib import Path

from data.label_mapping import HTFT_LABEL_TO_ID, FULLTIME_LABEL_TO_ID, map_fulltime_label, map_handicap_label, map_htft_label


def load_json_or_jsonl(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    if file_path.suffix == ".jsonl":
        records = []
        with file_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    with file_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, list):
        return payload
    return [payload]


def value_at_path(record, path):
    current = record
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def compute_missingness(records, fields):
    total = len(records)
    results = {}
    for field in fields:
        missing = 0
        for record in records:
            value = value_at_path(record, field)
            if value is None or value == "":
                missing += 1

        ratio = (missing / total) if total else 0.0
        results[field] = {
            "missing": missing,
            "total": total,
            "missing_ratio": round(ratio, 6),
        }
    return results


def compute_season_counts(match_meta):
    counts = {}
    for row in match_meta:
        key = (row.get("season"), row.get("league"))
        counts[key] = counts.get(key, 0) + 1

    output = []
    for (season, league), count in sorted(counts.items()):
        output.append({"season": season, "league": league, "match_count": count})
    return output


def invert_dict(dct):
    return {value: key for key, value in dct.items()}


def class_distribution(values, id_to_name):
    total = len(values)
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1

    output = []
    for label_id in sorted(id_to_name.keys()):
        count = counts.get(label_id, 0)
        ratio = (count / total) if total else 0.0
        output.append(
            {
                "label_id": label_id,
                "label_name": id_to_name[label_id],
                "count": count,
                "ratio": round(ratio, 6),
            }
        )
    return output


def compute_class_balance(match_meta, lottery_market):
    meta_by_match = {row["match_id"]: row for row in match_meta}

    handicap_market = {}
    for row in lottery_market:
        if row.get("play_type") == "handicap_1x2":
            handicap_market[row["match_id"]] = row.get("handicap_line")

    fulltime_labels = []
    htft_labels = []
    handicap_labels = []
    handicap_missing = 0

    for match_id, meta in meta_by_match.items():
        fulltime_labels.append(map_fulltime_label(meta["final_result"]))
        htft_labels.append(
            map_htft_label(
                meta["halftime_home_goals"],
                meta["halftime_away_goals"],
                meta["home_goals"],
                meta["away_goals"],
            )
        )

        line = handicap_market.get(match_id)
        if line is None:
            handicap_missing += 1
            continue

        handicap_labels.append(map_handicap_label(meta["home_goals"], meta["away_goals"], line))

    return {
        "fulltime_1x2": class_distribution(fulltime_labels, invert_dict(FULLTIME_LABEL_TO_ID)),
        "htft_1x2": class_distribution(htft_labels, invert_dict(HTFT_LABEL_TO_ID)),
        "handicap_1x2": {
            "distribution": class_distribution(handicap_labels, invert_dict(FULLTIME_LABEL_TO_ID)),
            "missing_handicap_line_matches": handicap_missing,
            "eligible_matches": len(handicap_labels),
            "total_matches": len(meta_by_match),
        },
    }


def season_to_split(season_name, season_split):
    for split_name, seasons in season_split.items():
        if season_name in seasons:
            return split_name
    return None


def compute_split_integrity(match_meta, prematch_features, lottery_market, season_split):
    prematch_match_ids = {row["match_id"] for row in prematch_features}
    market_match_ids = {row["match_id"] for row in lottery_market}
    meta_by_match = {row["match_id"]: row for row in match_meta}

    split_to_match_ids = {split_name: set() for split_name in season_split}
    unknown_split_match_ids = []

    for row in match_meta:
        split_name = season_to_split(row["season"], season_split)
        if split_name is None:
            unknown_split_match_ids.append(row["match_id"])
            continue
        split_to_match_ids[split_name].add(row["match_id"])

    overlap_pairs = []
    split_names = list(split_to_match_ids.keys())
    for index, split_name in enumerate(split_names):
        for other_split in split_names[index + 1:]:
            overlap = split_to_match_ids[split_name] & split_to_match_ids[other_split]
            overlap_pairs.append(
                {
                    "split_a": split_name,
                    "split_b": other_split,
                    "overlap_count": len(overlap),
                }
            )

    split_counts = []
    split_class_balance = {}
    promoted_team_counts = {}
    handicap_market_by_match = {
        row["match_id"]: row.get("handicap_line")
        for row in lottery_market
        if row.get("play_type") == "handicap_1x2"
    }

    for split_name, match_ids in split_to_match_ids.items():
        match_rows = [meta_by_match[match_id] for match_id in match_ids]
        prematch_rows = [row for row in prematch_features if row["match_id"] in match_ids]
        split_counts.append(
            {
                "split": split_name,
                "match_meta_records": len(match_rows),
                "prematch_feature_records": len(prematch_rows),
                "lottery_market_records": sum(1 for row in lottery_market if row["match_id"] in match_ids),
            }
        )

        fulltime_labels = []
        htft_labels = []
        handicap_labels = []
        promoted_match_count = 0

        for row in match_rows:
            fulltime_labels.append(map_fulltime_label(row["final_result"]))
            htft_labels.append(
                map_htft_label(
                    row["halftime_home_goals"],
                    row["halftime_away_goals"],
                    row["home_goals"],
                    row["away_goals"],
                )
            )

            handicap_line = handicap_market_by_match.get(row["match_id"])
            if handicap_line is not None:
                handicap_labels.append(map_handicap_label(row["home_goals"], row["away_goals"], handicap_line))

        for row in prematch_rows:
            home_promoted = int(row.get("home", {}).get("promoted_this_season", 0) or 0)
            away_promoted = int(row.get("away", {}).get("promoted_this_season", 0) or 0)
            if home_promoted or away_promoted:
                promoted_match_count += 1

        split_class_balance[split_name] = {
            "fulltime_1x2": class_distribution(fulltime_labels, invert_dict(FULLTIME_LABEL_TO_ID)),
            "htft_1x2": class_distribution(htft_labels, invert_dict(HTFT_LABEL_TO_ID)),
            "handicap_1x2": class_distribution(handicap_labels, invert_dict(FULLTIME_LABEL_TO_ID)),
        }
        promoted_team_counts[split_name] = {
            "promoted_team_match_count": promoted_match_count,
            "total_matches": len(prematch_rows),
            "ratio": round((promoted_match_count / len(prematch_rows)) if prematch_rows else 0.0, 6),
        }

    all_train_ids = split_to_match_ids.get("train", set())
    all_test_ids = split_to_match_ids.get("test", set())

    return {
        "season_split": season_split,
        "unknown_split_match_count": len(unknown_split_match_ids),
        "unknown_split_match_ids": unknown_split_match_ids[:20],
        "pairwise_overlap": overlap_pairs,
        "train_contains_test_matches": len(all_train_ids & all_test_ids) > 0,
        "counts_by_split": split_counts,
        "class_balance_by_split": split_class_balance,
        "promoted_team_match_counts": promoted_team_counts,
        "coverage_consistency": {
            "match_meta_vs_prematch_missing": len(set(meta_by_match) - prematch_match_ids),
            "match_meta_vs_market_missing": len(set(meta_by_match) - market_match_ids),
        },
    }


def build_report(processed_dir: Path, config: dict):
    match_meta = load_json_or_jsonl(processed_dir / "match_meta.jsonl")
    prematch_features = load_json_or_jsonl(processed_dir / "prematch_features.jsonl")
    lottery_market = load_json_or_jsonl(processed_dir / "lottery_market.jsonl")

    match_fields = [
        "match_id",
        "season",
        "league",
        "datetime_utc",
        "home_team_id",
        "away_team_id",
        "final_result",
        "home_goals",
        "away_goals",
        "halftime_home_goals",
        "halftime_away_goals",
    ]
    prematch_fields = [
        "match_id",
        "home.team_id",
        "home.league_position",
        "home.points_last_5",
        "home.goals_scored_last_5",
        "home.goals_conceded_last_5",
        "home.elo_rating",
        "home.promoted_this_season",
        "home.team_strength_prior",
        "away.team_id",
        "away.league_position",
        "away.points_last_5",
        "away.goals_scored_last_5",
        "away.goals_conceded_last_5",
        "away.elo_rating",
        "away.promoted_this_season",
        "away.team_strength_prior",
    ]

    report = {
        "counts": {
            "match_meta_records": len(match_meta),
            "prematch_feature_records": len(prematch_features),
            "lottery_market_records": len(lottery_market),
        },
        "per_season_counts": compute_season_counts(match_meta),
        "missingness": {
            "match_meta": compute_missingness(match_meta, match_fields),
            "prematch_features": compute_missingness(prematch_features, prematch_fields),
        },
        "class_balance": compute_class_balance(match_meta, lottery_market),
        "split_integrity": compute_split_integrity(
            match_meta,
            prematch_features,
            lottery_market,
            config["season_split"],
        ),
    }

    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Audit processed dataset quality for Phase 1 acceptance checks.")
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Path to processed dataset directory containing JSONL files.",
    )
    parser.add_argument(
        "--config",
        default="config/data_config.json",
        help="Path to data config JSON containing season splits.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/data_quality_report.json",
        help="Path to write JSON report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    with Path(args.config).open("r", encoding="utf-8") as file:
        config = json.load(file)

    report = build_report(processed_dir, config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(json.dumps(report["counts"], indent=2))
    print(json.dumps(report["split_integrity"]["counts_by_split"], indent=2))
    print(f"Wrote audit report to: {output_path}")


if __name__ == "__main__":
    main()