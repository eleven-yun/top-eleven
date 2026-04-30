import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


DIVISION_BASELINE_PPG = {
    1: 1.35,
    2: 1.20,
}

COMPETITION_TO_LEAGUE_CODE = {
    "Premier League": "E0",
    "Championship": "E1",
}


def load_json(file_path: Path):
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ensure_directory(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def write_json(file_path: Path, payload):
    ensure_directory(file_path.parent)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def write_jsonl(file_path: Path, records):
    ensure_directory(file_path.parent)
    with file_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def safe_int(value, default=0):
    if pd.isna(value):
        return default
    return int(value)


def safe_float(value, default=None):
    if pd.isna(value):
        return default
    return float(value)


def season_sort_key(season_name):
    start_year = int(season_name.split("/")[0])
    return start_year


def season_name_to_folder(season_name):
    start_year, end_year = season_name.split("/")
    if len(start_year) != 4 or len(end_year) != 4:
        raise ValueError(f"Unexpected season format: {season_name}")

    # football-data.co.uk uses YYZZ for modern seasons (e.g., 2022/2023 -> 2223)
    return f"{start_year[2:]}{end_year[2:]}"


def previous_season_name(season_name):
    start_year, end_year = season_name.split("/")
    return f"{int(start_year) - 1}/{int(end_year) - 1}"


def parse_match_datetime(date_value, time_value):
    date_ts = pd.to_datetime(date_value, dayfirst=True, errors="coerce")
    if pd.isna(date_ts):
        return None

    if pd.isna(time_value):
        return date_ts.strftime("%Y-%m-%dT00:00:00")

    time_string = str(time_value).strip()
    if not time_string:
        return date_ts.strftime("%Y-%m-%dT00:00:00")

    dt = pd.to_datetime(
        f"{date_ts.strftime('%Y-%m-%d')} {time_string}",
        errors="coerce",
    )
    if pd.isna(dt):
        return date_ts.strftime("%Y-%m-%dT00:00:00")
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def fetch_csv_with_retry(url, retry_count, retry_sleep_seconds):
    last_error = None
    for attempt in range(1, retry_count + 1):
        try:
            return pd.read_csv(url)
        except Exception as error:
            last_error = error
            if attempt == retry_count:
                break
            time.sleep(retry_sleep_seconds)
    raise RuntimeError(f"CSV fetch failed after {retry_count} attempts: {last_error}") from last_error


def initialise_team_state():
    return {
        "points": 0,
        "goals_for": 0,
        "goals_against": 0,
        "matches_played": 0,
        "points_last_5": deque(maxlen=5),
        "goals_for_last_5": deque(maxlen=5),
        "goals_against_last_5": deque(maxlen=5),
    }


def update_team_state(team_state, goals_for, goals_against):
    if goals_for > goals_against:
        points = 3
    elif goals_for == goals_against:
        points = 1
    else:
        points = 0

    team_state["points"] += points
    team_state["goals_for"] += goals_for
    team_state["goals_against"] += goals_against
    team_state["matches_played"] += 1
    team_state["points_last_5"].append(points)
    team_state["goals_for_last_5"].append(goals_for)
    team_state["goals_against_last_5"].append(goals_against)


def compute_table_positions(season_states):
    ranked = sorted(
        season_states.items(),
        key=lambda item: (
            -item[1]["points"],
            -(item[1]["goals_for"] - item[1]["goals_against"]),
            -item[1]["goals_for"],
            item[0],
        ),
    )
    return {team_id: index + 1 for index, (team_id, _) in enumerate(ranked)}


def result_from_score(home_goals, away_goals):
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def team_feature_snapshot(
    team_id,
    team_name,
    season_name,
    division_level,
    season_states,
    previous_summaries,
    head_to_head_history,
    opponent_team_id,
):
    state = season_states[team_id]
    positions = compute_table_positions(season_states)
    previous_summary = previous_summaries.get((team_id, previous_season_name(season_name)))

    promoted_this_season = 0
    relegated_last_season = 0
    division_level_last_season = None
    seasons_in_current_division = 1

    if previous_summary is not None:
        division_level_last_season = previous_summary["division_level"]
        promoted_this_season = int(previous_summary["division_level"] > division_level)
        relegated_last_season = int(previous_summary["division_level"] < division_level)
        if previous_summary["division_level"] == division_level:
            seasons_in_current_division = previous_summary["seasons_in_current_division"] + 1

    baseline_ppg = DIVISION_BASELINE_PPG.get(division_level, 1.0)
    if previous_summary is None:
        team_strength_prior = baseline_ppg
    else:
        mix = 0.4 if previous_summary["division_level"] != division_level else 0.7
        team_strength_prior = mix * previous_summary["points_per_game"] + (1 - mix) * baseline_ppg

    h2h_key = tuple(sorted([team_id, opponent_team_id]))
    h2h_results = head_to_head_history[h2h_key]
    h2h_wins_last_5 = sum(1 for winner_team_id in h2h_results if winner_team_id == team_id)

    return {
        "team_id": str(team_id),
        "team_name": team_name,
        "league_position": positions.get(team_id),
        "points_last_5": int(sum(state["points_last_5"])),
        "goals_scored_last_5": float(sum(state["goals_for_last_5"])),
        "goals_conceded_last_5": float(sum(state["goals_against_last_5"])),
        "elo_rating": float(team_strength_prior * 1000.0),
        "lineup": [],
        "formation": None,
        "h2h_wins_last_5": h2h_wins_last_5,
        "promoted_this_season": promoted_this_season,
        "relegated_last_season": relegated_last_season,
        "seasons_in_current_division": seasons_in_current_division,
        "division_level_current": division_level,
        "division_level_last_season": division_level_last_season,
        "team_strength_prior": round(team_strength_prior, 4),
        "strength_gap_vs_division_avg": round(team_strength_prior - baseline_ppg, 4),
    }


def expand_competition_specs(specs):
    rows = []
    for spec in specs:
        competition_name = spec["competition_name"]
        league_code = spec.get("league_code") or COMPETITION_TO_LEAGUE_CODE.get(competition_name)
        if league_code is None:
            raise ValueError(
                f"Missing league_code for competition '{competition_name}'. "
                "Add league_code in config/requested_competitions."
            )

        for season_name in spec["season_names"]:
            rows.append(
                {
                    "country_name": spec.get("country_name", "England"),
                    "competition_name": competition_name,
                    "season_name": season_name,
                    "division_level": int(spec["division_level"]),
                    "league_code": league_code,
                }
            )
    return rows


def load_competitions_with_missing(rows, base_url, retry_count, retry_sleep_seconds):
    loaded = []
    missing = []

    for row in rows:
        season_folder = season_name_to_folder(row["season_name"])
        csv_url = f"{base_url}/{season_folder}/{row['league_code']}.csv"
        try:
            matches_df = fetch_csv_with_retry(
                csv_url,
                retry_count=retry_count,
                retry_sleep_seconds=retry_sleep_seconds,
            )
        except Exception:
            missing.append(
                {
                    "country_name": row["country_name"],
                    "competition_name": row["competition_name"],
                    "season_name": row["season_name"],
                }
            )
            continue

        row_with_data = dict(row)
        row_with_data["season_folder"] = season_folder
        row_with_data["csv_url"] = csv_url
        row_with_data["matches_df"] = matches_df
        loaded.append(row_with_data)

    return loaded, missing


def build_dataset(config):
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / config["paths"]["raw_dir"]
    processed_dir = project_root / config["paths"]["processed_dir"]
    ensure_directory(raw_dir)
    ensure_directory(processed_dir)

    retry_count = config["source"].get("retry_count", 3)
    retry_sleep_seconds = config["source"].get("retry_sleep_seconds", 2)
    base_url = config["source"].get("base_url", "https://www.football-data.co.uk/mmz4281")

    requested_rows = expand_competition_specs(config["requested_competitions"])
    loaded_requested, missing_requested = load_competitions_with_missing(
        requested_rows,
        base_url=base_url,
        retry_count=retry_count,
        retry_sleep_seconds=retry_sleep_seconds,
    )

    selected_rows = loaded_requested
    used_fallback = False

    should_try_fallback = (
        config["build_options"].get("use_validation_fallback_if_missing", False)
        and (missing_requested or not selected_rows)
    )

    if should_try_fallback:
        fallback_rows = expand_competition_specs(config["validation_fallback_competitions"])
        loaded_fallback, _ = load_competitions_with_missing(
            fallback_rows,
            base_url=base_url,
            retry_count=retry_count,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        if loaded_fallback:
            selected_rows = loaded_fallback
            used_fallback = True

    if not selected_rows:
        raise ValueError(
            "No competition-season pairs were resolved from football-data.co.uk. "
            f"Missing requested rows: {missing_requested}"
        )

    selected_rows = sorted(
        selected_rows,
        key=lambda row: (row["competition_name"], season_sort_key(row["season_name"])),
    )

    match_meta_records = []
    prematch_feature_records = []
    lottery_market_records = []

    build_summary = {
        "missing_requested": missing_requested,
        "used_fallback": used_fallback,
        "selected_competitions": [],
        "event_fetch_failures": [],
    }

    previous_summaries = {}
    league_states = defaultdict(lambda: defaultdict(initialise_team_state))
    head_to_head_history = defaultdict(lambda: deque(maxlen=5))
    team_name_to_id = {}
    next_team_id = 1

    for competition_row in selected_rows:
        country_name = competition_row.get("country_name", "Unknown")
        competition_name = competition_row["competition_name"]
        season_name = competition_row["season_name"]
        division_level = int(competition_row["division_level"])
        league_code = competition_row["league_code"]
        season_folder = competition_row["season_folder"]

        matches_df = competition_row["matches_df"].copy()
        if config["build_options"].get("write_raw_payloads", True):
            write_json(
                raw_dir / f"matches_{league_code}_{season_folder}.json",
                matches_df.to_dict(orient="records"),
            )

        required_columns = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}
        missing_columns = sorted(col for col in required_columns if col not in matches_df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {league_code} {season_name}: {missing_columns}"
            )

        matches_df = matches_df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
        matches_df["datetime_utc"] = matches_df.apply(
            lambda row: parse_match_datetime(row.get("Date"), row.get("Time")),
            axis=1,
        )
        matches_df = matches_df.dropna(subset=["datetime_utc"]).copy()
        matches_df = matches_df.sort_values(["datetime_utc", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

        build_summary["selected_competitions"].append(
            {
                "competition_id": league_code,
                "season_id": season_folder,
                "competition_name": competition_name,
                "season_name": season_name,
                "match_count": int(len(matches_df)),
            }
        )

        season_key = (competition_name, season_name)
        season_states = league_states[season_key]

        for idx, match_row in matches_df.iterrows():
            home_team = str(match_row["HomeTeam"])
            away_team = str(match_row["AwayTeam"])

            if home_team not in team_name_to_id:
                team_name_to_id[home_team] = next_team_id
                next_team_id += 1
            if away_team not in team_name_to_id:
                team_name_to_id[away_team] = next_team_id
                next_team_id += 1

            home_team_id = team_name_to_id[home_team]
            away_team_id = team_name_to_id[away_team]

            home_score = safe_int(match_row.get("FTHG"), default=0)
            away_score = safe_int(match_row.get("FTAG"), default=0)
            halftime_home_goals = safe_int(match_row.get("HTHG"), default=0)
            halftime_away_goals = safe_int(match_row.get("HTAG"), default=0)

            home_features = team_feature_snapshot(
                team_id=home_team_id,
                team_name=home_team,
                season_name=season_name,
                division_level=division_level,
                season_states=season_states,
                previous_summaries=previous_summaries,
                head_to_head_history=head_to_head_history,
                opponent_team_id=away_team_id,
            )
            away_features = team_feature_snapshot(
                team_id=away_team_id,
                team_name=away_team,
                season_name=season_name,
                division_level=division_level,
                season_states=season_states,
                previous_summaries=previous_summaries,
                head_to_head_history=head_to_head_history,
                opponent_team_id=home_team_id,
            )

            match_id = f"{season_folder}-{league_code}-{idx + 1}"
            match_meta_records.append(
                {
                    "match_id": match_id,
                    "country_name": country_name,
                    "league_code": league_code,
                    "season": season_name,
                    "league": competition_name,
                    "datetime_utc": match_row["datetime_utc"],
                    "venue": "unknown",
                    "home_team_id": str(home_team_id),
                    "away_team_id": str(away_team_id),
                    "referee_id": str(match_row.get("Referee", "unknown")),
                    "final_result": result_from_score(home_score, away_score),
                    "home_goals": home_score,
                    "away_goals": away_score,
                    "halftime_home_goals": halftime_home_goals,
                    "halftime_away_goals": halftime_away_goals,
                }
            )

            prematch_feature_records.append(
                {
                    "match_id": match_id,
                    "home": home_features,
                    "away": away_features,
                }
            )

            if config["build_options"].get("include_placeholder_markets", True):
                handicap_line = safe_float(
                    match_row.get("AHh"),
                    default=config["build_options"].get("default_handicap_line", 0),
                )
                issue_id = f"fd-{league_code}-{season_folder}-{idx + 1}"
                lottery_market_records.extend(
                    [
                        {
                            "match_id": match_id,
                            "issue_id": issue_id,
                            "play_type": "fulltime_1x2",
                            "handicap_line": None,
                            "home_odds": safe_float(match_row.get("AvgH")),
                            "draw_odds": safe_float(match_row.get("AvgD")),
                            "away_odds": safe_float(match_row.get("AvgA")),
                        },
                        {
                            "match_id": match_id,
                            "issue_id": issue_id,
                            "play_type": "htft_1x2",
                            "handicap_line": None,
                            "home_odds": None,
                            "draw_odds": None,
                            "away_odds": None,
                        },
                        {
                            "match_id": match_id,
                            "issue_id": issue_id,
                            "play_type": "handicap_1x2",
                            "handicap_line": handicap_line,
                            "home_odds": safe_float(match_row.get("AvgAHH")),
                            "draw_odds": None,
                            "away_odds": safe_float(match_row.get("AvgAHA")),
                        },
                    ]
                )

            update_team_state(season_states[home_team_id], home_score, away_score)
            update_team_state(season_states[away_team_id], away_score, home_score)

            winner_team_id = None
            if home_score > away_score:
                winner_team_id = home_team_id
            elif away_score > home_score:
                winner_team_id = away_team_id
            head_to_head_history[tuple(sorted([home_team_id, away_team_id]))].append(winner_team_id)

        for team_id, state in season_states.items():
            matches_played = max(state["matches_played"], 1)
            prev_summary = previous_summaries.get((team_id, previous_season_name(season_name)))
            seasons_in_current_division = 1
            if prev_summary is not None and prev_summary.get("division_level") == division_level:
                seasons_in_current_division = prev_summary.get("seasons_in_current_division", 0) + 1

            previous_summaries[(team_id, season_name)] = {
                "division_level": division_level,
                "points_per_game": state["points"] / matches_played,
                "seasons_in_current_division": seasons_in_current_division,
            }

    write_jsonl(processed_dir / "match_meta.jsonl", match_meta_records)
    write_jsonl(processed_dir / "prematch_features.jsonl", prematch_feature_records)
    write_jsonl(processed_dir / "lottery_market.jsonl", lottery_market_records)
    write_json(processed_dir / "build_summary.json", build_summary)

    summary_payload = {
        "match_meta_records": len(match_meta_records),
        "prematch_feature_records": len(prematch_feature_records),
        "lottery_market_records": len(lottery_market_records),
        "used_fallback": used_fallback,
        "missing_requested": missing_requested,
    }
    print(json.dumps(summary_payload, indent=2))
    return summary_payload


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build processed football lottery data from football-data.co.uk CSV datasets."
    )
    parser.add_argument(
        "--config",
        default="config/data_config.json",
        help="Path to the dataset build config JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    config = load_json(project_root / args.config)
    build_dataset(config)


if __name__ == "__main__":
    main()
