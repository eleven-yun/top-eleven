import argparse
import json
import time
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd
from statsbombpy import sb


DIVISION_BASELINE_PPG = {
    1: 1.35,
    2: 1.20,
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


def safe_value(value, default=None):
    if pd.isna(value):
        return default
    return value


def fetch_with_retry(fetch_fn, retry_count, retry_sleep_seconds, **kwargs):
    last_error = None
    for attempt in range(1, retry_count + 1):
        try:
            return fetch_fn(**kwargs)
        except Exception as error:  # network / SSL issues from raw.githubusercontent are expected here
            last_error = error
            if attempt == retry_count:
                break
            time.sleep(retry_sleep_seconds)
    raise RuntimeError(f"Fetch failed after {retry_count} attempts: {last_error}") from last_error


def competition_lookup_key(country_name, competition_name, season_name):
    return (country_name, competition_name, season_name)


def resolve_competition_rows(competitions_df, competition_specs):
    resolved = []
    missing = []

    for spec in competition_specs:
        for season_name in spec["season_names"]:
            mask = (
                (competitions_df["country_name"] == spec["country_name"])
                & (competitions_df["competition_name"] == spec["competition_name"])
                & (competitions_df["season_name"] == season_name)
            )
            matching = competitions_df.loc[mask]
            if matching.empty:
                missing.append(
                    {
                        "country_name": spec["country_name"],
                        "competition_name": spec["competition_name"],
                        "season_name": season_name,
                    }
                )
                continue

            row = matching.iloc[0].to_dict()
            row["division_level"] = spec["division_level"]
            resolved.append(row)

    return resolved, missing


def build_match_datetime(match_row):
    return f"{match_row['match_date']}T{safe_value(match_row.get('kick_off'), '00:00:00')}"


def result_from_score(home_goals, away_goals):
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def extract_starting_xi(events_df, team_name):
    rows = events_df[(events_df["type"] == "Starting XI") & (events_df["team"] == team_name)]
    if rows.empty:
        return [], None

    tactics = rows.iloc[0].get("tactics") or {}
    lineup = []
    for item in tactics.get("lineup", []):
        player = item.get("player") or {}
        player_id = player.get("id")
        if player_id is not None:
            lineup.append(str(player_id))

    formation = tactics.get("formation")
    return lineup, str(formation) if formation is not None else None


def compute_halftime_goals(events_df, home_team, away_team):
    halftime = {home_team: 0, away_team: 0}

    goal_mask = (events_df["period"] == 1) & (
        (events_df["shot_outcome"] == "Goal")
        | (events_df["type"] == "Own Goal For")
    )

    goal_events = events_df.loc[goal_mask, ["team", "type"]]
    for _, row in goal_events.iterrows():
        team = row.get("team")
        if team in halftime:
            halftime[team] += 1

    return halftime[home_team], halftime[away_team]


def previous_season_name(season_name):
    start_year, end_year = season_name.split("/")
    return f"{int(start_year) - 1}/{int(end_year) - 1}"


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


def team_feature_snapshot(
    team_id,
    team_name,
    season_name,
    division_level,
    season_states,
    previous_summaries,
    head_to_head_history,
    opponent_team_id,
    lineup,
    formation,
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
        "lineup": lineup,
        "formation": formation,
        "h2h_wins_last_5": h2h_wins_last_5,
        "promoted_this_season": promoted_this_season,
        "relegated_last_season": relegated_last_season,
        "seasons_in_current_division": seasons_in_current_division,
        "division_level_current": division_level,
        "division_level_last_season": division_level_last_season,
        "team_strength_prior": round(team_strength_prior, 4),
        "strength_gap_vs_division_avg": round(team_strength_prior - baseline_ppg, 4),
    }


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


def season_sort_key(season_name):
    start_year = int(season_name.split("/")[0])
    return start_year


def build_dataset(config):
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / config["paths"]["raw_dir"]
    processed_dir = project_root / config["paths"]["processed_dir"]
    ensure_directory(raw_dir)
    ensure_directory(processed_dir)

    retry_count = config["source"]["retry_count"]
    retry_sleep_seconds = config["source"]["retry_sleep_seconds"]

    competitions_df = fetch_with_retry(
        sb.competitions,
        retry_count=retry_count,
        retry_sleep_seconds=retry_sleep_seconds,
    )

    requested_rows, missing_requested = resolve_competition_rows(
        competitions_df,
        config["requested_competitions"],
    )

    selected_rows = requested_rows
    used_fallback = False

    should_try_fallback = (
        config["build_options"]["use_validation_fallback_if_missing"]
        and (missing_requested or not selected_rows)
    )

    if should_try_fallback:
        fallback_rows, _ = resolve_competition_rows(
            competitions_df,
            config["validation_fallback_competitions"],
        )
        if fallback_rows:
            selected_rows = fallback_rows
            used_fallback = True

    if not selected_rows:
        raise ValueError(
            "No competition-season pairs were resolved. "
            f"Missing requested rows: {missing_requested}"
        )

    write_json(raw_dir / "competitions.json", competitions_df.to_dict(orient="records"))

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

    selected_rows = sorted(
        selected_rows,
        key=lambda row: (row["competition_name"], season_sort_key(row["season_name"])),
    )

    for competition_row in selected_rows:
        competition_id = int(competition_row["competition_id"])
        season_id = int(competition_row["season_id"])
        season_name = competition_row["season_name"]
        competition_name = competition_row["competition_name"]
        division_level = int(competition_row["division_level"])

        matches_df = fetch_with_retry(
            sb.matches,
            retry_count=retry_count,
            retry_sleep_seconds=retry_sleep_seconds,
            competition_id=competition_id,
            season_id=season_id,
        )

        matches_df = matches_df.sort_values(["match_date", "kick_off", "match_id"])
        build_summary["selected_competitions"].append(
            {
                "competition_id": competition_id,
                "season_id": season_id,
                "competition_name": competition_name,
                "season_name": season_name,
                "match_count": int(len(matches_df)),
            }
        )

        if config["build_options"]["write_raw_payloads"]:
            write_json(
                raw_dir / f"matches_{competition_id}_{season_id}.json",
                matches_df.to_dict(orient="records"),
            )

        season_key = (competition_name, season_name)
        season_states = league_states[season_key]

        for _, match_row in matches_df.iterrows():
            match_id = int(match_row["match_id"])
            events_df = pd.DataFrame()
            fetch_events = config["build_options"].get("fetch_events", True)
            if fetch_events:
                try:
                    events_df = fetch_with_retry(
                        sb.events,
                        retry_count=retry_count,
                        retry_sleep_seconds=retry_sleep_seconds,
                        match_id=match_id,
                    )
                    if config["build_options"]["write_raw_payloads"]:
                        write_json(
                            raw_dir / "events" / f"{match_id}.json",
                            events_df.to_dict(orient="records"),
                        )
                except Exception as error:
                    build_summary["event_fetch_failures"].append(
                        {
                            "match_id": match_id,
                            "error": str(error),
                        }
                    )

            home_team = match_row["home_team"]
            away_team = match_row["away_team"]
            home_team_id = int(match_row["home_team_id"])
            away_team_id = int(match_row["away_team_id"])
            home_score = int(match_row["home_score"])
            away_score = int(match_row["away_score"])
            if events_df.empty:
                halftime_home_goals, halftime_away_goals = 0, 0
                home_lineup, home_formation = [], None
                away_lineup, away_formation = [], None
            else:
                halftime_home_goals, halftime_away_goals = compute_halftime_goals(
                    events_df,
                    home_team,
                    away_team,
                )
                home_lineup, home_formation = extract_starting_xi(events_df, home_team)
                away_lineup, away_formation = extract_starting_xi(events_df, away_team)

            home_features = team_feature_snapshot(
                team_id=home_team_id,
                team_name=home_team,
                season_name=season_name,
                division_level=division_level,
                season_states=season_states,
                previous_summaries=previous_summaries,
                head_to_head_history=head_to_head_history,
                opponent_team_id=away_team_id,
                lineup=home_lineup,
                formation=home_formation,
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
                lineup=away_lineup,
                formation=away_formation,
            )

            match_meta_records.append(
                {
                    "match_id": str(match_id),
                    "season": season_name,
                    "league": competition_name,
                    "datetime_utc": build_match_datetime(match_row),
                    "venue": safe_value(match_row.get("stadium"), "unknown"),
                    "home_team_id": str(home_team_id),
                    "away_team_id": str(away_team_id),
                    "referee_id": str(safe_value(match_row.get("referee_id"), "unknown")),
                    "final_result": result_from_score(home_score, away_score),
                    "home_goals": home_score,
                    "away_goals": away_score,
                    "halftime_home_goals": halftime_home_goals,
                    "halftime_away_goals": halftime_away_goals,
                }
            )
            prematch_feature_records.append(
                {
                    "match_id": str(match_id),
                    "home": home_features,
                    "away": away_features,
                }
            )

            if config["build_options"]["include_placeholder_markets"]:
                issue_id = f"statsbomb-{competition_id}-{season_id}-{match_id}"
                lottery_market_records.extend(
                    [
                        {
                            "match_id": str(match_id),
                            "issue_id": issue_id,
                            "play_type": "fulltime_1x2",
                            "handicap_line": None,
                            "home_odds": None,
                            "draw_odds": None,
                            "away_odds": None,
                        },
                        {
                            "match_id": str(match_id),
                            "issue_id": issue_id,
                            "play_type": "htft_1x2",
                            "handicap_line": None,
                            "home_odds": None,
                            "draw_odds": None,
                            "away_odds": None,
                        },
                        {
                            "match_id": str(match_id),
                            "issue_id": issue_id,
                            "play_type": "handicap_1x2",
                            "handicap_line": config["build_options"]["default_handicap_line"],
                            "home_odds": None,
                            "draw_odds": None,
                            "away_odds": None,
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
            previous_summaries[(team_id, season_name)] = {
                "division_level": division_level,
                "points_per_game": state["points"] / matches_played,
                "seasons_in_current_division": previous_summaries.get((team_id, previous_season_name(season_name)), {}).get(
                    "seasons_in_current_division",
                    0,
                ) + (1 if previous_summaries.get((team_id, previous_season_name(season_name)), {}).get("division_level") == division_level else 1),
            }

    write_jsonl(processed_dir / "match_meta.jsonl", match_meta_records)
    write_jsonl(processed_dir / "prematch_features.jsonl", prematch_feature_records)
    write_jsonl(processed_dir / "lottery_market.jsonl", lottery_market_records)
    write_json(processed_dir / "build_summary.json", build_summary)

    print(
        json.dumps(
            {
                "match_meta_records": len(match_meta_records),
                "prematch_feature_records": len(prematch_feature_records),
                "lottery_market_records": len(lottery_market_records),
                "used_fallback": used_fallback,
                "missing_requested": missing_requested,
            },
            indent=2,
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build processed football lottery data from StatsBomb open data.")
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