from typing import Optional

FULLTIME_LABEL_TO_ID = {
    "home_win": 0,
    "draw": 1,
    "away_win": 2,
}

HTFT_LABEL_TO_ID = {
    "H/H": 0,
    "H/D": 1,
    "H/A": 2,
    "D/H": 3,
    "D/D": 4,
    "D/A": 5,
    "A/H": 6,
    "A/D": 7,
    "A/A": 8,
}


def score_to_result(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if home_goals < away_goals:
        return "away_win"
    return "draw"


def map_fulltime_label(final_result: str) -> int:
    if final_result not in FULLTIME_LABEL_TO_ID:
        raise ValueError(f"Unknown fulltime result: {final_result}")
    return FULLTIME_LABEL_TO_ID[final_result]


def map_htft_label(
    halftime_home_goals: int,
    halftime_away_goals: int,
    fulltime_home_goals: int,
    fulltime_away_goals: int,
) -> int:
    halftime_result = score_to_result(halftime_home_goals, halftime_away_goals)
    fulltime_result = score_to_result(fulltime_home_goals, fulltime_away_goals)

    short = {
        "home_win": "H",
        "draw": "D",
        "away_win": "A",
    }
    htft_code = f"{short[halftime_result]}/{short[fulltime_result]}"

    if htft_code not in HTFT_LABEL_TO_ID:
        raise ValueError(f"Unknown HT/FT label: {htft_code}")
    return HTFT_LABEL_TO_ID[htft_code]


def map_handicap_label(
    fulltime_home_goals: int,
    fulltime_away_goals: int,
    handicap_line: Optional[int],
) -> int:
    if handicap_line is None:
        raise ValueError("handicap_line is required for handicap label mapping")

    adjusted_home = fulltime_home_goals + handicap_line

    if adjusted_home > fulltime_away_goals:
        return FULLTIME_LABEL_TO_ID["home_win"]
    if adjusted_home < fulltime_away_goals:
        return FULLTIME_LABEL_TO_ID["away_win"]
    return FULLTIME_LABEL_TO_ID["draw"]
