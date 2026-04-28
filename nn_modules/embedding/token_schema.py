import torch

# Fixed token order for V1 match representation.
TOKEN_NAMES = [
    "home_league_position",
    "away_league_position",
    "home_points_last_5",
    "away_points_last_5",
    "home_goals_scored_last_5",
    "away_goals_scored_last_5",
    "home_goals_conceded_last_5",
    "away_goals_conceded_last_5",
    "home_elo_rating",
    "away_elo_rating",
    "home_promoted_this_season",
    "away_promoted_this_season",
    "home_team_strength_prior",
    "away_team_strength_prior",
    "home_strength_gap_vs_division_avg",
    "away_strength_gap_vs_division_avg",
    "odds_fulltime_home",
    "odds_fulltime_draw",
    "odds_fulltime_away",
    "handicap_line",
    "odds_handicap_home",
    "odds_handicap_away",
]

TOKEN_COUNT = len(TOKEN_NAMES)

# 0=team_stat, 1=market_odds, 2=handicap_line
TOKEN_TYPE_IDS = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 2, 1, 1,
]

# 0=neutral, 1=home, 2=away
TOKEN_SIDE_IDS = [
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    1, 0, 2, 0, 1, 2,
]

# Slot IDs distinguish repeated concept groups.
TOKEN_SLOT_IDS = [
    0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
    8, 9, 10, 11, 12, 13,
]

NUM_TOKEN_TYPES = max(TOKEN_TYPE_IDS) + 1
NUM_TOKEN_SIDES = max(TOKEN_SIDE_IDS) + 1
NUM_TOKEN_SLOTS = max(TOKEN_SLOT_IDS) + 1


def token_type_tensor(device=None):
    return torch.tensor(TOKEN_TYPE_IDS, dtype=torch.long, device=device)


def token_side_tensor(device=None):
    return torch.tensor(TOKEN_SIDE_IDS, dtype=torch.long, device=device)


def token_slot_tensor(device=None):
    return torch.tensor(TOKEN_SLOT_IDS, dtype=torch.long, device=device)
