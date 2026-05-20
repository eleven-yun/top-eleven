import torch

# Fixed token order for V2 match representation (Sprint A feature engineering).
# Positions 0-15:  original 16 team-stat tokens
# Positions 16-31: Sprint A extended form tokens (8 per side)
# Positions 32-34: Sprint A gap tokens (neutral side)
# Positions 35-40: market tokens (unchanged)
TOKEN_NAMES = [
    # --- original team stats (slots 0-7) ---
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
    # --- Sprint A: extended form per team (slots 14-21) ---
    "home_win_rate_last_5",
    "away_win_rate_last_5",
    "home_goal_diff_last_5",
    "away_goal_diff_last_5",
    "home_points_last_10",
    "away_points_last_10",
    "home_goals_scored_last_10",
    "away_goals_scored_last_10",
    "home_goals_conceded_last_10",
    "away_goals_conceded_last_10",
    "home_goal_diff_last_10",
    "away_goal_diff_last_10",
    "home_form_score_weighted",
    "away_form_score_weighted",
    "home_h2h_goal_diff_last_5",
    "away_h2h_goal_diff_last_5",
    # --- Sprint A: gap features (slots 22-24, neutral side) ---
    "elo_gap",
    "points_gap_last_5",
    "goal_diff_gap_last_5",
    # --- market tokens (slots 25-30) ---
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
    # original 16 team stats
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # Sprint A: 16 new team stats
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # Sprint A: 3 gap tokens
    0, 0, 0,
    # market tokens
    1, 1, 1, 2, 1, 1,
]

# 0=neutral, 1=home, 2=away
TOKEN_SIDE_IDS = [
    # original 16 team stats: alternating home/away pairs
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    # Sprint A: 16 new team stats: alternating home/away pairs
    1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
    # Sprint A: 3 gap tokens (neutral)
    0, 0, 0,
    # market tokens
    1, 0, 2, 0, 1, 2,
]

# Slot IDs distinguish repeated concept groups.
TOKEN_SLOT_IDS = [
    # original 16 team stats
    0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7,
    # Sprint A: 16 new team stats (slots 14-21)
    14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21,
    # Sprint A: 3 gap tokens (slots 22-24)
    22, 23, 24,
    # market tokens (slots 25-30)
    25, 26, 27, 28, 29, 30,
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
