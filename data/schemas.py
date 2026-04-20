from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MatchMeta:
    match_id: str
    season: str
    league: str
    datetime_utc: str
    venue: str
    home_team_id: str
    away_team_id: str
    referee_id: str
    final_result: str
    home_goals: int
    away_goals: int
    halftime_home_goals: int
    halftime_away_goals: int


@dataclass
class TeamPreMatchFeatures:
    team_id: str
    league_position: Optional[int]
    points_last_5: Optional[int]
    goals_scored_last_5: Optional[float]
    goals_conceded_last_5: Optional[float]
    elo_rating: Optional[float]
    lineup: List[str]
    formation: Optional[str]
    h2h_wins_last_5: Optional[int]
    promoted_this_season: int
    relegated_last_season: int
    seasons_in_current_division: Optional[int]
    division_level_current: Optional[int]
    division_level_last_season: Optional[int]
    team_strength_prior: Optional[float]
    strength_gap_vs_division_avg: Optional[float]


@dataclass
class PreMatchFeatures:
    match_id: str
    home: TeamPreMatchFeatures
    away: TeamPreMatchFeatures


@dataclass
class LotteryMarket:
    match_id: str
    issue_id: str
    play_type: str
    handicap_line: Optional[int]
    home_odds: Optional[float]
    draw_odds: Optional[float]
    away_odds: Optional[float]


@dataclass
class ModelSample:
    match_id: str
    features: List[float]
    fulltime_label: int
    htft_label: Optional[int]
    handicap_label: Optional[int]
