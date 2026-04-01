# Design Document: Football Lottery Outcome Modeling (Pre-Match First)

## 1. Vision

This project builds a **football-only, pre-match prediction system** that maps model
probabilities to China Sports Lottery football products.

The system prioritizes:

1. Simpler architecture and data pipeline (no live streaming required)
2. Strong probability calibration
3. Clean mapping from model output to lottery play types

Initial focus is on these play types from the China lottery football products:

1. Fulltime 1X2
2. Halftime/Fulltime 1X2
3. Handicap 1X2

---

## 2. Problem Definition

### 2.1 Prediction Tasks

The core task family is multi-class classification with calibrated probabilities.

Task A: Fulltime 1X2

A 3-class calibrated classifier producing a single prediction before kick-off:

| Class | Label | Description |
|-------|-------|-------------|
| A | `home_win` | Home team wins at full time |
| B | `away_win` | Away team wins at full time |
| C | `draw` | The match ends in a draw |

Output: probability vector $[P(A), P(B), P(C)]$ with $P(A) + P(B) + P(C) = 1$.

Definition used in this project:

- Fulltime means regulation 90 minutes plus stoppage/injury time.
- Extra time and penalty shootout outcomes are excluded from labels.

Task B: Halftime/Fulltime 1X2

A 9-class calibrated classifier for the pair outcomes:

- `H/H`, `H/D`, `H/A`
- `D/H`, `D/D`, `D/A`
- `A/H`, `A/D`, `A/A`

Task C: Handicap 1X2

A 3-class calibrated classifier over outcome after applying official handicap line.

### 2.2 Prediction Cadence

Predictions are produced **once per match, before kick-off**.

### 2.3 Information Boundary (Critical)

> Only information available before kick-off is allowed as model input.
> Any in-play events, halftime statistics, and post-match fields are forbidden.

Information boundary is enforced at:
- Dataset build time (see Section 4).
- Feature engineering step (all rolling statistics use only past matches).
- Evaluation (one prediction per match).

### 2.4 Labels

- Fulltime 1X2 label: full-time result (`home_win | draw | away_win`)
- Halftime/Fulltime 1X2 label: pair of halftime and fulltime outcomes (9 classes)
- Handicap 1X2 label: full-time result after handicap adjustment

For all three play types, the fulltime component follows the same definition:
90 minutes plus stoppage/injury time, excluding extra time and penalties.

---

## 3. Scope (v1)

| Dimension | v1 Scope |
|-----------|----------|
| Sport | Football only |
| Product scope | Fulltime 1X2 first, then HT/FT and Handicap 1X2 |
| League | One league at a time |
| Seasons | 2 to 4 consecutive seasons |
| Inputs | Pre-match team history, lineup, schedule context |
| Live info | Excluded in v1 |
| Commentary/video | Excluded |

---

## 4. Data Schema

### 4.1 Match Record (`match_meta.json`)

One record per match.

```json
{
  "match_id": "string",
  "season": "string",
  "league": "string",
  "datetime_utc": "ISO8601",
  "venue": "string",
  "home_team_id": "string",
  "away_team_id": "string",
  "referee_id": "string",
  "final_result": "home_win | away_win | draw",
  "home_goals": "int",
  "away_goals": "int"
}
```

### 4.2 Pre-Match Static Features (`prematch_features.json`)

One record per match, computed from data prior to kick-off.

```json
{
  "match_id": "string",
  "home": {
    "team_id": "string",
    "league_position": "int",
    "points_last_5": "int",
    "goals_scored_last_5": "float",
    "goals_conceded_last_5": "float",
    "elo_rating": "float",
    "lineup": ["player_id"],
    "formation": "string",
    "h2h_wins_last_5": "int"
  },
  "away": {
    "...same fields as home..."
  }
}
```

### 4.2.1 Promotion/Relegation Fields (required)

To handle cross-season league movement, each team side should include these fields:

- `promoted_this_season`: `0 | 1`
- `relegated_last_season`: `0 | 1`
- `seasons_in_current_division`: `int`
- `division_level_current`: `int` (for example: top tier = 1)
- `division_level_last_season`: `int`
- `team_strength_prior`: `float` (pre-season initialized rating)
- `strength_gap_vs_division_avg`: `float`

For newly promoted teams with limited top-tier history, historical performance from
lower tiers must be transformed to division-relative features before use.

### 4.3 Lottery Market Fields (`lottery_market.json`)

Lottery-side identifiers and market settings used for label mapping and evaluation.

```json
{
  "match_id": "string",
  "issue_id": "string",
  "play_type": "fulltime_1x2 | htft_1x2 | handicap_1x2",
  "handicap_line": "int | null",
  "home_odds": "float | null",
  "draw_odds": "float | null",
  "away_odds": "float | null"
}
```

### 4.4 Match Event Timeline (`events.jsonl`) [optional, not used in v1]

One row per event, sorted by minute.

```json
{
  "match_id": "string",
  "minute": "int",
  "extra_time_minute": "int | null",
  "event_type": "goal | shot | shot_on_target | corner | yellow_card | red_card | substitution | foul | var_review | penalty",
  "team": "home | away",
  "player_id": "string | null",
  "xg": "float | null",
  "details": {}
}
```

### 4.5 Minute-Level State Snapshot (`timeline.jsonl`) [optional, not used in v1]

One row per (match, minute) pair. This is the primary training table.

```json
{
  "match_id": "string",
  "minute": "int",
  "home_goals": "int",
  "away_goals": "int",
  "home_shots": "int",
  "away_shots": "int",
  "home_shots_on_target": "int",
  "away_shots_on_target": "int",
  "home_xg_cumulative": "float",
  "away_xg_cumulative": "float",
  "home_yellow_cards": "int",
  "away_yellow_cards": "int",
  "home_red_cards": "int",
  "away_red_cards": "int",
  "home_corners": "int",
  "away_corners": "int",
  "home_possession_last_5min": "float",
  "away_possession_last_5min": "float",
  "label": "home_win | away_win | draw"
}
```

### 4.6 Commentary Transcript (`commentary.jsonl`) [deferred]

```json
{
  "match_id": "string",
  "minute": "int",
  "text": "string",
  "source": "asr | manual"
}
```

### 4.7 Video Clips (`clips/`) [deferred]

Clips are stored as pre-extracted embeddings, not raw video.

```json
{
  "match_id": "string",
  "minute": "int",
  "clip_start_sec": "float",
  "clip_end_sec": "float",
  "embedding_path": "path/to/npy",
  "encoder": "string"
}
```

### 4.8 Feature Catalog (v1)

The schema defines available fields; this catalog defines model inputs and
transformations.

#### 4.8.1 Tier A: Base Handcrafted Features

| Feature | Definition (pre-match only) | Window |
|---------|-----------------------------|--------|
| home_points_last_5 | Sum of home team points in last 5 matches | 5 matches |
| away_points_last_5 | Sum of away team points in last 5 matches | 5 matches |
| home_goal_diff_last_5 | (goals for - goals against) for home team | 5 matches |
| away_goal_diff_last_5 | (goals for - goals against) for away team | 5 matches |
| home_win_rate_last_10 | Home team win rate before this match | 10 matches |
| away_win_rate_last_10 | Away team win rate before this match | 10 matches |
| elo_home | Home team pre-match strength rating | rolling |
| elo_away | Away team pre-match strength rating | rolling |
| elo_gap | elo_home - elo_away | rolling |
| rest_days_gap | Home rest days - away rest days | fixture-based |

#### 4.8.2 Tier B: Semantic Handcrafted Features

| Feature | Definition | Notes |
|---------|------------|-------|
| form_momentum_gap | Weighted recent form score(home) - score(away) | Higher weight on latest matches |
| attack_defense_balance_gap | (home attack - away defense) - (away attack - home defense) | Composite strength interaction |
| schedule_congestion_gap | Recent match density(home) - density(away) | Captures fatigue pressure |
| lineup_continuity_gap | Returning starters ratio(home) - ratio(away) | Requires lineup history |
| coach_tenure_gap | Coach tenure days(home) - tenure days(away) | Proxy for tactical stability |
| coach_change_flag_pair | Binary pair describing any recent coach changes | Use small lookback, e.g. 30 days |
| promoted_risk_gap | Promotion risk score(home) - score(away) | Derived from promotion indicators |
| division_adjusted_strength_gap | team_strength_prior(home) - team_strength_prior(away) after shrinkage | Required for promoted teams |
| h2h_context_score | Strength-adjusted head-to-head signal | Downweight stale h2h matches |
| motivation_proxy_gap | Proxy from competition stage/importance(home-away) | Use conservative encoding |

#### 4.8.3 Play-Type Usage Matrix

| Feature Group | Fulltime 1X2 | HT/FT 1X2 | Handicap 1X2 |
|---------------|--------------|-----------|--------------|
| Tier A base team form | required | required | required |
| Promotion/relegation fields | required | required | required |
| Tier B semantic composites | optional in v1, likely useful | optional in v1 | recommended |
| Player/coach aggregates | optional in v1 | optional | optional |

#### 4.8.4 Leakage and Quality Rules

Each feature must satisfy all rules below before entering training:

1. Uses only data available before kick-off of the target match.
2. Uses only matches strictly earlier than target match datetime.
3. Uses fixed windows or explicitly documented rolling logic.
4. Has missing-value handling documented (default, impute, or drop rule).
5. Includes a reproducible computation function in data pipeline code.

#### 4.8.5 Player and Coach Feature Policy

Player and coach features can be useful, but they are phase-gated to control
complexity and data collection cost.

v1 policy:

1. Do not model individual player embeddings.
2. Use only low-dimensional aggregates if data is reliable:
  - lineup_continuity
  - injured_or_absent_starter_count
  - average_minutes_of_expected_starters
3. Use coach-level aggregates only:
  - coach_tenure_days
  - recent_coach_change_flag
4. Add these features only after Tier A baseline is stable.

Escalation policy:

- If aggregate player/coach features improve held-out log loss and calibration,
  keep them.
- If they do not improve robustness, revert to team-level feature set.

---

## 5. Dataset Split

> Splits must be **by season**, never by random match.  
> Splitting randomly leaks future season knowledge into training.

| Split | Seasons |
|-------|---------|
| Train | Seasons 1 – N-2 |
| Validation | Season N-1 |
| Test | Season N |

Within training, a held-out subset of full matches (not minutes) is used for early
stopping to avoid leaking validation season statistics.

Promotion/relegation handling requirement for splits:

- Ensure each split records the number of matches involving newly promoted teams.
- Report separate validation/test statistics on promoted-team fixtures.
- Avoid filtering out promoted teams, since this creates unrealistic bias.

---

## 6. Evaluation Metrics

All metrics are computed per play type and globally.

| Metric | Rationale |
|--------|-----------|
| Log loss | Primary. Measures calibrated probabilistic predictions. |
| Brier score | Quadratic probability score. More interpretable than log loss. |
| Accuracy (argmax) | Secondary. Not sufficient alone for probability outputs. |
| Expected Calibration Error (ECE) | Measures over/under-confidence. |
| AUC (one-vs-rest) | Per-class discrimination ability. |

**Play-type slices** used in reporting:
- Fulltime 1X2
- Halftime/Fulltime 1X2
- Handicap 1X2

**Ablation axis**: pre-match structured baseline → improved structured model

**Promotion/relegation slices** used in reporting:

- Promoted-team matches (home or away promoted in current season)
- Non-promoted matches
- Early-season promoted matches (first 8 matchdays)

Calibration and log loss must be reviewed on these slices before releasing a model.

---

## 7. Target Model Architecture

### 7.1 v1: Structured Pre-Match Classifier

```
Static Encoder
  ├── Pre-match team features (concat → linear → d_model)
  └── Positional information (venue, league)

Match-History Encoder (Transformer/MLP)
  ├── Input: last-N match summary features
  └── Output: contextualized history representation

Fusion
  └── Concatenation + projection

Classification Head
  ├── Head A: Fulltime 1X2 (3 classes)
  ├── Head B: HT/FT 1X2 (9 classes)
  └── Head C: Handicap 1X2 (3 classes)
```

### 7.2 v2: Multi-Task and Calibration Refinements

```
Shared Backbone
  ├── Team strength and form features
  └── Opponent interaction features

Task-specific heads
  ├── Fulltime 1X2
  ├── HT/FT 1X2
  └── Handicap 1X2

Post-hoc calibration
  └── Temperature scaling / isotonic calibration per head
```

### 7.3 Team-Strength Prior Across Seasons

To reduce drift caused by promotion/relegation:

1. Maintain a rolling team-strength rating (for example ELO-like) over matches.
2. At new season start, initialize each team with prior-season closing rating.
3. Apply shrinkage toward current division average at season reset.
4. Apply stronger shrinkage for promoted teams than established top-tier teams.
5. Use this adjusted rating as `team_strength_prior` in pre-match features.

---

## 8. Target Repository Structure

```
top_eleven/
├── config/
│   ├── config.json            # model hyperparameters
│   ├── data_config.json       # raw data paths, league/season selection
│   ├── feature_config.json    # pre-match feature inclusion flags
│   └── experiment_config.json # experiment tracking settings
├── data/
│   ├── raw/                   # original downloaded data (unmodified)
│   ├── processed/             # cleaned, normalised, pre-match tables
│   ├── schemas.py             # dataclass definitions for all tables
│   ├── build_dataset.py       # raw → processed pipeline
│   └── data_loader.py         # PyTorch Dataset / DataLoader
├── nn_modules/
│   ├── encoders/
│   │   ├── static_encoder.py
│   │   └── history_encoder.py
│   ├── fusion/
│   │   └── gated_fusion.py
│   ├── heads/
│   │   └── classification_head.py
│   └── multitask/
│       └── lottery_predictor.py # shared backbone + task heads
├── scripts/
│   ├── build_dataset.py
│   ├── train_baseline.py
│   ├── train_multitask.py
│   ├── eval.py
│   └── backtest.py
├── utils/
│   ├── metrics.py             # log loss, Brier, ECE, AUC
│   ├── calibration.py
│   ├── split.py
│   └── logging.py
├── docs/
│   ├── design.md              # this file
│   └── milestones.md
└── README.md
```

---

## 9. Non-Goals (v1)

- End-to-end video or speech encoding training from scratch.
- Live in-play forecasting.
- Real-time streaming inference.
- Multi-league joint training in v1.
- Non-football sports.

## 10. Promotion/Relegation Policy (Operational)

This policy is mandatory for all experiments:

1. Do not assume team quality is stationary across seasons.
2. Add promotion/relegation indicators as first-class features.
3. Use season-reset team-strength priors with division-based shrinkage.
4. Run dedicated evaluation slices for promoted-team matches.
5. If promoted-team calibration is poor, apply segment-specific calibration before
  considering deployment.
---

## 11. Tournament (Non-League) Matches: Phase 5+ Deferral

The current v1 system is designed for domestic league football. Tournament matches
(World Cup, Euros, Copa America, etc.) have fundamentally different contexts and
data properties.

### Design Constraints in v1

- Assumes rolling team form from repeated league matches (last 5, ELO trajectory).
- Promotion/relegation handling is irrelevant for national teams.
- No pre-tournament training window or seasonal structure.
- League-based team strength priors will not transfer to international level.

### Adaptation Strategy for Tournaments (Deferred to Phase 5+)

If tournament support becomes a priority (for example, World Cup 2026):

1. **Feature Engineering**
   - Replace rolling domestic-form features with international-match history.
   - Use tournament pre-tournament preparation metrics (friendlies, squad depth).
   - Add tournament-stage indicators (group vs knockout vs finals).
   - Use international strength priors (FIFA ranking, international ELO) instead of
     domestic league ELO.

2. **Data Requirements**
   - Curate separate tournament match datasets with pre-tournament team records.
   - Extract international match history with appropriate lookback windows.
   - Handle missing or sparse national-team records with regularized priors.

3. **Label Mapping**
   - Fulltime and Handicap labels remain unchanged (90 mins + stoppage).
   - HT/FT 1X2 labeling is unchanged but may be less reliable in
     knockout tournaments where extra time and penalties change late-stage dynamics.
   - Consider separate models for group stage vs knockout stage if data permits.

4. **Evaluation Strategy**
   - Evaluate on cross-tournament generalization (train on past tournaments, test on
     newly held tournament).
   - Separate calibration for tournament vs league predictions if both are in
     production.

### Recommendation

- v1 focuses entirely on domestic league matches.
- Do not attempt to model tournaments with league-trained models without explicit
  feature reengineering.
- Defer tournament support until after Phase 4 (backtest & strategy layer) is
  stable for leagues.
- If World Cup 2026 is a hard deadline, consider a separate minimal tournament-only
  model in parallel (international ranking gap, recent international form,
  squad freshness proxy).