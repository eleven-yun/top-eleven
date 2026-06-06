"""Unit tests for Sprint B schedule/fatigue features in build_dataset.py.

Tests:
  - rest_days: computed correctly from last match date
  - congestion_14d: counts matches within 14-day window correctly
  - rest_days default (7) when no previous match
  - season-start edge cases

Run with:
    python data/test_build_dataset_sprint_b.py
"""
import sys
import os
import unittest
from collections import defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.build_dataset import (
    initialise_team_state,
    update_team_state,
    team_feature_snapshot,
)


def _snap(state, team_id=1, opponent_id=2, match_dt=None):
    season_states = {team_id: state, opponent_id: initialise_team_state()}
    h2h = defaultdict(lambda: deque(maxlen=5))
    return team_feature_snapshot(
        team_id=team_id,
        team_name="T",
        season_name="2024/2025",
        division_level=1,
        season_states=season_states,
        previous_summaries={},
        head_to_head_history=h2h,
        opponent_team_id=opponent_id,
        current_match_datetime=match_dt,
    )


class TestRestDays(unittest.TestCase):
    def test_default_seven_when_no_previous_match(self):
        """Teams at season start have no last_match_date → default 7 days."""
        s = initialise_team_state()
        snap = _snap(s, match_dt="2024-08-17T15:00:00")
        self.assertEqual(snap["rest_days"], 7)

    def test_rest_days_computed_correctly(self):
        """After one match on Aug 10, next match Aug 17 → 7 days rest."""
        s = initialise_team_state()
        update_team_state(s, 2, 1, "2024-08-10T15:00:00")
        snap = _snap(s, match_dt="2024-08-17T15:00:00")
        self.assertEqual(snap["rest_days"], 7)

    def test_short_rest(self):
        """Match on Aug 10, next Aug 13 → 3 days rest."""
        s = initialise_team_state()
        update_team_state(s, 1, 0, "2024-08-10T20:00:00")
        snap = _snap(s, match_dt="2024-08-13T20:00:00")
        self.assertEqual(snap["rest_days"], 3)

    def test_long_rest(self):
        """Match on Aug 10, next Sep 14 → 35 days rest."""
        s = initialise_team_state()
        update_team_state(s, 0, 0, "2024-08-10T15:00:00")
        snap = _snap(s, match_dt="2024-09-14T15:00:00")
        self.assertEqual(snap["rest_days"], 35)

    def test_rest_days_nonnegative(self):
        """rest_days should always be >= 0."""
        s = initialise_team_state()
        update_team_state(s, 1, 1, "2024-08-20T15:00:00")
        snap = _snap(s, match_dt="2024-08-15T15:00:00")  # current before last (edge)
        self.assertGreaterEqual(snap["rest_days"], 0)


class TestCongestion14d(unittest.TestCase):
    def test_no_recent_matches(self):
        """No previous matches → congestion = 0."""
        s = initialise_team_state()
        snap = _snap(s, match_dt="2024-08-17T15:00:00")
        self.assertEqual(snap["congestion_14d"], 0)

    def test_one_match_within_14_days(self):
        s = initialise_team_state()
        update_team_state(s, 1, 0, "2024-08-10T15:00:00")
        snap = _snap(s, match_dt="2024-08-17T15:00:00")  # 7 days later
        self.assertEqual(snap["congestion_14d"], 1)

    def test_match_older_than_14_days_not_counted(self):
        s = initialise_team_state()
        update_team_state(s, 2, 0, "2024-08-01T15:00:00")
        snap = _snap(s, match_dt="2024-08-17T15:00:00")  # 16 days later
        self.assertEqual(snap["congestion_14d"], 0)

    def test_three_matches_in_14_days(self):
        s = initialise_team_state()
        for dt in ["2024-08-03T15:00:00", "2024-08-07T15:00:00", "2024-08-11T15:00:00"]:
            update_team_state(s, 1, 0, dt)
        snap = _snap(s, match_dt="2024-08-14T15:00:00")
        # Aug 3: 11 days ago ✓, Aug 7: 7 days ✓, Aug 11: 3 days ✓
        self.assertEqual(snap["congestion_14d"], 3)

    def test_exact_14_days_boundary_included(self):
        s = initialise_team_state()
        update_team_state(s, 1, 0, "2024-08-01T15:00:00")
        # Aug 1 + 14 days = Aug 15 → exactly 14 days → included
        snap = _snap(s, match_dt="2024-08-15T15:00:00")
        self.assertEqual(snap["congestion_14d"], 1)

    def test_same_day_not_counted(self):
        """A match on the exact same date (0 days) should not count in congestion."""
        s = initialise_team_state()
        update_team_state(s, 1, 0, "2024-08-17T12:00:00")
        snap = _snap(s, match_dt="2024-08-17T20:00:00")  # same day
        self.assertEqual(snap["congestion_14d"], 0)


class TestSprintBStateTracking(unittest.TestCase):
    def test_last_match_date_stored(self):
        s = initialise_team_state()
        update_team_state(s, 2, 1, "2024-08-10T15:00:00")
        self.assertEqual(s["last_match_date"], "2024-08-10T15:00:00")

    def test_last_match_date_updated_on_subsequent(self):
        s = initialise_team_state()
        update_team_state(s, 1, 0, "2024-08-10T15:00:00")
        update_team_state(s, 0, 2, "2024-08-17T15:00:00")
        self.assertEqual(s["last_match_date"], "2024-08-17T15:00:00")

    def test_recent_match_dates_deque(self):
        s = initialise_team_state()
        for i in range(16):
            update_team_state(s, 1, 0, f"2024-08-{i+1:02d}T15:00:00")
        self.assertEqual(len(s["recent_match_dates"]), 14)  # maxlen=14

    def test_sprint_b_fields_in_snapshot(self):
        s = initialise_team_state()
        snap = _snap(s, match_dt="2024-09-01T15:00:00")
        self.assertIn("rest_days", snap)
        self.assertIn("congestion_14d", snap)


if __name__ == "__main__":
    unittest.main(verbosity=2)
