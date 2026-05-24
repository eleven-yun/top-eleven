"""Unit tests for Sprint A feature engineering additions to build_dataset.py.

Tests cover:
  - New rolling state deques (wins_last_5, points_last_10, etc.)
  - New feature fields in team_feature_snapshot (win_rate, goal_diff, form_score_weighted, h2h_goal_diff)
  - H2H history tuple format change

Run with:
    python data/test_build_dataset_sprint_a.py
"""
import sys
import os
import unittest
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.build_dataset import (
    initialise_team_state,
    update_team_state,
    team_feature_snapshot,
)


class TestInitialiseTeamState(unittest.TestCase):
    def test_new_deques_present(self):
        state = initialise_team_state()
        for key in ("wins_last_5", "draws_last_5", "points_last_10",
                    "goals_for_last_10", "goals_against_last_10"):
            self.assertIn(key, state, f"Missing key: {key}")
        self.assertEqual(state["wins_last_5"].maxlen, 5)
        self.assertEqual(state["points_last_10"].maxlen, 10)


class TestUpdateTeamState(unittest.TestCase):
    def _fresh(self):
        return initialise_team_state()

    def test_win_increments_wins_last_5(self):
        s = self._fresh()
        update_team_state(s, 3, 0)
        self.assertEqual(list(s["wins_last_5"]), [1])
        self.assertEqual(list(s["draws_last_5"]), [0])
        self.assertEqual(list(s["points_last_5"]), [3])

    def test_draw_increments_draws_last_5(self):
        s = self._fresh()
        update_team_state(s, 1, 1)
        self.assertEqual(list(s["wins_last_5"]), [0])
        self.assertEqual(list(s["draws_last_5"]), [1])
        self.assertEqual(list(s["points_last_5"]), [1])

    def test_loss_is_zero_win_zero_draw(self):
        s = self._fresh()
        update_team_state(s, 0, 2)
        self.assertEqual(list(s["wins_last_5"]), [0])
        self.assertEqual(list(s["draws_last_5"]), [0])
        self.assertEqual(list(s["points_last_5"]), [0])

    def test_last10_deque_fills_correctly(self):
        s = self._fresh()
        for gf, ga in [(2, 0), (1, 1), (0, 1), (3, 2), (0, 0),
                       (1, 0), (2, 2), (0, 3), (1, 1), (3, 0), (2, 1)]:
            update_team_state(s, gf, ga)
        # deque(maxlen=10) should only keep last 10
        self.assertEqual(len(s["points_last_10"]), 10)

    def test_last5_deque_capacity(self):
        s = self._fresh()
        for i in range(7):
            update_team_state(s, 1, 0)
        self.assertEqual(len(s["wins_last_5"]), 5)
        self.assertEqual(len(s["points_last_5"]), 5)


class TestFormScoreWeighted(unittest.TestCase):
    """Test weighted form score in team_feature_snapshot."""

    def _build_state_from_results(self, results):
        """results: list of (goals_for, goals_against)"""
        s = initialise_team_state()
        for gf, ga in results:
            update_team_state(s, gf, ga)
        return s

    def _snapshot(self, state, team_id=1, opponent_id=2):
        """Call team_feature_snapshot with minimal args."""
        season_states = {team_id: state, opponent_id: initialise_team_state()}
        previous_summaries = {}
        h2h_history = {}
        from collections import defaultdict
        h2h_history = defaultdict(lambda: deque(maxlen=5))
        return team_feature_snapshot(
            team_id=team_id,
            team_name="TestTeam",
            season_name="2024/2025",
            division_level=1,
            season_states=season_states,
            previous_summaries=previous_summaries,
            head_to_head_history=h2h_history,
            opponent_team_id=opponent_id,
        )

    def test_all_wins_form_score_1(self):
        s = self._build_state_from_results([(2, 0)] * 5)
        snap = self._snapshot(s)
        # All wins (3pts each) with weights [1,2,3,4,5]:
        # sum(3*w for w in [1..5]) / (sum(1..5)*3) = 45/45 = 1.0
        self.assertAlmostEqual(snap["form_score_weighted"], 1.0, places=3)

    def test_all_losses_form_score_0(self):
        s = self._build_state_from_results([(0, 2)] * 5)
        snap = self._snapshot(s)
        self.assertAlmostEqual(snap["form_score_weighted"], 0.0, places=3)

    def test_no_matches_form_score_0(self):
        s = initialise_team_state()
        snap = self._snapshot(s)
        self.assertEqual(snap["form_score_weighted"], 0.0)

    def test_mixed_form_score_in_range(self):
        s = self._build_state_from_results([(2, 0), (0, 1), (1, 1), (3, 0), (0, 2)])
        snap = self._snapshot(s)
        self.assertGreaterEqual(snap["form_score_weighted"], 0.0)
        self.assertLessEqual(snap["form_score_weighted"], 1.0)


class TestWinRateAndGoalDiff(unittest.TestCase):
    def _snapshot(self, results, team_id=1, opponent_id=2):
        from collections import defaultdict
        s = initialise_team_state()
        for gf, ga in results:
            update_team_state(s, gf, ga)
        season_states = {team_id: s, opponent_id: initialise_team_state()}
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
        )

    def test_win_rate_all_wins(self):
        snap = self._snapshot([(2, 0)] * 5)
        self.assertAlmostEqual(snap["win_rate_last_5"], 1.0, places=3)

    def test_win_rate_no_wins(self):
        snap = self._snapshot([(0, 1)] * 5)
        self.assertAlmostEqual(snap["win_rate_last_5"], 0.0, places=3)

    def test_win_rate_partial(self):
        # 2 wins, 1 draw, 2 losses out of last 5
        snap = self._snapshot([(1, 0), (0, 0), (0, 1), (1, 0), (0, 1)])
        self.assertAlmostEqual(snap["win_rate_last_5"], 2 / 5, places=3)

    def test_goal_diff_last_5_positive(self):
        snap = self._snapshot([(3, 0), (2, 1), (1, 0), (2, 0), (1, 1)])
        # goals for: 9, against: 2 → diff = 7
        self.assertAlmostEqual(snap["goal_diff_last_5"], 7.0, places=2)

    def test_goal_diff_last_10(self):
        snap = self._snapshot([(1, 2)] * 10)  # all losses by 1
        self.assertAlmostEqual(snap["goal_diff_last_10"], -10.0, places=2)

    def test_new_fields_present(self):
        snap = self._snapshot([(1, 0)] * 3)
        for field in ("win_rate_last_5", "goal_diff_last_5", "points_last_10",
                      "goals_scored_last_10", "goals_conceded_last_10",
                      "goal_diff_last_10", "form_score_weighted", "h2h_goal_diff_last_5"):
            self.assertIn(field, snap, f"Missing field: {field}")


class TestH2HGoalDiff(unittest.TestCase):
    def test_h2h_goal_diff_home_perspective(self):
        """H2H goal diff from home perspective when team is always home."""
        from collections import defaultdict
        s1 = initialise_team_state()
        s2 = initialise_team_state()
        season_states = {1: s1, 2: s2}
        h2h = defaultdict(lambda: deque(maxlen=5))

        # Team 1 is home with score 3-1, 2-0, 1-2 in last 3 H2H matches
        for hg, ag in [(3, 1), (2, 0), (1, 2)]:
            winner = 1 if hg > ag else (2 if ag > hg else None)
            h2h[tuple(sorted([1, 2]))].append((winner, hg - ag, 1))  # home_team_id=1

        snap = team_feature_snapshot(
            team_id=1,
            team_name="Home",
            season_name="2024/2025",
            division_level=1,
            season_states=season_states,
            previous_summaries={},
            head_to_head_history=h2h,
            opponent_team_id=2,
        )
        # From team 1 (home) perspective: (3-1) + (2-0) + (1-2) = +2 + 2 + (-1) = +3
        self.assertAlmostEqual(snap["h2h_goal_diff_last_5"], 3.0, places=2)

    def test_h2h_goal_diff_away_perspective(self):
        """H2H goal diff from away team's perspective (negative of home diff)."""
        from collections import defaultdict
        s1 = initialise_team_state()
        s2 = initialise_team_state()
        season_states = {1: s1, 2: s2}
        h2h = defaultdict(lambda: deque(maxlen=5))

        # Team 1 was home, scored 2-1 vs team 2
        h2h[tuple(sorted([1, 2]))].append((1, 2 - 1, 1))  # home_team_id=1

        # From team 2 (away) perspective: gd = -(2-1) = -1
        snap = team_feature_snapshot(
            team_id=2,
            team_name="Away",
            season_name="2024/2025",
            division_level=1,
            season_states=season_states,
            previous_summaries={},
            head_to_head_history=h2h,
            opponent_team_id=1,
        )
        self.assertAlmostEqual(snap["h2h_goal_diff_last_5"], -1.0, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
