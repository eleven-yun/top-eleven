"""Unit tests for enrich_lottery_odds.py.

Covers:
  - Original 7 review-comment fixes (parse_iso_dt, season bonus guard, date
    indexing, datetime-based deduplication, team name enrichment).
  - 5 fixes from the second review round (play_type=None handling, falsy team
    name check, no_scored_candidates reason, and test comment clarity).

Run with:
    python scripts/test_enrich_lottery_odds.py
"""

import sys
import os
import unittest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enrich_lottery_odds import (
    parse_iso_dt,
    score_candidate,
    _build_meta_date_index,
    _get_meta_candidates,
    match_odds_to_meta,
    deduplicate_market_records,
    build_alias_index,
)


# ---------------------------------------------------------------------------
# Fix #1 + #2: parse_iso_dt — no timezone, various formats
# ---------------------------------------------------------------------------

class TestParseIsoDt(unittest.TestCase):
    def test_full_datetime(self):
        dt = parse_iso_dt("2024-09-15T15:00:00")
        self.assertEqual(dt, datetime(2024, 9, 15, 15, 0, 0))

    def test_datetime_no_seconds(self):
        dt = parse_iso_dt("2024-09-15T15:00")
        self.assertEqual(dt, datetime(2024, 9, 15, 15, 0))

    def test_space_separated(self):
        dt = parse_iso_dt("2024-09-15 15:00:00")
        self.assertEqual(dt, datetime(2024, 9, 15, 15, 0, 0))

    def test_date_only(self):
        dt = parse_iso_dt("2024-09-15")
        self.assertEqual(dt, datetime(2024, 9, 15, 0, 0))

    def test_none_input(self):
        self.assertIsNone(parse_iso_dt(None))

    def test_empty_string(self):
        self.assertIsNone(parse_iso_dt(""))

    def test_invalid_string(self):
        self.assertIsNone(parse_iso_dt("not-a-date"))

    def test_returns_naive(self):
        """Result must be a naive datetime (no tzinfo) — Fix #1."""
        dt = parse_iso_dt("2024-09-15T15:00:00")
        self.assertIsNotNone(dt, "parse_iso_dt should not return None for a valid timestamp")
        self.assertIsNone(dt.tzinfo)


# ---------------------------------------------------------------------------
# Fix #5: season bonus only fires when score > 0 already
# ---------------------------------------------------------------------------

class TestScoreCandidateSeasonBonus(unittest.TestCase):
    def _make_meta(self, dt_utc):
        return {
            "match_id": "test-1",
            "datetime_utc": dt_utc,
            "league": "",
            "country_name": "",
            "league_code": "",
            "home_team_name": "",
            "away_team_name": "",
        }

    def test_season_bonus_not_awarded_when_no_other_signal(self):
        """Fix #5: season +0.05 must not fire when all other signals scored 0."""
        odds_row = {
            "home_team_raw": "ZZZ Unknown FC",
            "away_team_raw": "AAA Unknown FC",
            "league_name_raw": "ZZZ Unknown League",
        }
        kickoff_dt = datetime(2024, 9, 15, 15, 0, 0)
        # Use a meta_row with a deliberately far kickoff (2020) so no signal fires:
        # teams don't match, league doesn't match, kickoff is >12h away → no proximity bonus.
        # The season bonus must NOT fire in this case and make an otherwise-zero score nonzero.
        meta_row_far = self._make_meta("2020-01-01T12:00:00")
        score_far = score_candidate(odds_row, meta_row_far, {}, kickoff_dt)
        self.assertEqual(score_far, 0.0, "Season bonus must not fire without other signals")

    def test_season_bonus_awarded_when_kickoff_matches(self):
        """Fix #5: season +0.05 fires only when at least kickoff proximity matched."""
        odds_row = {
            "home_team_raw": "",
            "away_team_raw": "",
            "league_name_raw": "",
        }
        meta_row = self._make_meta("2024-09-15T15:00:00")
        kickoff_dt = datetime(2024, 9, 15, 15, 0, 0)
        score = score_candidate(odds_row, meta_row, {}, kickoff_dt)
        # Kickoff is exact → +0.30, same year → +0.05 → total 0.35
        self.assertAlmostEqual(score, 0.35, places=3)


# ---------------------------------------------------------------------------
# Fix #6: date-indexed candidate narrowing
# ---------------------------------------------------------------------------

class TestMetaDateIndex(unittest.TestCase):
    def _make_meta(self, match_id, dt_utc):
        return {"match_id": match_id, "datetime_utc": dt_utc}

    def test_index_groups_by_date(self):
        rows = [
            self._make_meta("A", "2024-09-15T15:00:00"),
            self._make_meta("B", "2024-09-16T20:00:00"),
            self._make_meta("C", "2024-09-17T18:00:00"),
            self._make_meta("D", None),
        ]
        date_index, no_date = _build_meta_date_index(rows)
        self.assertIn("2024-09-15", date_index)
        self.assertIn("2024-09-16", date_index)
        self.assertIn("2024-09-17", date_index)
        self.assertEqual(len(no_date), 1)
        self.assertEqual(no_date[0]["match_id"], "D")

    def test_candidates_within_window(self):
        rows = [
            self._make_meta("A", "2024-09-15T15:00:00"),
            self._make_meta("B", "2024-09-16T20:00:00"),
            self._make_meta("C", "2024-09-30T18:00:00"),
        ]
        date_index, no_date = _build_meta_date_index(rows)
        kickoff_dt = datetime(2024, 9, 15, 12, 0, 0)
        candidates = _get_meta_candidates(kickoff_dt, date_index, no_date, window_days=1)
        ids = {r["match_id"] for r in candidates}
        self.assertIn("A", ids)
        self.assertIn("B", ids)
        self.assertNotIn("C", ids)

    def test_no_kickoff_returns_all(self):
        rows = [
            self._make_meta("A", "2024-09-15T15:00:00"),
            self._make_meta("B", "2024-09-16T20:00:00"),
        ]
        date_index, no_date = _build_meta_date_index(rows)
        candidates = _get_meta_candidates(None, date_index, no_date, window_days=1)
        ids = {r["match_id"] for r in candidates}
        self.assertEqual(ids, {"A", "B"})


# ---------------------------------------------------------------------------
# Fix #7: datetime-based deduplication
# ---------------------------------------------------------------------------

class TestDeduplicateMarketRecords(unittest.TestCase):
    def _rec(self, match_id, ts, odds_home):
        return {
            "match_id": match_id,
            "play_type": "fulltime_1x2",
            "odds_stage": "close",
            "odds_capture_time": ts,
            "home_odds": odds_home,
        }

    def test_keeps_latest_by_datetime(self):
        """Fix #7: use datetime comparison, not string comparison."""
        records = [
            self._rec("m1", "2024-09-15T10:00:00", 1.8),
            self._rec("m1", "2024-09-15T12:00:00", 2.1),  # later → keep this
            self._rec("m1", "2024-09-15T09:00:00", 1.5),
        ]
        result = deduplicate_market_records(records)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["home_odds"], 2.1)

    def test_fallback_string_when_both_unparseable(self):
        """Fix #7: fall back to string comparison when neither ts parses."""
        records = [
            self._rec("m1", "bad-ts-A", 1.8),
            self._rec("m1", "bad-ts-B", 2.1),  # "bad-ts-B" >= "bad-ts-A" lexically
        ]
        result = deduplicate_market_records(records)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["home_odds"], 2.1)

    def test_null_ts_replaced_by_parseable(self):
        """Fix #7: a parseable ts always wins over a missing ts."""
        records = [
            self._rec("m1", None, 1.8),
            self._rec("m1", "2024-09-15T12:00:00", 2.1),
        ]
        result = deduplicate_market_records(records)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["home_odds"], 2.1)

    def test_different_keys_kept_separate(self):
        records = [
            self._rec("m1", "2024-09-15T10:00:00", 1.8),
            self._rec("m2", "2024-09-15T10:00:00", 2.2),
        ]
        result = deduplicate_market_records(records)
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# Fix #4: team name enrichment flows through match_odds_to_meta
# ---------------------------------------------------------------------------

class TestTeamNameEnrichment(unittest.TestCase):
    """Verify that enriched home_team_name / away_team_name in meta_rows
    allows the team-score component to produce non-zero scores, making
    match_odds_to_meta succeed where numeric IDs alone would not."""

    def _meta_row(self, match_id, name_home, name_away, dt_utc):
        return {
            "match_id": match_id,
            "home_team_id": "1001",
            "away_team_id": "1002",
            "home_team_name": name_home,
            "away_team_name": name_away,
            "datetime_utc": dt_utc,
            "league": "Premier League",
            "country_name": "England",
            "league_code": "E0",
        }

    def test_exact_team_name_match_passes_threshold(self):
        """Exact team name match should achieve score >= 0.75."""
        odds_row = {
            "home_team_raw": "Man United",
            "away_team_raw": "Arsenal",
            "league_name_raw": "Premier League",
            "kickoff_local": "2024-09-15T15:00:00",
            "play_type_raw": "fulltime_1x2",
            "odds_home": 2.1, "odds_draw": 3.4, "odds_away": 3.2,
        }
        meta_rows = [
            self._meta_row("m1", "Man United", "Arsenal", "2024-09-15T15:00:00"),
            self._meta_row("m2", "Liverpool", "Chelsea", "2024-09-22T15:00:00"),
        ]
        alias_index = build_alias_index({"teams": {}})
        matched, unresolved = match_odds_to_meta(
            [odds_row], meta_rows, alias_index, min_score=0.75, min_gap=0.05
        )
        self.assertEqual(len(matched), 1, f"Expected 1 match, got unresolved: {unresolved}")
        self.assertEqual(matched[0]["match_id"], "m1")

    def test_numeric_id_only_fails_threshold(self):
        """Without team name enrichment (only numeric IDs), team score = 0 and
        match should fail unless other signals are sufficient."""
        odds_row = {
            "home_team_raw": "Man United",
            "away_team_raw": "Arsenal",
            "league_name_raw": "",
            "kickoff_local": "2024-09-15T15:00:00",
            "play_type_raw": "fulltime_1x2",
        }
        # meta_rows with no team names (simulate un-enriched state)
        meta_rows = [
            {
                "match_id": "m1",
                "home_team_id": "1001",
                "away_team_id": "1002",
                "home_team_name": "",
                "away_team_name": "",
                "datetime_utc": "2024-09-15T15:00:00",
                "league": "", "country_name": "", "league_code": "",
            },
        ]
        alias_index = build_alias_index({"teams": {}})
        matched, unresolved = match_odds_to_meta(
            [odds_row], meta_rows, alias_index, min_score=0.75, min_gap=0.05
        )
        # Kickoff exact match gives 0.30, season bonus 0.05 → total 0.35 < 0.75
        self.assertEqual(len(matched), 0, "Should fail threshold without team names")


# ---------------------------------------------------------------------------
# New fix #1: unknown play_type marked unresolved
# ---------------------------------------------------------------------------

class TestUnknownPlayType(unittest.TestCase):
    def _meta_row(self, match_id, dt_utc):
        return {
            "match_id": match_id,
            "home_team_name": "Man United",
            "away_team_name": "Arsenal",
            "datetime_utc": dt_utc,
            "league": "Premier League",
            "country_name": "England",
            "league_code": "E0",
        }

    def test_unknown_play_type_raw_is_unresolved(self):
        """New fix: odds rows with unknown/missing play_type_raw must be unresolved."""
        odds_row = {
            "home_team_raw": "Man United",
            "away_team_raw": "Arsenal",
            "kickoff_local": "2024-09-15T15:00:00",
            "play_type_raw": "some_unknown_type",
        }
        meta_rows = [self._meta_row("m1", "2024-09-15T15:00:00")]
        alias_index = build_alias_index({"teams": {}})
        matched, unresolved = match_odds_to_meta(
            [odds_row], meta_rows, alias_index, min_score=0.0, min_gap=0.0
        )
        self.assertEqual(len(matched), 0)
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["reason"], "unknown_play_type")

    def test_missing_play_type_raw_is_unresolved(self):
        """New fix: odds rows with no play_type_raw field must be unresolved."""
        odds_row = {
            "home_team_raw": "Man United",
            "away_team_raw": "Arsenal",
            "kickoff_local": "2024-09-15T15:00:00",
        }
        meta_rows = [self._meta_row("m1", "2024-09-15T15:00:00")]
        alias_index = build_alias_index({"teams": {}})
        matched, unresolved = match_odds_to_meta(
            [odds_row], meta_rows, alias_index, min_score=0.0, min_gap=0.0
        )
        self.assertEqual(len(matched), 0)
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["reason"], "unknown_play_type")


# ---------------------------------------------------------------------------
# New fix #2: no_candidates vs no_scored_candidates reason
# ---------------------------------------------------------------------------

class TestUnresolvedReasonAccuracy(unittest.TestCase):
    def _meta_row(self, match_id, dt_utc):
        return {
            "match_id": match_id,
            "home_team_name": "",
            "away_team_name": "",
            "datetime_utc": dt_utc,
            "league": "", "country_name": "", "league_code": "",
        }

    def test_no_candidates_reason_when_meta_empty(self):
        """New fix: 'no_candidates' when no date-bucketed candidates exist."""
        odds_row = {
            "home_team_raw": "Team A",
            "away_team_raw": "Team B",
            "kickoff_local": "2024-09-15T15:00:00",
            "play_type_raw": "fulltime_1x2",
        }
        alias_index = build_alias_index({"teams": {}})
        matched, unresolved = match_odds_to_meta(
            [odds_row], [], alias_index, min_score=0.75, min_gap=0.10
        )
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["reason"], "no_candidates")

    def test_no_scored_candidates_reason_when_all_score_zero(self):
        """New fix: 'no_scored_candidates' when candidates exist but all score 0.

        The meta_row is on the same calendar date as the odds kickoff so it enters
        the date bucket, but the kickoff is >12h apart and there is no team/league
        overlap, so score_candidate returns 0.0 for every candidate.
        """
        # odds kickoff: 2024-09-15 00:30 (just after midnight)
        odds_row = {
            "home_team_raw": "ZZZ Unknown FC",
            "away_team_raw": "AAA Unknown FC",
            "kickoff_local": "2024-09-15T00:30:00",
            "league_name_raw": "ZZZ Unknown League",
            "play_type_raw": "fulltime_1x2",
        }
        # meta kickoff: 2024-09-15 15:00 — same date → in bucket, but 14.5h apart → no proximity
        meta_rows = [self._meta_row("m1", "2024-09-15T15:00:00")]
        alias_index = build_alias_index({"teams": {}})
        matched, unresolved = match_odds_to_meta(
            [odds_row], meta_rows, alias_index, min_score=0.75, min_gap=0.10
        )
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0]["reason"], "no_scored_candidates")


if __name__ == "__main__":
    unittest.main(verbosity=2)
