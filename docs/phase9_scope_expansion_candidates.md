# Phase 9 Scope Expansion Candidates

This note captures the largest out-of-scope league buckets from
`data/processed/prematch_scope_gap_report.json` to guide dataset-scope expansion.

## Current Gap Snapshot

- Out-of-scope rows: 8380 (after scope expansion batch-1)
- Out-of-scope share: 48.30% (down from 55.55%)
- Current pre-match coverage on 2025/26 test: 56.3%
- Phase 9 acceptance target: >= 70% pre-match CN coverage
- Current in-scope rows under expanded config: 8974 (up from 7713)
- Current in-scope matching coverage: 84.7%
- Current in-scope unresolved alias backlog: 674 rows

## Top Candidate Competitions

Ranked by share of out-of-scope rows in current pre-match scrape:

| Rank | Competition (resolved) | Rows | Share of out-of-scope |
|---|---|---:|---:|
| 1 | 欧冠 | 678 | 8.09% |
| 2 | j-league | 574 | 6.85% |
| 3 | 欧罗巴 | 553 | 6.60% |
| 4 | mls | 529 | 6.31% |
| 5 | allsvenskan | 489 | 5.84% |
| 6 | eliteserien | 470 | 5.61% |
| 7 | 澳超 | 462 | 5.51% |
| 8 | K1联赛 | 454 | 5.42% |
| 9 | 荷乙 | 333 | 3.97% |
| 10 | 日职乙 | 288 | 3.44% |
| 11 | 俄超 | 285 | 3.40% |
| 12 | 德丙 | 231 | 2.76% |

## Execution Order Suggestion

1. Domestic leagues with stable season data feed support first (for faster integration and model retraining).
2. International cups second (higher variance, multi-country scheduling, naming complexity).
3. Re-run `scripts/enrich_prematch_odds.py` and `scripts/cn_lottery_backtest.py` after each scope batch.

## Notes

- This file is intentionally data-driven and should be refreshed after each scrape window refresh.
- Keep `config/data_config.json` updates incremental (batch of 2-4 competitions) to isolate coverage impact.
