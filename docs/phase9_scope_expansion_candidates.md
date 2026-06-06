# Phase 9 Scope Expansion Candidates

This note captures the largest out-of-scope league buckets from
`data/processed/prematch_scope_gap_report.json` to guide dataset-scope expansion.

## Current Gap Snapshot

- Out-of-scope rows: 9641
- Out-of-scope share: 55.55%
- Current pre-match coverage on 2025/26 test: 56.3%
- Phase 9 acceptance target: >= 70% pre-match CN coverage

## Top Candidate Competitions

Ranked by share of out-of-scope rows in current pre-match scrape:

| Rank | Competition (resolved) | Rows | Share of out-of-scope |
|---|---|---:|---:|
| 1 | 欧冠 | 678 | 7.03% |
| 2 | eredivisie | 647 | 6.71% |
| 3 | primeira liga | 614 | 6.37% |
| 4 | j-league | 574 | 5.95% |
| 5 | 欧罗巴 | 553 | 5.74% |
| 6 | mls | 529 | 5.49% |
| 7 | allsvenskan | 489 | 5.07% |
| 8 | eliteserien | 470 | 4.88% |
| 9 | 澳超 | 462 | 4.79% |
| 10 | K1联赛 | 454 | 4.71% |
| 11 | 荷乙 | 333 | 3.45% |
| 12 | 日职乙 | 288 | 2.99% |

## Execution Order Suggestion

1. Domestic leagues with stable season data feed support first (for faster integration and model retraining).
2. International cups second (higher variance, multi-country scheduling, naming complexity).
3. Re-run `scripts/enrich_prematch_odds.py` and `scripts/cn_lottery_backtest.py` after each scope batch.

## Notes

- This file is intentionally data-driven and should be refreshed after each scrape window refresh.
- Keep `config/data_config.json` updates incremental (batch of 2-4 competitions) to isolate coverage impact.
