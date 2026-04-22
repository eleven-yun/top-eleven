# Experiment Log

- 2026-04-22: Added a reusable dataset quality audit command at scripts/data_quality_audit.py covering core missingness, class balance, split integrity (no overlap/leakage), and promoted-team counts by split; generated data/processed/data_quality_report.json as the baseline quality artifact.
- 2026-04-22: Expanded football-data.co.uk ingestion to include 2021/2022 for EPL and Championship; updated split to train=[2021/2022, 2022/2023], validation=[2023/2024], test=[2024/2025]; rebuild and audit passed with 3728 matches, zero split overlap/leakage, and promoted-team matches in train increased from 0 to 108.
