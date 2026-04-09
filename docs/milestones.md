# Milestones

Tracked by phase. Mark items `[x]` when done. Stop at each **Decision Gate** and
evaluate before proceeding to the next phase.

---

## Phase 0 — Lock the Product Scope
*Goal: fix lottery products, labels, and pre-match-only boundaries before coding.*

- [ ] Confirm football-only scope
- [ ] Confirm play type order: Fulltime 1X2 -> HT/FT 1X2 -> Handicap 1X2
- [ ] Confirm prediction timing: once before kick-off
- [ ] Write down pre-match information boundary and examples of forbidden fields
- [ ] Confirm single league and number of seasons for v1
- [ ] Confirm single league and number of seasons for v1 (tournament matches out of scope)
- [ ] Define promotion/relegation policy for new season transition
- [ ] Document all decisions in `docs/design.md` Section 3 (Scope)
- [ ] Fix final metric set and acceptance thresholds

---

## Phase 1 — Build Pre-Match Dataset Backbone
*Goal: replace synthetic tensors with a real, leakage-free pre-match dataset.*

### Data acquisition
- [ ] Identify and document primary data source (API, open dataset, scrape)
- [ ] Download raw fixture, lineup, team history, and result data
- [ ] Store unmodified data in `data/raw/`

### Schema and processing
- [ ] Implement `data/schemas.py` with dataclasses for all tables in `docs/design.md`
- [ ] Implement `data/build_dataset.py` (raw → processed pipeline)
- [ ] Generate `match_meta`, `prematch_features`, `lottery_market` tables
- [ ] Add promotion/relegation fields (`promoted_this_season`, division level, prior strength)
- [ ] Validate: zero future-leaking columns in pre-match tables
- [ ] Add rolling statistics using only past matches (no look-ahead)
- [ ] Implement season-reset team-strength prior with promoted-team shrinkage

### Splits and loaders
- [ ] Implement `utils/split.py` with season-based split logic
- [ ] Generate train/val/test split manifests
- [ ] Refactor `data/data_loader.py` to consume processed data
- [ ] Add `config/data_config.json` and `config/feature_config.json`

### Acceptance criteria
- [ ] DataLoader produces a clean batch without errors
- [ ] Confirm zero test-season data visible in training split
- [ ] Check class balance across all target heads; document result
- [ ] Report promoted-team match counts for train/val/test splits

---

**Decision Gate 1**: If the dataset is noisy, poorly aligned, or class distribution
is degenerate across splits, fix data quality before proceeding.

---

## Phase 2 — Strong Baselines
*Goal: build robust baselines for the three lottery play types.*

### Tabular baseline
- [ ] Implement LightGBM/XGBoost pre-match baseline
- [ ] Train Fulltime 1X2 baseline
- [ ] Train HT/FT 1X2 baseline
- [ ] Train Handicap 1X2 baseline
- [ ] Record log loss, Brier score, accuracy, ECE per play type

### Neural baseline
- [ ] Implement `nn_modules/encoders/static_encoder.py`
- [ ] Implement `nn_modules/encoders/history_encoder.py` (Transformer or MLP)
- [ ] Implement `nn_modules/heads/classification_head.py`
- [ ] Wire together in `nn_modules/multitask/lottery_predictor.py`
- [ ] Implement `utils/metrics.py` (log loss, Brier, ECE, AUC)
- [ ] Implement `scripts/train_baseline.py`
- [ ] Refactor `scripts/eval.py` to produce per-play-type report

### Documentation
- [ ] Log all experiment results in `docs/experiment_log.md`
- [ ] Record best checkpoint path and metrics for each baseline

### Acceptance criteria
- [ ] Temporal model beats naive prior (uniform 1/3 or historical class frequencies)
- [ ] ECE < 0.05 on validation set (reasonably calibrated)
- [ ] Eval script produces per-play-type metric table
- [ ] Eval script reports promoted vs non-promoted slices

---

**Decision Gate 2**: If pre-match baselines are weak, investigate feature quality,
label mapping, and split design before adding model complexity.

---

## Phase 3 — Multi-Task Heads and Calibration
*Goal: improve practical usability by calibrating outputs for each play type.*

- [ ] Add three-task head support: Fulltime, HT/FT, Handicap
- [ ] Add per-head class weighting if class imbalance is severe
- [ ] Implement temperature scaling or isotonic calibration per head
- [ ] Compare pre/post calibration using ECE and Brier score
- [ ] Compare pre/post calibration for promoted-team slice specifically
- [ ] Freeze a validated checkpoint for backtest

---

## Phase 4 — Backtest and Strategy Layer
*Goal: evaluate practical value under lottery-style decisions.*


- [ ] Implement `scripts/backtest.py`
- [ ] Define selection rule (for example: only predict when confidence > threshold)
- [ ] Evaluate hit-rate and expected return under historical outcomes
- [ ] Run sensitivity analysis over confidence thresholds
- [ ] Add risk controls (max picks per issue, max exposure per day)

### Acceptance criteria
- [ ] Backtest report reproducible from one command
- [ ] Risk metrics included (drawdown, volatility proxy, hit-rate by play type)

---

**Decision Gate 3**: If no stable edge appears in backtest after calibration and
risk controls, stop and reassess data sources or problem framing.

---

## Phase 5 — Optional Complexity Upgrade (Only If Needed)
*Goal: add complexity only when justified by measured gain.*


- [ ] Add richer opponent interaction features
- [ ] Add ensemble models (GBDT + neural)
- [ ] Try league-specific calibration variants
- [ ] Add explainability output for prediction review
- [ ] **(Deferred) Tournament support**: if World Cup 2026 or similar is live, design separate
  tournament model with international strength priors; see `docs/design.md` Section 11

### Acceptance criteria
- [ ] Upgrade improves both log loss and ECE on held-out test set

---

**Decision Gate 4**: If added complexity does not improve robustness, keep the
simpler baseline system in production.

---

## Phase 6 — Evaluation and Reporting
*Goal: produce a rigorous, reproducible report by play type.*

- [ ] Implement per-play-type reporting in `scripts/eval.py`
- [ ] Run full model comparison matrix:
  - [ ] Fulltime 1X2
  - [ ] HT/FT 1X2
  - [ ] Handicap 1X2
  - [ ] Combined multi-task model
- [ ] Check for performance drift across seasons
- [ ] Check for performance drift on promoted-team fixtures across seasons
- [ ] Document all results in `docs/experiment_log.md`
- [ ] Summarise play-type performance and calibration in a table
- [ ] Summarise promoted vs non-promoted metrics in a separate table

---

## Phase 7 — Batch Inference and Issue Output
*Goal: produce actionable pre-match outputs for each lottery issue.*

- [ ] Implement pre-match inference command (issue-based batch)
- [ ] Output predictions by issue with all three play types
- [ ] Add output schema checks and sanity guards
- [ ] Add simple CLI summary (top confident picks per play type)
- [ ] Document usage in `README.md`

---

## Ongoing

- [ ] Keep `docs/experiment_log.md` updated after every training run
- [ ] Keep `data/raw/` read-only; never modify raw data in-place
- [ ] Regenerate processed dataset if any schema change is made
- [ ] Keep split manifests under version control; never regenerate mid-experiment
