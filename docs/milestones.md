# Milestones

Tracked by phase. Mark items `[x]` when done. Stop at each **Decision Gate** and
evaluate before proceeding to the next phase.

---

## Phase 0 — Lock the Product Scope
*Goal: fix lottery products, labels, and pre-match-only boundaries before coding.*

- [x] Confirm football-only scope
- [x] Confirm play type order: Fulltime 1X2 -> HT/FT 1X2 -> Handicap 1X2
- [x] Confirm prediction timing: once before kick-off
- [x] Write down pre-match information boundary and examples of forbidden fields
- [x] Confirm league and season scope for v1 (expanded to 10 leagues, 5 countries, 6 seasons)
- [x] Tournament matches out of scope (deferred to Phase 5+, see Section 11)
- [x] Define promotion/relegation policy for new season transition
- [x] Document all decisions in `docs/design.md` Section 3 (Scope)
- [x] Fix final metric set and acceptance thresholds (log loss, Brier, accuracy, ECE)

---

## Phase 1 — Build Pre-Match Dataset Backbone
*Goal: replace synthetic tensors with a real, leakage-free pre-match dataset.*

### Data acquisition
- [x] Identify and document primary data source (football-data.co.uk via `data/build_dataset.py`)
- [x] Download raw fixture, team history, and result data
- [x] Store unmodified data in `data/raw/`

### Schema and processing
- [x] Implement `data/build_dataset.py` (raw → processed pipeline)
- [x] Generate `match_meta.jsonl` and `prematch_features.jsonl` tables
- [x] Validate: zero future-leaking columns in pre-match tables
- [x] Add rolling statistics using only past matches (no look-ahead)
- [ ] Add promotion/relegation fields (`promoted_this_season`, division level, prior strength) — **deferred**
- [ ] Implement season-reset team-strength prior with promoted-team shrinkage — **deferred**

### Splits and loaders
- [x] Season-based split logic (train: seasons 1–N-2, val: N-1, test: N)
- [x] Refactor `data/data_loader.py` to consume processed data
- [x] Add `config/data_config.json`
- [ ] `config/feature_config.json` — not yet created

### Acceptance criteria
- [x] DataLoader produces a clean batch without errors
- [x] Confirm zero test-season data visible in training split
- [ ] Check class balance across all target heads; document result — **pending**
- [ ] Report promoted-team match counts for train/val/test splits — **deferred**

---

**Decision Gate 1**: If the dataset is noisy, poorly aligned, or class distribution
is degenerate across splits, fix data quality before proceeding.

---

## Phase 2 — Strong Baselines
*Goal: build robust baselines for the three lottery play types.*

### Tabular baseline
- [ ] Implement LightGBM/XGBoost pre-match baseline — **not yet started**

### Neural baseline (match-token embedding architecture)
- [x] Implement match-token embedding encoder (`nn_modules/encoder/top_encoder.py`)
- [x] Implement transformer backbone (`nn_modules/transformer/top_former.py`)
- [x] Implement decoder / classification head (`nn_modules/decoder/top_decoder.py`)
- [x] Implement `scripts/train.py`, `scripts/eval.py`
- [x] Implement `scripts/autoresearch_local.py` for hyperparameter sweep
- [x] Train all three play types: Fulltime 1X2, HT/FT 1X2, Handicap 1X2

### Best measured metrics (autoresearch sweep, lr ∈ {0.0001,0.0002,0.0003}, epochs=20)

| Task | Val Log-Loss | Val Accuracy | Test Log-Loss | Test Accuracy | Val ECE |
|------|-------------|-------------|--------------|--------------|--------|
| Handicap 1X2 | 0.8784 | 48.3% | 0.8837 | 48.5% | 0.0032 |
| Fulltime 1X2 | 1.0777 | 43.0% | 1.0763 | 43.3% | 0.0086 |
| HT/FT 1X2 | 1.9747 | 24.9% | 1.9883 | 25.6% | 0.0040 |

*Note: LR sweep had minimal effect — all runs converge to nearly identical scores.
Model is well-calibrated (ECE < 0.01) but accuracy is near-random-baseline,
indicating feature representation is the primary bottleneck.*

### Acceptance criteria
- [x] ECE < 0.05 on validation set ✅ (ECE < 0.01)
- [ ] Model beats naive prior on accuracy — **marginal** (43–48% vs. ~33% for fulltime, ~11% for htft)
- [ ] Eval script reports promoted vs non-promoted slices — **not yet implemented**
- [ ] Log all experiment results in `docs/experiment_log.md` — **not yet done**

---

**Decision Gate 2**: If pre-match baselines are weak, investigate feature quality,
label mapping, and split design before adding model complexity.

---

## Phase 3 — Multi-Task Heads and Calibration
*Goal: improve practical usability by calibrating outputs for each play type.*

- [x] Add three-task head support: Fulltime, HT/FT, Handicap (trained separately via `--task` flag)
- [ ] Add per-head class weighting if class imbalance is severe — **pending class balance audit**
- [ ] Implement temperature scaling or isotonic calibration per head — **not yet done**
- [ ] Compare pre/post calibration using ECE and Brier score
- [ ] Compare pre/post calibration for promoted-team slice specifically
- [ ] Freeze a validated checkpoint for backtest

*Current ECE is already low (< 0.01) — but this may be due to limited feature diversity
rather than well-tuned calibration. Temperature scaling should still be applied before backtest.*

---

## Phase 4 — Backtest and Strategy Layer
*Goal: evaluate practical value under lottery-style decisions.*

- [x] Implement `scripts/backtest_ev.py` (EV backtest framework built)
- [x] Implement `scripts/predict.py` (per-match inference)
- [x] Implement `scripts/enrich_lottery_odds.py` (fuzzy-match external odds to match_meta)
- [ ] Feed **real** China Lottery odds into pipeline — **blocked: no real odds data yet**
- [ ] Define selection rule (only predict when confidence > threshold)
- [ ] Evaluate hit-rate and expected return under historical outcomes
- [ ] Run sensitivity analysis over confidence thresholds
- [ ] Add risk controls (max picks per issue, max exposure per day)
- [ ] Validation-split EV backtest (to detect overfitting before touching test set)

### Acceptance criteria
- [ ] Backtest report reproducible from one command
- [ ] Risk metrics included (drawdown, volatility proxy, hit-rate by play type)
- [ ] Positive EV on validation set at some confidence threshold

---

**Decision Gate 3**: If no stable edge appears in backtest after calibration and
risk controls, stop and reassess data sources or problem framing.

---

## Phase 5 — Feature Engineering & Model Upgrade
*Goal: break accuracy ceiling by adding domain-relevant signals. Previous name:
"Optional Complexity Upgrade" — now promoted to a required phase given baseline
accuracy is near-random.*

### Sprint A — Recent Form Features (highest priority)
- [ ] Rolling goals scored/conceded (last 5, last 10 matches) per team
- [ ] Rolling win/draw/loss rate (last 5, last 10) per team
- [ ] Home/away performance split features
- [ ] Goal difference trend (momentum proxy)
- [ ] Weighted recency form score

### Sprint B — Head-to-Head & Context Features
- [ ] H2H result record (last 5 meetings between the two teams)
- [ ] League position / points gap at match date
- [ ] Rest days gap (home rest days − away rest days)
- [ ] Schedule congestion (matches in last 14 days per team)

### Sprint C — Market Signals (if real odds available)
- [ ] Market-implied probabilities from closing odds as input features
- [ ] Log-odds transformation for each outcome
- [ ] Odds movement signal (closing − opening odds)

### Sprint D — Architecture Experiments (if features plateau)
- [ ] LightGBM/XGBoost tabular baseline for comparison
- [ ] Deeper transformer encoder (4–6 layers)
- [ ] Ensemble: neural output stacked with GBDT
- [ ] League-specific calibration variants

- [ ] **(Deferred) Tournament support**: World Cup 2026 or similar — separate model
  needed; see `docs/design.md` Section 11

### Acceptance criteria
- [ ] Val accuracy ≥ 50% for Fulltime 1X2 (currently 43%)
- [ ] Val log-loss improvement ≥ 0.03 vs. current baseline
- [ ] Upgrade improves both log loss and ECE on held-out test set

---

**Decision Gate 4**: If feature engineering does not move accuracy beyond 50%,
consider market-odds-as-features as primary signal or reassess dataset coverage.

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
