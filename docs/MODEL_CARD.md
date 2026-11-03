# Model card: ParaliWatch burn classifier (v3)

Reference 123456 → R1 Punjab Ludhiana–Patiala · S4 MODIS · D5 Burn scars · T4 Multi-sensor fusion.

## Summary

A two-stage teacher → student model that flags suspected burn scars at 20 m between clear Sentinel-2 observations. This submission is a retrospective 2024 replay. The student uses the next clear observation (`b12_next`, `nbr_next`, `next_gap`), which may fall after the reported week; it has not been validated as a Monday now-cast.

- **Teacher:** logistic regression on Sentinel-2 features, trained on weak labels from MODIS active fires. Good at ranking, biased at its decision boundary.
- **Student:** gradient-boosted trees (`HistGradientBoosting`) on Sentinel-2, temporal and MODIS features, trained on photo-interpreted labels. This is the final model.

MODIS supplies the teacher's training signal, the student's fire features and the burn date; Sentinel-2 supplies field-scale geometry. Season: 2024-09-20 to 2024-11-30. AOI: 76.10–76.45 E, 30.36–30.69 N (requested box about 1,229 km², Sentinel-2 tile 43RFP). All mapped totals and evaluation samples use the snapped 1,330 km² analysis grid, which includes padding around this box.

## Inputs

| Source | Product | Use |
|--------|---------|-----|
| Sentinel-2 L2A (27 scenes) | B8A, B11, B12, B02, B03, B04 at 20 m (6 bands), SCL mask, haze mask | Features |
| MOD14A1 / MYD14A1 (20 8-day files) | Daily 1 km fire mask, Terra + Aqua | Teacher weak labels, student MODIS features, burn dating, evidence flag |
| MCD64A1 (3 monthly files) | 500 m burned area | Baseline only; never a model input |

**Masks.** SCL cloud and shadow masking as before. The v2 ARD adds a haze mask: the Haze-Optimised Transform `HOT = B02 − 0.5·B04`; pixels with HOT > 0.06 are invalid. In clear scenes 1–2% of pixels exceed it; in the smog scenes of Nov 9 and 12, 30–46% do. Sen2Cor SCL does not flag smog. The mask removed a fake week-47 spike (6,424 ha → 1,095 ha). The ARD version bump invalidated the catalogue and rebuilt all 27 scenes.

**Features** (`data/products/model.json` lists all 29):

- *Teacher (18, Sentinel-2 only):* `dnbr`, `nbr_post`, `b8a_post`, `b11_post`, `b12_post`, `d_b12`, `nbr2_post`, `d_nbr2`, `mirbi_post`, `d_b11`, `nbr_pre`, `b11_pre`, `vis_post`, `d_vis`, `gap_days`, plus 5×5 local context `dnbr_ctx5`, `d_b11_ctx5`, `b12_post_ctx5`.
- *Student adds temporal (may be NaN):* `b12_next`, `nbr_next`, `next_gap` (char persistence in the next clear view), `hot_post`, `hot_pre`.
- *Student adds MODIS:* `modis_own`, `modis_near` (fire counts in and near the pixel's 1 km cell), `modis_high` (high-confidence count), `modis_frp_max`, `modis_seen_frac` (visibility), `teacher_p`.

## Outputs

`data/products/` (20 m, EPSG:32643):

| File | Content |
|------|---------|
| `burn_season.tif` | Final (student) map, five bands (below) |
| `burn_season_teacher.tif` | Teacher map, same layout |
| `burn_season_pixel.tif` | Student map without field-object voting (identical to the final map while voting is disabled) |
| `burn_season_v1.tif` | Optional retired v1 comparison; not committed or rebuilt by the current pipeline |
| `model.json` | Teacher coefficients, thresholds, label counts, CV scores |
| `model.joblib` | Fitted student |

Bands of `burn_season.tif`:

| Band | Meaning |
|------|---------|
| `burn_day` | Day of burn as days since season start + 1; 0 = no burn. Re-dated by MODIS when a fire falls in the interval |
| `prev_day` | Previous clear Sentinel-2 observation of the pixel (start of the interval) |
| `prob_pct` | Model probability, 0–100 |
| `evidence` | 0 none, 1 Sentinel-2 only, 2 MODIS-confirmed; +10 if the interval exceeds 20 days |
| `s2_obs_day` | Day Sentinel-2 first observed the burn |

H3 feature store (`data/features`, GeoParquet partitioned by ISO week, H3 res 8, roll-up to res 7): `week`, `week_start`, `burnt_ha`, `burnt_ha_confirmed`, `burnt_ha_s2_only`, `burnt_ha_long_gap`, `mcd64_ha`, `modis_fire_cell_days`, `p_mean`, `s2_seen_frac`, `s2_clear_scenes`, `blind_spot`. `blind_spot` = MODIS fire cell-days × (1 − Sentinel-2 clear fraction).

## Method

1. **Candidates.** Stream scenes in date order, keeping each pixel's last clear observation. A clear-to-clear transition is a candidate if NBR drops by more than 0.10 **or** SWIR-1 (B11) darkens by more than 0.06 (stubble → char barely moves NBR). 11.54 M candidate transitions. An endpoint-change diagnostic compares interval endpoints on the test set; it is not a recall ceiling, because fusion gates consecutive clear views.
2. **Teacher.** Weak labels (MODIS weak labels from sealed-test 1 km cells are excluded from fitting the teacher and its threshold): positive = MODIS fire (class ≥ 7) in the pixel's own 1 km cell during the interval; negative = no fire within ±2 cells and ±2 days, and the cell was observed (cloud is never a negative). Balanced logistic regression (up to 100,000 per class) on the 18 Sentinel-2 features.
3. **Gold labels.** Photo-interpreted transitions (below), matched to candidates on the exact (pixel, pre, post) triple. Recycled test-A labels map to a transition only when exactly one transition lies wholly inside the labelled interval. Round 3 is active learning (`fusion.active_sample`): 240 transitions closest to the student threshold in logit space (at most one per 1 km cell) plus 80 random mapped-burn audit points.
4. **Student.** Gradient-boosted trees on teacher + temporal + MODIS features. Because the gold labels no longer come from MODIS, MODIS can enter as features without leaking its own label.
5. **Threshold.** The sample over-represents teacher-score bins differently from the population, so each label carries a design weight (teacher-score bin share ÷ labels in that bin). The student threshold (0.25) maximises population-weighted F1 out of fold. Active-learning labels are not a probability sample, so they train only outside the held-out spatial blocks and are never scored: OOF F1, the threshold and the self-training gate use the 600 probability-sampled labels only (design weight 0 for AL rows). The teacher threshold is Neyman–Pearson (5% false-alarm cap on MODIS-quiet changes).
6. **Validation.** Spatially nested cross-validation over 3 km blocks, with the teacher refit inside each fold. Pseudo-labels are re-derived per fold with the fold teacher and kept only where it agrees.
7. **Self-training.** Up to 500 pseudo-labels per class at weight 0.1, taken where the teacher is confident and MODIS agrees, excluded at 1 km-cell level from gold-label and sealed-test cells (the sealed sets now touch all 156 3 km blocks, so block-level exclusion would leave nothing). Trialled with 510 pseudo-labels: nested-CV F1 0.581 < 0.592 without, so the gate rejected it (v2 had adopted it).
8. **Attribution.** A pixel's burn is its first transition above threshold. If a MODIS fire lies in the interval, the burn is dated to the last such fire and flagged confirmed; otherwise the Sentinel-2 date is kept.

## Training data

| Source | Count |
|--------|-------|
| Teacher spatial-CV AUC against weak labels | 0.746 ± 0.029 (agreement with MODIS, not map accuracy) |
| Gold labels, round 1 | 222 |
| Gold labels, round 2 (haze-free scene pairs only) | 282 |
| Gold labels recycled from the retired test set A (unambiguous transitions only) | 96 |
| Gold labels, round 3 (active learning; trained on, never scored) | 262 |
| **Gold total** | **862 transitions, 189 burns (600 scored)** |

Any gold label, weak label, pseudo-label or active-learning sample inside a 1 km cell that holds a sealed test point (test set 1 or holdout B: 570 of 1,330 cells) was excluded. Labels are from AI interpreters (see Caveat). Student nested spatial-CV (3 km blocks, teacher refit per fold, scored rows only): AUC 0.899, population-weighted F1 0.592 (precision 0.587, recall 0.598), threshold 0.25. History: v2 0.559 at threshold 0.23; with holdout B sealed but before active learning 0.573.

## Metrics

**Season map:** **9,722 ha** burnt, 9.2% MODIS-confirmed, median MODIS re-dating 1 day. (v1: 18,133 ha; teacher map 24,712 ha.)

| Week | W38 | W39 | W40 | W41 | W42 | W43 | W44 | W45 | W46 | W47 | W48 |
|------|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|----:|
| Confirmed ha | 0 | 0 | 0 | 3 | 17 | 107 | 564 | 25 | 52 | 104 | 18 |
| S2-only ha | 0 | 84 | 115 | 928 | 1395 | 1536 | 1374 | 634 | 130 | 1268 | 1368 |
| MCD64A1 ha | 0 | 0 | 43 | 1223 | 451 | 237 | 386 | 43 | 21 | 0 | 0 |
| MODIS fire cell-days | 0 | 0 | 0 | 2 | 5 | 19 | 57 | 25 | 5 | 13 | 2 |

**Late-season caveat.** The W47–W48 Sentinel-2-only area (about 1,300 ha/week) coincides with the weakest holdout-B interval (late November, F1 0.08). It is likely partly commission from tilled or emerging-wheat fields; treat late-season mapped area as potentially inflated, not as a statistical upper bound.

**Holdout B, the headline (`evaluation.json`, key `holdout_b`).** Drawn once from the v2 maps (`reference/testB/`, seed and date in `sample_meta.json`) and sealed before any further model development; first scored during final v3 evaluation, according to the development record. Reproduction runs re-score this frozen reference set without tuning the model. 552 points on three new intervals: 2024-10-13→10-23 (early), 2024-10-30→11-07 (peak), 2024-11-19→11-29 (late). Per interval: mapped burn 60 / teacher-only burn 40 / other change 60 / stable 24 (other change is over-allocated because missed burns there dominate the recall interval). 1 km cells are disjoint from every training point and from earlier test sets; strata and population counts were frozen at draw time. Interpreters A and B (Cohen's κ 0.551 on 427 points where both were confident) plus blind adjudicator C for 158 disputed or unclear points (155 resolved); 549 points used, 67 burns. Stratum-weighted (Olofsson et al. 2014); estimate [95% bootstrap CI].

| Map | Precision | Recall | F1 |
|-----|-----------|--------|----|
| **v3 student** | 0.272 [0.169, 0.403] | 0.146 [0.082, 0.299] | **0.190** [0.117, 0.306] |
| teacher | 0.109 | 0.068 | 0.084 [0.040, 0.151] |
| v1 | 0.340 | 0.120 | 0.178 [0.021, 0.425] |
| MODIS active fire | 0.005 | 0.006 | 0.005 |
| Sentinel-2 dNBR | 0.006 | 0.016 | 0.008 |
| MCD64A1 | 0 | 0 | 0.0 |

Student overall accuracy 0.980.

Metrics apply to the sampled clear-date intervals and eligible spatial cells, not to every pixel and week of the season. Bootstrap intervals describe resampling of the reference points; they do not include uncertainty in the AI interpreters' judgements.

*Unanimous points only (A = B; 394 points, 24 burns):* student F1 **0.378** [0.239, 0.510] (precision 0.26, recall 0.72); teacher 0.116; v1 0.015.

*Dispute sensitivity:* all disputes resolved as A 0.374 [0.245, 0.489]; as B 0.215 [0.130, 0.379]. *By interval (adjudicated):* early 0.322 [0.148, 0.625], peak 0.182 [0.054, 0.458], late November 0.083 [0.039, 0.194].

**Error structure.** Omission dominates (weighted omission 1.38% of area vs commission 0.63%). It concentrates in the "other change" stratum, where each burn point carries about 0.3% of the area, especially in the late interval (6 of 60 points burnt, mostly adjudicator-sourced). The late interval is also where A and B split: A read dull grey-green patches after tillage as emerging wheat, B as char. Adjudicator C calls burns on 28% of disputed points vs 6% on A=B points, so the headline is sensitive to the adjudicator; the adjudicated and unanimous-subset figures describe different reference cohorts, not bounds on the true F1.

**Development set (test set 1).** 360 points, two intervals (2024-10-25→11-04, 2024-11-07→11-19), 70/50/40/20 per interval, κ 0.866. It was scored four times during development, so it is a development set, not a holdout. v3 student F1 0.339 [0.149, 0.624] (precision 0.473, recall 0.264), unanimous 0.427 [0.265, 0.575]; teacher 0.126; v1 0.084. (v2 history: 0.280 adjudicated, 0.258 unanimous.) The drop to 0.19 on holdout B is the expected price of reuse plus harder new intervals. Holdout B's interval is also tighter (width 0.19 vs 0.46–0.48), which was the point of drawing it.

**Leakage check and known approximations.** A Codex (`gpt-6-astra`) verification of v3 found no holdout-B label leakage: holdout-B locations only feed exclusion masks, no holdout-B cell overlaps training, test A or test set 1, and the frozen v2 strata remain valid for scoring v3 because the stored stratum sizes N_h are used. Known approximations, not fixed:

- Design weights re-bin the scored labels by the retrained teacher's score, so they approximate (not reproduce) inclusion probabilities across cohorts with different designs (round 2 haze-free only; test-A agreement filter). OOF F1 and the threshold may be biased.
- The active-learning batch was chosen by a model fit on all labels, and the threshold maximises the same OOF F1 that is reported, so the student CV F1 (0.592) is mildly optimistic. Holdout B is the held-out estimate against AI-interpreted reference labels.
- The A/B dispute-sensitivity analysis on test set 1 uses a slightly different cohort than its adjudicated figure.

**Receipt scope.** `reference/testB/scoring_receipt.json` stores a legacy hash of rounded model diagnostics, excluding the build timestamp. `diagnostics_match_receipt` only reports whether those diagnostics match. The receipt does not hash fitted weights, predictions, samples or labels, and cannot enforce single use. The holdout history is a development-record claim, not a guarantee supplied by this file. The evaluator now also writes `unanimous_reference` metrics, so the published unanimous-subset comparison can be reproduced directly.

**Overall accuracy is not informative.** On holdout B it is 0.976–0.984 for the student, teacher, v1 and MCD64A1 because most of the area is unburnt (MCD64A1 flags only 2 of the 549 reference points and scores 0.984). Use precision, recall and F1.

**Interpretation.** The adjudicated holdout F1 is 0.190 [0.117, 0.306]. The unanimous-subset score of 0.378 uses fewer, easier reference points and is not an upper confidence bound. The student improves on the teacher, but omissions and uncertain labels prevent an operational accuracy claim.

**Caveat on the reference.** All interpreters are AI vision models (Claude Sonnet sub-agents for A and B, Claude Opus for C) reading 20 m SWIR/visible chip sheets, blind to strata and model output. A first interpreter run was rejected for defaulting unlabelled points to 0. These labels are a transparent, reproducible stand-in, not field truth. `paraliwatch label-page reference/testB` writes a self-contained `label.html`: the same four chips per point at native 20 m pixels, keys 1/0/9, notes, browser autosave, and an export to `labels_<name>.csv` in the pipeline's format, so a human set slots in as interpreter A, B or C (no model output is embedded). `paraliwatch evaluate` then recomputes everything. `reference/scored_points.csv` has one row per point with `ref_source` (`A=B` unanimous, `C` adjudicated).

## Intended use

- Triage: rank village clusters for verification visits and awareness or CRM-machinery outreach, with the evidence flag shown.
- Seasonal and weekly area totals as an order-of-magnitude indicator, with the uncertainty above.

## Out of scope

- **Never use as evidence to fine an individual farmer.** Precision is about 0.27 and the unit is a pixel, not a legal field.
- Not validated for other seasons, regions or crops.
- Not a measure of emissions or air-quality attribution.
- Not a replacement for field verification or official fire reports.

## Limitations and failure modes

- **Retrospective timing.** Next-clear-view features and full-season model fitting can revise earlier weeks. A strict information cutoff and a separate as-of-date evaluation are required for a live service.
- **Low recall.** Holdout B recall is about 0.15 (unanimous subset 0.72 at precision 0.26); many burns are missed, mostly in the "other change" stratum.
- **Small, AI-made label set.** 189 gold burn labels and AI-interpreted test sets; holdout B κ was only 0.55, and adjudicated F1 CI is 0.12–0.31.
- **Late-season commission.** Late-November tilled or emerging-wheat fields; W47–W48 mapped area may be inflated.
- **Tillage confusion.** Farmers till within days of burning, so MODIS-fire cells contain tillage; the teacher learned "large NBR drop". The student reduces this but does not remove it.
- **Smog gaps.** Sentinel-2 was blind about Nov 9–17. Burns in the gap surface on the first clear scene (Nov 19). Confirmed ones are re-dated by MODIS; Sentinel-2-only ones cannot be. HOT masks hazy pixels but cannot recover them.
- **Burn-then-till inside a gap.** The scar may be gone before Sentinel-2 returns.
- **Field edges and mixed pixels**, villages, roads, shadow and water margins can produce spectral drops.
- **MODIS end-of-life.** Terra and Aqua are ageing; VIIRS 375 m is the successor. MODIS also misses late-afternoon burning and its 1 km pixels are far larger than 0.5–2 ha fields.
- **Stable strata not enlarged.** One part of the Codex review was not done: enlarging the stable strata, because the stratification map had already been superseded.

## Tried and rejected

- **Self-training (v3).** Rejected by its own gate: 0.581 vs 0.592 without.
- **Field-object voting** (`objects.py`, disabled via `fusion.field_voting: false`). Segments are about 50 px but burn pixels cover about 5% of a segment, so voting erased scars.
- **Char-albedo gate** (v1, post B12 < 0.18). F1 0.10 → 0.145 on the tuning interval, 0.061 → 0.066 on the other one.
- **0.5 probability threshold** (v1). Mapped about 80% of the AOI as burnt.
