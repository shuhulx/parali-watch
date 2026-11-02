# AI usage log — full detail (appendix to AI_LOG.md)

## Assistants used

| Tool | Role |
|------|------|
| Claude Code, Claude Opus (main agent) | Architecture, most of the code, running the pipeline, reading results |
| Claude Sonnet sub-agents | Drafts of README, ARCHITECTURE, MODEL_CARD and the report; the photo-interpreters (below) |
| Claude Opus sub-agents | Interpreter C: blind adjudication of 70 disputed or unclear test-set-1 points (63 resolved) and, in v3, of 158 holdout-B points (2 agents, 155 resolved) |
| OpenAI Codex (Codex plugin) | Independent code audit of v1; `gpt-6-astra` review of the pipeline; wrote `objects.py` |

## Codex audit (v1)

Found 13 issues, 3 of them crashes and the rest wrong results. All were fixed.

- Crashes: int16 overflow in MCD64 date conversion; empty-season handling; JSON date serialisation.
- Wrong results: the excluded pre-day was counted in MODIS windows; stale weekly partitions were kept; superseded Sentinel-2 reprocessings were still used.

## Codex `gpt-6-astra` review (v2)

Ten recommendations, all implemented:

1. More gold labels (round 2 plus labels recycled from the retired test set A).
2. Exact (pixel, pre, post) transition matching of labels.
3. Population-weighted threshold selection.
4. Nested teacher + student cross-validation over 3 km blocks.
5. Measure candidate-gate behaviour as an endpoint-change diagnostic (not a recall ceiling).
6. Temporal and haze features (next-view char persistence, HOT pre/post).
7. Local 5x5 context features instead of object voting.
8. MODIS confidence and FRP as student features.
9. Blind third-interpreter adjudication (C, Opus).
10. Self-training, deferred first and then trialled; adopted only because nested-CV F1 rose 0.548 to 0.559.

A final Codex verification then found location-independence issues, all fixed: MODIS weak labels from sealed-test cells fitted the teacher and its threshold; self-training pseudo-labels could come from sealed-test blocks; nested-CV pseudo-labels carried the full-data teacher's labels (now re-derived per fold); recycled test-A labels were mapped to the first candidate transition even if ambiguous (test-A labels used fell 149 to 96). Two items stay as known approximations: design weights are a teacher-score-bin approximation across cohorts with different sampling designs, and the A/B dispute-sensitivity analysis uses a slightly different cohort than the adjudicated headline. The sealed test set was scored three times during development; nothing was tuned on it, but repeated scoring is a mild form of selection. Corrected v2 F1 on that set was 0.280 (adjudicated), down from the pre-fix unanimous 0.424. In v3 that set was scored a fourth time and is now reported only as the development set (see below).

One part was not done: enlarging the stable strata. The stratification map it referred to had already been superseded.

Codex also wrote `objects.py` (field segmentation and voting). It was tried and rejected, so it is disabled by config: segments are about 50 px but burn pixels cover about 5% of a segment, so voting erased scars.

## AI photo-interpreters

All reference and training labels come from AI vision models reading 20 m chip sheets, blind to strata and model output.

| Set | Agents |
|-----|--------|
| Test set 1 / development set (360 points) | 6 Sonnet agents (interpreters A and B); Opus adjudicator C |
| Holdout B (552 points, v3) | 6 Sonnet agents (A forward order, B reverse); 2 Opus agents as adjudicator C |
| Training labels | 6 Sonnet agents (round 1), 4 Sonnet agents (round 2, haze-free pairs only), 4 Sonnet agents (round 3, active-learning batch) |

## v3: holdout B and active learning

- **Holdout B.** Because test set 1 had been scored repeatedly, a new set was drawn from the v2 maps and sealed before any further model work, then scored once on the final model. 6 Sonnet agents labelled it blind as interpreters A and B; 2 Opus agents adjudicated 158 disputed or unclear points (155 resolved).
- **Adjudicator prompt.** One key line was added relative to test set 1: "Late-November fields may show dull grey-green emerging wheat — that is crop (green in SWIR), not char."
- **Agreement was poor.** Cohen's κ between A and B was 0.55 on holdout B (0.87 on test set 1). In late November A read dull grey-green tilled patches as emerging wheat and B as char. The adjudicator then called burns on 28% of disputed points vs 6% on unanimous ones, so the headline moves with the adjudicator; both the adjudicated (0.190) and unanimous (0.378) figures and the A/B bracket (0.215–0.374) are reported.
- **Active learning.** 4 Sonnet agents labelled the round-3 batch (262 usable). These rows train the student but are never scored, because they are not a probability sample.
- **Self-training rejected.** The same gate that adopted it in v2 rejected it in v3 (0.581 vs 0.592).
- **Label desk.** `paraliwatch label-page` exists so a human can replace any AI interpreter.
- **Codex verification of v3.** Found no holdout-B label leakage (holdout-B locations only feed exclusion masks; no cell overlap with training, test A or test set 1). It documented two caveats: design weights only approximate inclusion probabilities across cohorts, and the student CV F1 (0.592) is mildly optimistic because the AL batch and threshold use the same labels. "Scored once" is now enforced by `reference/testB/scoring_receipt.json`.

## What the human did

- Chose to keep the use-case and user framing (district nodal officer, Monday dispatch list).
- Reviewed design choices.
<!-- Fill in: what you personally wrote, checked or changed -->

## Judgment calls made in the loop

- **Rejected the 0.5 threshold.** It mapped about 80% of the AOI as burnt. Replaced with a Neyman–Pearson threshold (5% false-alarm cap on MODIS-quiet changes).
- **Rejected the first interpreter run.** It defaulted unlabelled points to 0. The run was redone.
- **Did not adopt the char-albedo gate (v1).** It improved F1 on the interval it was tuned on (0.10 to 0.145) but not on the held-out interval (0.061 to 0.066). Reported as a negative result.
- **Disabled field-object voting.** It erased scars (see above).
- **Added a haze mask** after finding Sen2Cor SCL does not flag smog; it removed a fake week-47 spike (6,424 ha to 1,095 ha).
- **Reported both reference variants** (adjudicated and unanimous) instead of picking the better one.

## Limits of the AI-made reference labels

The interpreters were AI vision models, not humans. Earlier prompts told them to expect 25–50% burns, a prior that could bias labels; they found far fewer. On holdout B the two interpreter sets agreed only moderately (κ 0.55). Treat the labels as a stand-in, not ground truth. The numbers in this repository carry that limit.
