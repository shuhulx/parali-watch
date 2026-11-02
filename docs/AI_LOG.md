# AI usage log

**Assistants.** Claude Code as a pair programmer (design discussion, code, running the pipeline); OpenAI Codex as an independent code reviewer; Claude vision models as blind photo-interpreters for reference and training labels.

**How I used them.** I set the use case, the user and the decision the product serves, and asked the assistant to propose and justify each design step: STAC catalogue, haze mask, teacher → student fusion, design-based evaluation. A choice stayed only if it held up against the data. The rejected ones: a 0.5 threshold (mapped ~80% of the area as burnt), field-object voting (erased real scars), a char-albedo gate (gains did not transfer to a held-out interval), and self-training in the final model (failed its own cross-validation gate). I used the second assistant mainly as an auditor (it also wrote the field-voting module, later disabled). It found real bugs: an integer overflow, off-by-one MODIS date windows, and leakage between training and test areas. All were fixed.

**Labels.** All reference and training labels come from AI interpreters reading Sentinel-2 chips, blind to strata and model output and double-labelled with an agreement statistic (κ 0.87 on the development set, 0.55 on holdout B). This is the main limit of the evaluation; `paraliwatch label-page` lets a human replace any interpreter.

**Integrity calls.** After noticing the first test set had been reused during development, I sealed a new holdout before further tuning and report its lower score (F1 0.19 adjudicated, 0.38 unanimous) rather than the development figure (0.34).

**What I did myself.** I chose the use case, the user and the decision the product serves, required a full 2024 season, pushed for an honest accuracy figure over a flattering one, and reviewed each result before it went into the report.
