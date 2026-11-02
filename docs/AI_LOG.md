# AI usage log

**Assistants.** Claude Code (Opus) as the main agent: architecture, most of the code, running the pipeline, reading results. Claude Sonnet sub-agents drafted documentation and acted as blind photo-interpreters; Claude Opus sub-agents adjudicated disputed reference points. OpenAI Codex (`gpt-6-astra`) did three independent audits and wrote `objects.py` (field voting, tried and disabled).

**How they were used.** I prompted for the pipeline design (STAC catalogue → ARD → MODIS-taught teacher → student → H3 feature store → static dashboard), for code, and for audits. Codex audits found real bugs: crashes (int16 overflow, empty season), wrong results (off-by-one MODIS windows, stale partitions), and location leakage between training and test cells. All were fixed and are listed in the detail log.

**AI-made labels — the main limit.** Every training and reference label comes from AI vision models reading Sentinel-2 chips, blind to strata and model output: two interpreters plus an adjudicator per test set. Agreement was κ 0.87 on test set 1 but only 0.55 on holdout B, so labels are a stand-in for field truth. `paraliwatch label-page` exists so a human can replace any AI interpreter.

**Judgement calls kept in the loop.** Rejected a 0.5 threshold that mapped ~80% of the area as burnt; rejected an interpreter run that defaulted unknowns to 0; added a haze mask after finding Sen2Cor misses smog; disabled field voting when it erased scars; sealed holdout B before further modelling and reported both adjudicated and unanimous scores rather than the better one.

**What I did myself.** <!-- Fill in, one or two sentences: what you personally wrote, checked or changed -->

Full log: [AI_LOG_DETAIL.md](AI_LOG_DETAIL.md).
