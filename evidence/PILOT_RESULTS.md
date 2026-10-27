# First real-data pilot — 2026-10-27

This is a pipeline verification experiment, not a hidden-test estimate.
No hidden targets or final contest predictions were available.

## Scope

- Official Yambda-50M likes and artist mappings, revision in `dataset_manifest.json`.
- Cohort file is the intersection of supplied test uids and users with any likes in 50M:
  245 users. This is future-activity-selected at the whole-log level and not representative
  of the full supplied cohort. 150 were sampled for evaluation; seed 42.
- Train cutoff 13,000,000, validation cutoff 17,000,000, audit cutoff 21,000,000.
- Targets: next 100 first-ever unique liked items, constrained by available observation.
  Training labels end before validation; validation labels end before audit; audit labels
  use the rest of the public log. These differently censored periods require caution.
- 150 sampled training users; 5,000 CF users; top 5,000 CF items; 100 per source;
  capped union 500 per user. No listen file, ALS or audio in this real-data run.
- CatBoost selection: 50 vs 100 iterations; validation chose 100.

## Results

| Method | Recall@100, all 150 users | Recall@100, 149 active users |
| --- | ---: | ---: |
| Historical popularity | 0.034533 | 0.034764 |
| Decayed recent popularity | 0.047616 | 0.047936 |
| Reciprocal rank fusion | 0.055781 | 0.056156 |
| CatBoost on union candidates | 0.062753 | 0.063174 |
| Candidate oracle ceiling | — | 0.117331 |

CatBoost minus RRF, paired active-user delta: **+0.007019**, bootstrap 95% interval
**[+0.001761, +0.012334]**. This is within-cohort uncertainty, not temporal generalization
uncertainty. It does not resolve the hidden task ambiguities.

Only **19 / 150** users have 100 observed future target items. Their CatBoost recall is
**0.057895**. Among active users, median observed target count is 33. A user with a short
observed target prefix may still accumulate many more likes later, so we cannot call that
prefix the complete future truth. See `pilot_censoring.json` for the distribution.

The final instrumented rerun took about **18.9 seconds** and the Linux child process
reported maximum resident memory about **696 MiB** on this hosted environment. This excludes
download/installation and is not a laptop runtime promise or a full-5B resource estimate.

## Interpretation

The pipeline runs and the tabular model improves over its simple fusion baseline on this
small split. Fresh popularity beats whole-history popularity here. Both conclusions are
local to the stated experiment. Candidate retrieval still misses most observed positives:
improving data coverage, personalized retrieval and listening features has higher priority
than a large CatBoost hyperparameter sweep.

The global 50M cohort-coverage result is decisive: only **245 / 26,021 (0.942%)** supplied
users have like history there. Moving to full likes is necessary for broad personalization.
Being absent from likes does not prove a user has no listens in that dataset; this pilot
only downloaded likes. Full-data listen history remains a separate opportunity.

Source diagnostics in `pilot_report.json` refer to candidates retained after fusion/capping.
The leave-one-source-out diagnostics remove source-exclusive retained candidates without
regenerating or backfilling the union; use them as marginal coverage diagnostics, not as
full retrained source ablations.

Six automated tests pass, including official metric agreement, first-like boundaries,
future-event invariance with UInt32 source types, multi-batch feature schema consistency,
end-to-end model/export, optional CPU ALS/audio paths, and listen timestamp/completion
filtering. Several assertions are grouped within the six test functions. ALS/audio were
tested on synthetic fixtures only; their full-catalog speed and real-data gains are unknown.
