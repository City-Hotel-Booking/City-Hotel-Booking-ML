# Project reports

This directory contains data findings, experiment policies, task reports
and the integrated Milestone 1 progress report.

## Report ownership

| File | Owner | Purpose |
| --- | --- | --- |
| `eda_notes.md` | Zeina | Data understanding, summaries and EDA figures |
| `data_review.md` | Zeina | Unusual records, duplicate evidence and interpretation |
| `member1_integration.md` | Zeina, with integration updates | Source identity and data-interface compatibility |
| `decisions.md` | Shared policy maintained through integration | Starting assumptions and rules for experiment changes |
| `leakage_audit_m1.md` | Michael, with task-owner contributions | Leakage risks and mitigations across all three tasks |
| `handoff_member2.md` | Michael, with integration updates | Shared preparation interfaces and model deliverables |
| `regression_m1.md` | Paula | Regression experiment and interpretation |
| `classification_m1.md` | Youssef | Classification experiment and interpretation |
| `clustering_m1.md` | Aya | Clustering experiment and interpretation |
| `milestone1_report.md` | Makarious, checked by all owners | Integrated progress report |

## Current work

Data validation, EDA and shared preparation evidence are available.
The model reports remain templates until their owners complete actual
baseline experiments.

The integrated report is assembled progressively. It must distinguish
completed work, starting assumptions, experiment results and remaining work.

## Experiment reports

Each model owner records:

1. Purpose, prediction or analysis point and population.
2. Algorithm and fixed baseline settings.
3. Features, preprocessing, source checksum and relevant versions.
4. Actual results with links to saved metrics, figures and tables.
5. Interpretation and limitations.
6. Departures from the starting policy and the reasons for them.
7. Exact reproduction command and actual contribution.
8. Planned Milestone 2 work.

Follow `decisions.md` for the starting experiment policy.
Use the result format described in `../outputs/README.md`.

## Generated reports

Running `main.py analyze` regenerates `eda_notes.md` and `data_review.md`.
Changes to generated wording must also be made in the generating code,
otherwise a later analysis run will overwrite them.

The current experiment policy is maintained in `decisions.md`.
Historical observations in generated evidence must not be presented as
new model-performance results.

## Milestone 1 assembly

The integrated report includes:

- Verified dataset understanding and preparation.
- Experimental setup and starting assumptions.
- Initial regression, classification and clustering results.
- A substantive preliminary leakage audit.
- Actual member contributions.
- Remaining work and the Milestone 2 plan.
- Tested environment and reproduction instructions.

Each owner checks their contribution and interpretation before submission.