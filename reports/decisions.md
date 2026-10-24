# Shared experiment decisions

Status: decisions are open. Member 2 maintains this record with Members 1 and 3-5. Update the corresponding config values only when a decision is agreed.

| Decision | Status | Evidence and agreed rule |
| --- | --- | --- |
| Prediction/analysis point for each task | Open | Explain which inputs are available then and dataset snapshot limitations. |
| ADR meaning and regression population | Open | Address canceled bookings, zero nights, zero ADR, and unusual rates. |
| Duplicate handling | Open | Distinguish identical recorded bookings from confirmed accidental duplicates. |
| Invalid-row handling | Open | Record the reason and affected count for every proposed exclusion. |
| Missing/not-applicable values | Open | Separate unknown values from Agent/Company not-applicable categories. |
| Supervised split | Open | Record method, parameters, seed/cutoffs, overlap checks, and test policy. |
| Feature lists and engineering | Open | Justify every task's inputs at the selected time. |
| Baseline models and configurations | Open | Select one untuned model/algorithm for each task. |
| Clustering population and sampling | Open | State segmentation purpose and any sampling method. |

Record who reviewed a decision and when. Increment the feature-policy or split version when changing an agreed rule; rerun affected experiments. No experiment settings have been approved by this template.
