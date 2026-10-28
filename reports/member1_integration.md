# Member 1 integration

Zeina's validation and EDA contribution is adapted to the shared repository at base commit `d0875ff4d4bd64917ad817dacb61d3ee21584037`. The local branch tracks `origin/data-validation`. This report records technical compatibility and source identity; it does not approve scientific policies or claim a completed GitHub delivery.

## Shared interfaces

- `src/data/validate.py` and `src/data/eda.py` expose `run(config, project_root)` for the existing command router.
- `load_h2(path)` returns a DataFrame containing `row_id` and the 31 official source fields. It preserves source ordering, parses numerical fields, and keeps unknown values distinct from not-applicable identifiers. It does not filter or impute rows.
- `read_dataset(path)` is the internal validation/EDA interface returning both original tokens and the normalized view, plus the byte checksum.
- Input and output directories come from the shared `config.yaml`. Its raw path remains `data/H2.csv`.
- The existing preparation route, model routes, split assignments, feature policy, preprocessing builder and their tests are retained.
- The team's existing requirements and contribution files are retained. Earlier local deletions were not applied to the shared repository.

## Source identity reconciliation

The supplied export and the course mirror contain the same 79,330 records and the same 31 labels. Every field was compared row by row after trimming whitespace, interpreting documented missing tokens, comparing numerical measures with decimal arithmetic, and parsing the two explicit reservation-date formats. No canonical value or ordering differences were found.

| Source | SHA-256 |
|---|---|
| Supplied export | `ea4ef1bce2fc0d819e0abbbc1a475e789bca3a675be60cbc3313762aaff542ae` |
| Course mirror and existing split source | `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4` |

Text-format differences occur in IsRepeatedGuest, Agent, Company and ReservationStatusDate. Their byte checksums therefore differ even though the checked record values agree. The local working file uses the exact [course mirror](https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv), matching the existing split metadata. The supplied export remains backed up outside the repository, and neither raw file is included in Git.

The validator supports ISO `YYYY-MM-DD` and the supplied export's US `M/D/YYYY` reservation dates using explicit formats. Original tokens remain available for auditing. Graph interpretations and population counts are unchanged; some raw-token padding/type descriptions reflect the chosen source representation.

## Split and preprocessing compatibility

The shared manifest has 55,532 training rows, 11,899 validation rows and 11,899 test rows. These assignments are preserved. EDA describes the full source and does not fit or evaluate a predictive model. The combined integration test fits a small, disposable transformer only on rows identified as training by the shared manifest.

The EDA view preserves Agent/Company NULL as not applicable. The existing modeling builder maps NULL to its missing representation; Agent and Company are currently excluded from all active feature lists. No Member 2 policy or preprocessing code was changed. Before enabling those fields, agree the intended category treatment and version the policy with Michael.

## Reproduction and verification

From the repository root:

```powershell
.\.venv\Scripts\python.exe main.py validate
.\.venv\Scripts\python.exe main.py analyze
.\.venv\Scripts\python.exe main.py prepare
.\.venv\Scripts\python.exe scripts/verify_outputs.py
.\.venv\Scripts\python.exe -m pytest -q
```

`outputs/verification.json` records source/output reconciliation, current dependency versions and the combined test result. The existing manifest and split metadata must remain identical to the base repository after these commands. Main/config routing and the Member 2 modules must remain available. No baseline results are claimed.

## Verification outcome

All four command checks passed: shared help/routing, validate, analyze and prepare. The independent verifier and pytest each passed all 30 combined tests, with no skips. All 15 result tables were independently reconciled against the working source. The seven figures were checked for valid files and visually reviewed. Dependency consistency and Git exclusions passed.

The existing split manifest and metadata, scientific configuration, requirements, contribution rules, Member 2 modules and their tests match the base commit byte for byte. Only command help text changes in main.py. EDA reports and outputs were regenerated against the source whose checksum matches the frozen split. The independent Member 2 reader remains responsible for reproducing its grouping hashes; the normalized modeling DataFrame is not a replacement for that reader.

## Review and delivery

Makarious should review the shared README and command-help updates. Michael should review the source evidence and scientific proposals without silently changing the frozen split. The model owners receive the relevant findings from `data_review.md`. Commit and push the actual changes under Zeina's own account after local verification; no historical commits or reviewer approvals are fabricated.
