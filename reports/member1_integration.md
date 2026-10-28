# Data interface and source compatibility

This report documents the data-loading interfaces, source identity and compatibility with the shared split and preprocessing components. Scientific choices remain provisional where recorded in `reports/decisions.md`.

## Data interfaces

- The validation and EDA modules expose `run(config, project_root)` for the shared command router.
- `load_h2(path)` returns a DataFrame containing `row_id` and the 31 official source fields. It preserves source order, parses numerical fields and does not filter or impute rows.
- `read_dataset(path)` additionally provides original tokens, a normalized analysis view and the source byte checksum for validation and EDA.
- Input and output directories come from `config.yaml`; the standard source path is `data/H2.csv`.
- The existing split, feature policy and preprocessing implementations are retained. The normalized modeling DataFrame does not replace the independent split reader when reproducing its group hashes.

## Source identity

The supplied export and the course mirror contain the same 79,330 records and 31 labels. Every field was compared in source order after trimming whitespace, interpreting documented missing tokens, comparing numerical measures with decimal arithmetic and parsing the two explicit reservation-date formats. No canonical value or ordering differences were found.

| Source | SHA-256 |
|---|---|
| Supplied export | `ea4ef1bce2fc0d819e0abbbc1a475e789bca3a675be60cbc3313762aaff542ae` |
| Course mirror and existing split source | `6796f5b9b2b9e7281662b29730fc2880f8e614107d0d14d09be2ede1a6e605a4` |

Text-format differences occur in IsRepeatedGuest, Agent, Company and ReservationStatusDate. The checksums therefore differ despite agreement of the compared record values. Shared experiments use the exact [course mirror](https://raw.githubusercontent.com/afaq-ahmad/Hotel-booking-demand-Case-Study/master/H2.csv), whose checksum matches the existing split metadata. Raw CSV files are excluded from Git.

The validator supports ISO `YYYY-MM-DD` and US `M/D/YYYY` reservation dates through explicit formats. Original tokens remain available for inspection. Population counts and graph interpretations agree across the two source representations; raw-token padding and observed dtype descriptions can differ.

## Split compatibility

The shared manifest contains 55,532 training rows, 11,899 validation rows and 11,899 test rows. Source identity, assignment counts and duplicate-group integrity are checked without changing these assignments. The source checksum matches the split metadata.

EDA describes the full source and does not fit or evaluate predictive models. The loader/preprocessor integration test fits a disposable transformer only on rows identified as training by the shared manifest. No baseline or final test-performance results are reported.

## Normalization decisions requiring review

| Area | EDA behavior | Existing shared behavior | Required review |
|---|---|---|---|
| Agent and Company NULL | Preserve NULL as a not-applicable category, following the dataset paper. | The modeling preprocessor converts NULL to missing. Both columns are excluded from all active feature lists. | Zeina and Michael must agree the representation before enabling either field and version the relevant feature/preprocessing policy. |
| Literal NA and duplicate groups | Interpret NA as missing in the analysis view. | The split reader trims text and normalizes blank/NULL only; literal NA remains part of the group-hash input. | Zeina and Michael must agree normalization for group identity. Any change requires review of its effect on hashes and assignments, a new split version where affected, and reproducible verification. |

These differences are documented, not resolved by this contribution. Exclusion of Agent and Company prevents their mismatch from affecting current active model inputs. The existing frozen manifest must remain intact until any normalization change is explicitly reviewed and versioned. Record agreed treatments, reviewers and dates in `reports/decisions.md`; no approval is implied here.

## Reproduction and verification

Run from the repository root after installing the development dependencies:

```text
python main.py validate
python main.py analyze
python main.py prepare
python scripts/verify_outputs.py
python -m pytest -q
```

The independent verifier reconciles source counts, missing/not-applicable semantics, unusual-record flags, duplicate summaries and output inventories. The combined tests cover Member 1 loading/validation, command and preprocessor integration, and Member 2 preparation. `outputs/verification.json` records the current result and dependency versions.

The existing split manifest, split metadata, scientific configuration and Member 2 implementation are preserved. Analysis produces 15 result tables and seven figures. Detailed findings and proposed treatments are in `eda_notes.md` and `data_review.md`.
