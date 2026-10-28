# Understanding Zeina's code

Start with the two commands. `main.py validate` reads the dataset and writes the check results. `main.py analyze` does that and also writes plots, tables, unusual-record evidence and generated reports. Neither command fits models or changes the raw CSV.

| Module | What to understand |
|---|---|
| `src/data/schema.py` | The expected labels and semantic roles. Agent and Company are categorical IDs, not measurements. |
| `src/data/load.py` | The shared `load_h2` returns a DataFrame with `row_id`; internal `read_dataset` also keeps original tokens in `raw` and a normalized analysis view. Row IDs are original zero-based record positions. The SHA-256 identifies the exact source file. |
| `src/data/validate.py` | Checks for schema/type/date problems and counts missing values, duplicates and unusual records. A review flag is not an instruction to delete a row. |
| `src/data/eda.py` | Produces descriptive summaries and seven plots. Logarithmic axes and zoomed plots change presentation, not the underlying records. |
| `src/data/review.py` | Looks at overlaps among unusual-record flags and distinguishes repeated groups, group members and excess duplicates. |
| `scripts/verify_outputs.py` | Uses independent CSV counting to reconcile saved evidence, then runs the unit tests. |

## Terms to explain

- **EDA:** Describe data before modeling using counts, summaries and graphs.
- **Median:** The middle value after sorting. A useful description of a skewed rate distribution.
- **Right skew:** Many lower values and a small number of much larger values.
- **Missing:** The value is unknown. Missing Children does not prove zero children.
- **Not applicable:** A meaningful category; Agent NULL means no agency booking.
- **Outlier:** An unusual value worth reviewing, not automatically an error.
- **Duplicate excess:** In a group of three identical records, there are three members but two excess rows.
- **Overlap:** The same row can have zero nights and zero ADR. Adding the flag counts double-counts it.
- **Leakage:** Future outcomes or information from evaluation data influencing a model or its selection. Michael maintains the existing provisional split and coordinates prediction-time decisions; learned preprocessing must use training only.

## Questions to practice

1. Why does the loader preserve both raw tokens and a normalized view?
2. Why are Company NULL values kept as a category?
3. How do the checksum and row ID work together?
4. Why not remove every identical row or extreme ADR automatically?
5. Why can the separate counts of unusual records not simply be added?
6. Which findings are confirmed, and which treatments still need scientific review?

## Shared integration

Commands use `run(config, project_root)` and the source path in `config.yaml`, currently `data/H2.csv`. The working source checksum matches the existing split metadata. The shared preparation code and assignments are retained; no new model results are claimed. See `reports/member1_integration.md` for details.
