# Zeina Milestone 1 checklist

Your role is to validate H2, explain its characteristics through EDA, and give the team reliable evidence for scientific decisions. Checked technical items mean evidence or notes have been prepared; they do not establish personal understanding or reviewer approval. Delivery and approval tasks remain open until actually completed.

This is a manual checklist. Running `main.py analyze` does not overwrite it. Generated reports are overwritten when analysis is rerun, so keep your own observations here. The canonical EDA report is `reports/eda_notes.md`; the investigation and recipient packages are `reports/data_review.md`. Tables and figures now live in `outputs/eda/tables/` and `outputs/eda/figures/`.

## Already generated and verified by the assistant

- [x] Create the H2 loader, validator, and EDA modules.
- [x] Preserve the original CSV and stable source row IDs.
- [x] Record the dataset checksum and confirm 79,330 rows and 31 source columns.
- [x] Generate validation JSON and the source schema table.
- [x] Generate fifteen EDA tables and seven figures.
- [x] Generate the initial handoff report.
- [x] Pass all 30 Member 1 and shared-project tests and independently reconcile all fifteen result tables.
- [x] Run the project successfully from the repository root.

These checks establish implementation and output generation. They do not mean Zeina or Michael has reviewed or approved the scientific interpretation.

## Start here

- [ ] Read `reports/eda_notes.md` and `reports/data_review.md` completely.
- [ ] Read `src/data/load.py` and explain how raw tokens, normalized values, the checksum, and row IDs are handled.
- [ ] Read `src/data/validate.py` and distinguish structural failures from descriptive review flags.
- [ ] Read `src/data/eda.py` and identify the tables and figures it produces.
- [x] Run validation and analysis yourself from the project folder (shown in the terminal during this session).
- [ ] Run the tests yourself and understand what their success establishes.

```powershell
.\.venv\Scripts\python.exe main.py validate
.\.venv\Scripts\python.exe main.py analyze
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Review the data evidence

- [ ] Review all column meanings and semantic types in `outputs/data/schema.csv`; check detailed meanings against the original dataset paper.
- [x] Explain the four missing Children values and 24 missing Country values without assuming their replacements.
- [x] Explain why Agent and Company NULL values are not-applicable categories, and why their IDs are categorical.
- [ ] Identify padded fields and explain the whitespace normalization.
- [ ] Explain the difference between 25,902 excess duplicate rows and 31,748 records belonging to duplicate groups.
- [x] Review `duplicate_group_sizes.csv` and write a proposed duplicate policy with reasons; leave final agreement to the team.
- [ ] Review `numeric_summary.csv` and `category_frequencies.csv` for skew, ranges, rare categories, and high cardinality.
- [x] Review `adr_extreme_examples.csv`, including ADR 5,400; record plausible explanations and uncertainty rather than declaring it an error.
- [x] Investigate 331 zero-night bookings, 1,208 zero-ADR records, and 167 records with zero total guests.
- [x] Review `adr_population_by_cancellation.csv` with Paula's regression population decision in mind.
- [ ] Explain the cancellation balance and why accuracy alone would be insufficient for evaluating a classifier.
- [x] Record proposed treatments, affected counts, and unresolved questions without deleting or imputing records independently.

## Figure observations discussed and recorded

- [x] `cancellation_balance.png`: explain the class proportions.
- [x] `adr_distribution.png`: explain the full range, log count axis, and percentile zoom.
- [x] `missing_values.png`: distinguish missing values from not-applicable identifiers.
- [x] `lead_time_and_stay_length.png`: describe skew and unusually long lead times or stays.
- [x] `arrivals_by_month.png`: describe patterns while accounting for partial endpoint years.
- [x] `cancellation_by_market_segment.png`: describe associations and flag tiny groups.
- [x] `adr_by_reserved_room.png`: describe differences while checking group sizes and noting hidden fliers.

## Give Michael the first handoff

Do this once your core validation findings are reviewed; you can continue refining EDA explanations afterward.

- [ ] Share `outputs/data/validation.json` and `outputs/data/schema.csv`.
- [ ] Share your reviewed notes on missingness, padding, duplicates, unusual values, and source meanings.
- [ ] Explain which observations are confirmed and which treatments are proposals.
- [ ] Ask Michael to scientifically review the evidence and its implications for cleaning and splitting.
- [ ] Record his feedback and any agreed decisions; distinguish agreement from open questions.

## Complete the remaining handoffs

- [ ] Paula: share ADR distributions, extremes, and evidence for canceled, zero-night, and zero-rate populations.
- [ ] Youssef: share class balance, category frequencies, and rare-category evidence.
- [ ] Aya: share numerical scales and skew, cardinalities, and population/duplicate findings.
- [ ] Makarious: share the reviewed figures, tables, notes, checksum, and exact reproduction commands.
- [ ] Give Makarious an accurate description of your actual contribution for the progress report.

## Finish your contribution

- [ ] Incorporate review feedback in your code or notes and rerun affected checks if anything changes.
- [x] Confirm the reported numbers match the saved output files.
- [x] Confirm your notes distinguish full-source descriptive EDA from model evaluation; Michael owns the existing provisional split and training-only preparation.
- [ ] Have Makarious review the integration changes to the shared entry-point help text and README.
- [ ] Commit your reviewed work incrementally using your own account in the team's GitHub repository.
- [x] Check that raw data, the virtual environment, caches, and trained artifacts are excluded from the commit.
- [ ] Submit your contribution for review through the team's agreed GitHub process.
- [ ] Review your section of the integrated Milestone 1 report and verify its figures, counts, interpretations, and contribution statement.
- [ ] Practice explaining your loader, checks, findings, limitations, and recommendations without reading the code line by line.

## Your observations and follow-up notes

Write the date, finding, proposed action, reviewer feedback, and agreed decision where relevant.

- Observations: seven graph notes are recorded below; unusual-record and duplicate evidence is in `reports/data_review.md`.
- Questions for Michael: agree missing-value treatment, duplicate/split policy and task populations; see the five open decisions in `reports/data_review.md`.
- Questions for the model owners: Paula reviews the ADR population; Youssef reviews cancellation features; Aya reviews segmentation features and population.
- Feedback received: none recorded yet.
- Decisions agreed: no scientific reviewer approval recorded.

### Missing values review

Country has 24 missing values, and Children has four. Agent and Company NULL values mean "not applicable" and are preserved as categories.

The missing-value inspection is complete. How to handle missing Country and Children values still needs agreement with Michael.

### Lead time and stay length review

Lead time and total nights are right-skewed. Short lead times and short stays are most common, while a small number of bookings have very long lead times or stays. Some bookings were made more than 500 days before arrival, and a few stays extend to almost 60 nights.

The stay-length graph uses a logarithmic count axis to make rare long stays visible. These unusual values need review before deciding whether any treatment is necessary. No records have been removed.

### Monthly arrivals review

The number of bookings scheduled to arrive varies across calendar months. May 2017 has the highest recorded count (4,556 bookings), while November 2015 has the lowest (1,235). In 2016, counts generally rise from winter into spring and early summer, with another peak in October.

These counts include canceled bookings, so they describe scheduled arrivals rather than actual guest arrivals or occupied rooms. The dataset covers July 2015 through August 2017; 2015 and 2017 contain only part of the year, so raw annual totals are not directly comparable. The limited time span suggests patterns worth investigating but does not establish a consistent seasonal pattern. No data treatment is required from this plot alone.

### Cancellations by market segment review

The bar height is the percentage canceled within each segment; n is the total number of bookings in that segment, not the number canceled. Groups has a cancellation rate of 68.9% across 13,975 bookings, compared with 42.8% for Offline TA/TO, 37.4% for Online TA, and 17.3% for Direct. Complementary has an observed rate of 11.8% across 542 bookings.

Undefined has a 100% cancellation rate, but contains only two bookings. This is insufficient evidence to generalize about future bookings in that category. The plot shows an association between market segment and cancellation, not proof that the segment causes cancellations or that the feature is available at the approved prediction point. Share the findings with Youssef and Michael for their feature and evaluation review. No category has been removed or changed based on this plot.

### ADR by reserved room type review

The orange line in each box is the median recorded ADR, and the box contains the middle 50% of values. The whiskers extend to the most extreme observed values within 1.5 interquartile ranges of the box edges; they do not necessarily show the minimum and maximum. Outlier points are hidden for readability, with no records removed.

Typical recorded ADR differs by reserved room type. A, B and C have medians near 90, while D, E, F and G have progressively higher medians, with G near 220. These are anonymized category codes, so the letters do not establish room quality or a meaningful numerical order. The distributions overlap.

Group sizes are very unequal: A has 62,595 bookings and D has 11,768, whereas C has only 14 and P only 10. P appears at zero ADR and needs inspection before interpretation; its small size does not justify assuming all future P bookings have a zero rate or deleting the category. Share the relationship and rare-category findings with Paula and Michael for regression feature and population review. No category has been removed or changed based on this plot.

### Cancellation balance review

There are 46,228 non-canceled bookings (58.3%) and 33,102 canceled bookings (41.7%). Non-canceled bookings are more common, but both classes have substantial numbers of records. These counts describe recorded outcomes, not model predictions or evaluation results. Share the balance with Youssef; classification evaluation should include precision, recall and F1 alongside accuracy.

### ADR distribution review

Recorded ADR is right-skewed: most bookings have relatively moderate rates, with common values roughly around 60 to 130, while a small number have much higher rates. The full-range plot includes the extreme ADR of 5,400 and uses a logarithmic count axis. The companion plot displays values at or below the 99th percentile (228) for readability; it does not remove records from the dataset.

Zero-rate bookings and the extreme recorded ADR need investigation with Paula and Michael before any treatment is chosen. The graph alone does not establish that these observations are errors. No ADR records have been removed or altered.


### Unusual records and duplicate evidence prepared

The reproducible investigation is in `reports/data_review.md`. It verifies the zero-night, zero-rate and zero-guest overlaps, preserves unknown guest counts, and identifies repeated groups without deleting records. Six supporting review tables are under `outputs/eda/tables/`. The handoff packages and proposed treatments are ready to review and share; delivery and scientific approval remain pending.

### Shared repository integration prepared

The local Git branch is `data-validation`, based on the shared main commit. The shared entry-point routing, configuration, dependency files and Member 2 code are retained. Member 1 runners use `run(config, project_root)`; the public loader returns a DataFrame with stable row IDs. The working CSV matches the existing split checksum. See `reports/member1_integration.md` for source-format reconciliation and checks. Scientific approval and GitHub delivery remain pending.
