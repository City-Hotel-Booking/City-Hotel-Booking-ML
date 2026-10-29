## Purpose

Integrate Member 1's checked H2 loader with Member 2's frozen split, finalize
unfitted task preprocessing, and hand off reproducible data access to Members 3-6.
This branch is based on `main` after PR #2 was merged. Its diff contains only
Member 2's integration work.

## Changed files

- `src/data/split.py`, `src/data/prepare.py`: shared raw-token reader, unchanged
  group rule, manifest and task-data interfaces.
- `src/features/policy.py`, `src/preprocessing/builders.py`, `config.yaml`:
  evidence-backed meanings and model-family preprocessing. Active feature
  lists and the frozen split version are unchanged.
- `src/preprocessing/checks.py`, `outputs/data/preprocessing_checks.json`:
  real-H2 train-only, leakage, determinism, and duplicate checks.
- `reports/decisions.md`, `reports/leakage_audit_m1.md`,
  `reports/handoff_member2.md`: evidence, open decisions and team handoff.

## Reproduce

```text
python -m src.data.prepare
python -m src.preprocessing.checks
python -m pytest -q
```

Expected evidence: `outputs/data/split_manifest.csv`,
`outputs/data/split_metadata.json`, and
`outputs/data/preprocessing_checks.json`. On the confirmed H2 file the frozen
manifest is identical, the nine preparation checks pass, and pytest reports
31 passed with two additional subtests passed. No model was fitted or scored
on final test rows. Scientific approvals remain pending in
`reports/decisions.md`.
