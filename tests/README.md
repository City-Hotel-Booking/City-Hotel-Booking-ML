# Focused checks

Owners add real tests with their implementations. No ML tests or test outcomes exist yet.

- Member 1: loader/source-label checks, padded strings, and meaningful not-applicable values.
- Member 2: disjoint reproducible splits, stable row IDs, approved input lists, unseen categories, and training-only learned preprocessing.
- Members 3 and 4: shared split use, required validation metrics, and no test evaluation in M1 runners.
- Member 5: outcome columns excluded from fitting and parameter selection; recorded sampling if used.
- Member 6: command routing and common result-record format.

After tests have been written, run `python -m pytest` from the project root with development dependencies installed. A passing setup check alone does not establish that the ML work is complete.

