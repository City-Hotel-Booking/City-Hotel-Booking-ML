# Focused checks

Owners add focused tests with their implementations. Member 2's policy, split, and preprocessing checks are in test_policy.py, test_split.py, and test_builders.py. Model tests are still pending.

- Member 1: loader/source-label checks, padded strings, and meaningful not-applicable values.
- Member 2: disjoint reproducible splits, stable row IDs, approved input lists, unseen categories, and training-only learned preprocessing.
- Members 3 and 4: shared split use, required validation metrics, and no test evaluation in M1 runners.
- Member 5: outcome columns excluded from fitting and parameter selection; recorded sampling if used.
- Member 6: command routing and common result-record format.

Run python -m pytest from the project root with development dependencies installed. Passing split tests does not establish that the model experiments are complete.

