# Clustering guide

Start with `reports/clustering_final.md` for actual results and limitations.
The M1 report contains the complete 31-column feature table. The source module
`src/clustering/complete.py` implements M2; the M1 runner remains separate.

## Run locally

From the repository root, using the existing working environment:

```powershell
New-Item -ItemType Directory -Path outputs/tmp/local-temp -Force | Out-Null
$env:TEMP = Join-Path (Get-Location) "outputs/tmp/local-temp"
$env:TMP = $env:TEMP
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MPLCONFIGDIR = Join-Path (Get-Location) "outputs/tmp/matplotlib"
.\env\Scripts\python.exe main.py cluster view
.\env\Scripts\python.exe main.py cluster search
.\env\Scripts\python.exe main.py cluster predict --input examples/clustering_bookings.csv --output outputs/tmp/assignments.csv
```

To regenerate all M2 experiments (replaces task outputs):

```powershell
.\env\Scripts\python.exe main.py cluster run
.\env\Scripts\python.exe -m src.clustering.verify_complete
```

To rerun tests, use a NEW nonexistent base-temp path below outputs/tmp so pytest
does not delete unrelated data or reuse the inaccessible old Windows cache:

```powershell
.\env\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp outputs/tmp/my-new-test-run
```

For a fresh setup, use Python 3.12, create an isolated venv, and install
`requirements-clustering.txt`. Obtain the unchanged H2 course source at
`data/H2.csv`; the shared loader verifies its checksum against the frozen
manifest. The bundle local env launcher reuses the already installed packages from the
OneDrive project through a local .pth file and the bundled Python base; it is
not included in the source ZIP. A fully independent environment must install
the version constraints. The local working environment is verified, but a
separate-machine installation has not been exercised. Libraries are ordinary
Python packages; no notebook or network service is needed to use the CLI.

`python scripts/package_clustering.py` regenerates
`submission/clustering_source.zip`. It includes source/config/tests/reports and
saved tables/figures, excludes raw data, environments and fitted artifacts,
checks ZIP integrity, and includes a checksum manifest. Keep fitted models
locally for the live discussion. No commit, push, PR or message is performed.

## Important code and concepts

Supervised learning predicts supplied targets such as ADR or IsCanceled.
Clustering receives booking attributes without target answers. Cluster IDs
are arbitrary and can change across models. In Egyptian Arabic: بنجمع الحجوزات
المتشابهة، مش بنتوقع إجابة جاهزة.

`get_task_data("clustering", root, "train", config=config)` selects source
row IDs from the frozen training manifest and returns `(X, None)`. Keep None:
there is no supervised y for fitting or parameter selection.

`build_preprocessor("clustering", config)` creates an unfitted transformer.
Its fit learns medians, scales and category vocabulary from training data.
The complete M2 Pipeline then caps the transformed distance coordinates to
[-3, 3]. Numeric scale matters because Euclidean distance otherwise depends
on units. One-hot encoding represents categories without inventing an order,
but correlated inputs and category-block weights can still distort similarity.

`fit_record` constructs an algorithm, measures fit time and obtains labels.
K-Means repeatedly assigns rows to nearby centers and updates those centers.
`n_clusters` is the number of groups; `n_init` is the number of starting
solutions tried. GMM alternates membership probabilities and component
parameters. Agglomerative clustering merges groups using linkage distances.

`search_space` lists every parameter combination before evaluation. `choose`
rejects degenerate, tiny/dominant-group and unconverged solutions, then ranks
silhouette, DB and complexity. Optimization means choosing under this rule,
not guaranteeing that every metric improves. k=2 wins for K-Means; k=3/Ward
remains best for the hierarchy; no improvement is fabricated.

Silhouette compares a booking's within-cluster distances with distances to a
neighboring group. Higher is generally better; values near zero suggest
overlap. Davies-Bouldin summarizes within-group scatter relative to group
separation; lower is generally better. Neither score proves a business story.
Inertia is K-Means' squared-distance objective and depends on scale and k.

`stability` changes the training subset and uses adjusted Rand index to compare
partitions on shared rows, ignoring label numbering. ARI close to one indicates
similar partitions. These are unsupervised model-label comparisons, not use
of the cancellation outcome. Three refits do not establish universal stability.

`profiles` describes original booking units rather than standardized values.
Use means, medians, group sizes and category shares together. A category being
most common does not imply it is a majority. Never rename a group high-risk or
high-value without evidence that supports that claim.

PCA rotates numerical inputs into directions of greatest variance. We plot
two components and report retained variance; clustering and scoring happen
before that projection. LDA uses supervised labels and is not used for this
unsupervised representation. Supervised PCA/LDA experiments belong to the
team's other work packages.

The dendrogram shows hierarchical merges and their linkage distances.
The truncated figure uses a recorded sample because full pairwise hierarchy
construction is costly. It is a view of the sampled hierarchy, not every H2 row.

`joblib.dump` saves fitted transformations and estimators. `cluster predict`
loads the full K-Means pipeline, selects required columns, calls predict and
writes assignments. It never fits. The saved hierarchy is offered for viewing
only because it has no predict method. K-Means assignment IDs describe its
own two-group solution, not the selected three-group hierarchy.

## Explain this during discussion

1. Why does outcome exclusion not fully prove no leakage? Snapshot timing,
   preprocessing fitting and outcome-influenced sampling also matter.
2. Why preserve identical rows? Identical attributes do not prove the same
   booking; source identity and the agreed grouped split are preserved.
3. Why cap scaled values? Uncapped distance isolated an extreme singleton.
   Capping bounds its influence, but sacrifices distinctions in the tails.
4. Why have two selected methods? The hierarchy has stronger subset geometry;
   K-Means offers faster, more stable new-booking assignment.
5. What remains limited? No held-out clustering generalization, unresolved
   snapshot availability, small stability study and representation assumptions.

## Team handoff

Michael reviews population, timing, the task-specific cap and leakage audit.
Makarious receives the root CLI route, results.json, comparison.csv, search.json,
profiles and five figures. The source ZIP and regenerated local artifacts are
ready for review. No teammate's fitted model is required. Owner understanding,
scientific approval, identifiable Git history and official team assembly remain
human/team steps; no approval or remote publication is claimed here.

## Bundle verification

The final bundle test run passed 81 tests and 2 subtests in 32.23 seconds.
Both TEMP/TMP and pytest base-temp are workspace-local for this machine.
The source bundle is in the authorized workspace because final writes to the
moved OneDrive source encountered permission-review timeouts. It preserves
the existing team architecture and includes the final local documentation and
source updates. No remote update or merge into the OneDrive working copy is
claimed.
