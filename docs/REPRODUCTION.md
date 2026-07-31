# Reproduction

This is the single entry point for reproducing the public claims. The deterministic
lifecycle requires no external data and is specified in
[`ONE_DIMENSIONAL_REFERENCE.md`](ONE_DIMENSIONAL_REFERENCE.md). The C-MAPSS study
requires a user-obtained NASA dataset and should write to a new directory so the
checked-in evidence remains untouched.

## 1. Environment

Use Python 3.10 or newer on CPU:

```bash
python -m venv .venv
python -m pip install -e ".[benchmark,test]"
python -m pytest -q
```

Record Python, PyTorch, NumPy, SciPy and scikit-learn versions, operating system,
processor and total runtime with any reproduction report.

## 2. Deterministic lifecycle

```bash
python -m examples.quickstart
python -m examples.continual_stream
```

The quickstart must report:

- no served change before review;
- useful promotion and improved supported-region RMSE;
- rejection of the deliberately harmful candidate;
- no old/unknown behavior change; and
- exact restoration after revoke.

Small floating-point differences in the final improved RMSE are acceptable; any
Boolean lifecycle difference is not.

## 3. Obtain C-MAPSS

C-MAPSS is not redistributed. Download the original archive from the
[NASA dataset page](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
and extract it to `data/CMAPSSData/`. The directory must contain
`train_FD002.txt` exactly once.

Do not commit the downloaded data. The preparation command records hashes and source
provenance but cannot independently prove that a locally supplied archive is the
official download.

## 4. Isolate reproduction outputs

Set `VRSE_RESULTS_DIR` to a fresh directory before running any benchmark command.

PowerShell:

```powershell
$env:VRSE_RESULTS_DIR = "reproductions/cmapss-fd002-local"
```

Bash:

```bash
export VRSE_RESULTS_DIR="reproductions/cmapss-fd002-local"
```

The repository ignores `reproductions/`. Do not point this variable at the checked-in
`results/` directory.

## 5. Prepare and audit data

```bash
python -m experiments.cmapss_prepare_data \
  --source data/CMAPSSData \
  --source-origin user-attested-original-extraction
python -m experiments.cmapss_preconditions
```

Stop unless the precondition verdict is `READY_FOR_MATRIX`. An `INVALID`,
`STOP_TASK`, `PIVOT_DETECTOR` or `PIVOT_LEARNER` result is part of the reproduction
outcome and must not be bypassed by editing the artifact.

The debug-only `--skip-contract-gates` option can diagnose later checks but can never
produce a valid matrix precondition.

## 6. Run matrix, verdict and figures

```bash
python -m experiments.cmapss_matrix
python -m experiments.cmapss_verdict
python -m experiments.cmapss_visualize
```

Expected terminal verdict: `PASS`.

The output directory should contain the prepared data, source manifest, five
checkpoints, raw JSON and pickle matrices, preconditions, mechanical verdict, metric
table and three figures. Prepared arrays, checkpoints and pickle files are local
reproduction products; the release branch publishes the auditable JSON summaries and
figures.

## 7. Compare with the frozen result

Compare these public artifacts:

| Reproduction output | Frozen reference |
|---|---|
| `cmapss_fd002_data_manifest.json` | `results/cmapss_fd002_data_manifest.json` |
| `cmapss_fd002_matrix.json` | `results/cmapss_fd002_matrix.json` |
| `cmapss_fd002_verdict.json` | `results/cmapss_fd002_verdict.json` |
| `cmapss_fd002_metrics.md` | `results/cmapss_fd002_metrics.md` |

Exact JSON equality may include runtime fields that vary. A report should provide a
structured metric diff and call out every changed seed, promotion decision, route
fraction and error value. Promotion and routing decisions are expected to match.

## 8. Reproduction report template

```text
Environment:
Source-data hashes:
Commit:
Commands:
Runtime:
Precondition verdict:
Final verdict:
Promotion decision differences:
Maximum metric difference:
Figure differences:
Unexpected warnings or failures:
```

Open a reproduction issue even when the result fails to match. An unexplained
disagreement is more useful to the project than a silently adjusted rerun.
