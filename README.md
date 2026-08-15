# Budget-Aware Statistical Alerting

Author: Le Chen

Code and outputs for the dissertation using the Los Alamos National Laboratory (LANL) Cyber1 authentication data. The original code is on the Apollo computer server, Department of Mathematics, at Imperial College London.

## Data

Download the [LANL Cyber1 dataset](https://doi.org/10.17021/1179829) and place these files in `raw/`:

```text
raw/auth.txt.gz
raw/redteam.txt.gz
```

The raw data and large Parquet files are not included in the repository.

## Environment

The analysis uses Python 3. Create an environment and install the packages with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository structure

1. Data preparation: `01_aggregate_windows.py` to `03_redteam_user_window_labels.py` construct the five-minute count data and weak labels.
2. Scoring: `04_poisson_score_human.py` and `25_scoring_backsolved.py` produce the two baseline specifications and raw anomaly scores.
3. Calibration: `26_recalibration.py` and `27_baseline_comparison.py` apply empirical recalibration and compare the baselines.
4. Sensitivity checks: `check/` contains the diagnostic scripts examine discreteness, negative binomial dispersion, recalibrated rankings, and weak labels.
5. Output policies: `policies/` contains the six policy implementations and their common comparison.
6. The remaining scripts produce data summaries and figures, and the utility modules provide common functions.

Generated tables and figures are stored in `outputs/tables/` and `outputs/figures/`.

## License

The original code in this repository is available under the MIT License. See [LICENSE](LICENSE). The LANL data are not redistributed and remain subject to their original terms.
