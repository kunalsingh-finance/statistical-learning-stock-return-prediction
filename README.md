# FA590 Stock Return Prediction

Statistical learning project for cross-sectional stock return prediction in Python. The project compares linear, regularized, tree-based, and neural-network models and exports charts plus summary CSVs.

## Project structure

- `notebooks/fa590_stock_return_prediction.ipynb`: original notebook
- `src/fa590_stock_return_prediction/run.py`: command-line runner
- `src/fa590_stock_return_prediction/pipeline.py`: preprocessing, modeling, and chart generation
- `src/fa590_stock_return_prediction/prepare_datashare_proxy.py`: builds a public proxy dataset from `datashare.zip`
- `sample_data/public_proxy_sample.csv`: small sample dataset for quick local runs
- `sample_outputs/`: curated charts and CSV outputs from the public proxy run
- `outputs/`: generated CSVs and charts

## Models

- OLS
- Ridge
- Lasso
- Random Forest
- Gradient Boosting
- Neural Network

## Data

The original notebook expects a CSV with at least these columns:

- `permno`
- `DATE`
- `RET`

If `sic2` or `name` exist, the runner handles them automatically.

If no dataset path is provided, the project generates a finance-shaped demo dataset so the full workflow can still run end to end.

The public `datashare.zip` file from Gu, Kelly, and Xiu includes the firm characteristics but does not include CRSP returns. To build the full project dataset, merge:

- `datashare.csv` for characteristics
- a WRDS CRSP monthly return export with `permno`, `DATE`, and `RET`

Use the helper below after you download both files:

```bash
python -m src.fa590_stock_return_prediction.prepare_wrds_merge --chars path\to\datashare.csv --returns path\to\crsp_returns.csv --out data\return_predictability_data_2009to2021.csv
```

If you do not have WRDS access, you can still build a public proxy dataset directly from `datashare.csv`. This version uses `mom1m` as a stand-in response variable, removes it from the feature set to avoid leakage, and keeps a manageable sample of stocks for a fully runnable public project:

```bash
python -m src.fa590_stock_return_prediction.prepare_datashare_proxy --zip-path "C:\Users\perso\Downloads\datashare (1).zip" --out data\public_proxy_return_predictability_2009to2021.csv --max-permnos 600
```

## Run

From the project root:

```bash
python -m src.fa590_stock_return_prediction.run
```

Run against a real dataset:

```bash
python -m src.fa590_stock_return_prediction.run --data-path "C:\path\to\return_predictability_data_2009to2021.csv"
```

Run against the public proxy dataset:

```bash
python -m src.fa590_stock_return_prediction.run --data-path data\public_proxy_return_predictability_2009to2021.csv --output-dir outputs\public_proxy_run
```

Run a quick sample version from the repo:

```bash
python -m src.fa590_stock_return_prediction.run --data-path sample_data\public_proxy_sample.csv --output-dir outputs\sample_run
```

Send outputs to a custom directory:

```bash
python -m src.fa590_stock_return_prediction.run --output-dir outputs\real_run
```

## Outputs

Each run creates:

- `predictive_performance_detailed.csv`
- `portfolio_performance.csv`
- `feature_importance.csv`
- `run_summary.json`
- `charts/` with visual diagnostics and model comparison plots

## Included showcase files

The repo includes a curated set of artifacts from the public proxy run:

- `sample_outputs/run_summary.json`
- `sample_outputs/predictive_performance_detailed.csv`
- `sample_outputs/portfolio_performance.csv`
- `sample_outputs/feature_importance.csv`
- `sample_outputs/charts/04_correlation_heatmap.png`
- `sample_outputs/charts/09_r2_comparison.png`
- `sample_outputs/charts/13_cumulative_returns.png`
- `sample_outputs/charts/16_risk_return_tradeoff.png`

## Resume description

Built a stock return prediction project in Python using linear, regularized, tree-based, and neural network models to evaluate predictive performance on panel market data.
