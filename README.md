# Statistical Learning for Stock Return Prediction

This project compares statistical learning models for cross-sectional stock return prediction in Python. It evaluates linear, regularized, tree-based, and neural-network models on market-style panel data and exports model diagnostics, summary CSVs, and charts.

The project is educational and does not represent financial advice, a live trading strategy, or production investment software.

## What This Project Does

- Prepares panel-style stock return data for modeling
- Handles numeric feature selection, preprocessing, and train/test workflow
- Compares OLS, Ridge, Lasso, Random Forest, Gradient Boosting, and Neural Network models
- Exports predictive performance metrics and model comparison outputs
- Generates feature importance, correlation, cumulative return, and risk-return diagnostic charts
- Includes a public proxy dataset path so the workflow can run without restricted WRDS data

## Project Structure

- `notebooks/fa590_stock_return_prediction.ipynb`: original notebook workflow
- `src/fa590_stock_return_prediction/run.py`: command-line runner
- `src/fa590_stock_return_prediction/pipeline.py`: preprocessing, modeling, and chart generation
- `src/fa590_stock_return_prediction/prepare_datashare_proxy.py`: public proxy dataset builder
- `sample_data/public_proxy_sample.csv`: small sample dataset for quick local runs
- `sample_outputs/`: curated charts and CSV outputs from the public proxy run
- `outputs/`: generated model outputs

## Models Compared

- Ordinary Least Squares
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting
- Neural Network

## Data Notes

The original workflow expects a stock-level panel dataset with at least:

- `permno`
- `DATE`
- `RET`

The project also supports a public proxy dataset built from the Gu, Kelly, and Xiu `datashare` characteristics file. Because the public characteristics file does not include CRSP returns, the proxy version uses `mom1m` as a stand-in response variable and removes it from features to avoid direct leakage.

For a full research version, merge:

- `datashare.csv` for firm characteristics
- WRDS CRSP monthly returns with `permno`, `DATE`, and `RET`

## How to Run

Run the default workflow:

```bash
python -m src.fa590_stock_return_prediction.run
```

Run against a real dataset:

```bash
python -m src.fa590_stock_return_prediction.run --data-path "C:\path\to\return_predictability_data_2009to2021.csv"
```

Run the included sample workflow:

```bash
python -m src.fa590_stock_return_prediction.run --data-path sample_data\public_proxy_sample.csv --output-dir outputs\sample_run
```

Build a public proxy dataset from `datashare.zip`:

```bash
python -m src.fa590_stock_return_prediction.prepare_datashare_proxy --zip-path "C:\path\to\datashare.zip" --out data\public_proxy_return_predictability_2009to2021.csv --max-permnos 600
```

## Outputs

Each run creates:

- `predictive_performance_detailed.csv`
- `portfolio_performance.csv`
- `feature_importance.csv`
- `run_summary.json`
- charts with model comparison and diagnostics

Included showcase artifacts live under `sample_outputs/`.

## Skills Demonstrated

- Cross-sectional predictive modeling
- Financial panel data preparation
- Regularized regression
- Tree-based models
- Neural network baseline modeling
- Model comparison and diagnostic reporting

## Limitations

- Predictive results are highly sensitive to data source, sample period, feature set, and validation method.
- The public proxy dataset is for demonstration only and is not equivalent to a full CRSP/WRDS return panel.
- This is an educational modeling project, not a live trading strategy.
