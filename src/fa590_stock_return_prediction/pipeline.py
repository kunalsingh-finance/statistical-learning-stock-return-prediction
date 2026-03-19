from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models


@dataclass
class RunConfig:
    data_path: str | None
    output_dir: Path
    random_seed: int = 42
    rf_estimators: int = 50
    gb_estimators: int = 50
    nn_epochs: int = 25
    nn_batch_size: int = 256
    demo_months: int = 36
    demo_stocks: int = 180


def set_plot_style() -> None:
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 220
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["font.size"] = 10


def set_random_seeds(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def generate_demo_dataset(months: int, stocks_per_month: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-31", periods=months, freq="ME")
    rows: List[dict] = []

    for idx, date in enumerate(dates):
        market = rng.normal(0.008, 0.03)
        value_regime = np.sin(idx / 6.0)
        for stock in range(stocks_per_month):
            permno = 10000 + stock
            momentum_1m = rng.normal(0, 0.08)
            momentum_6m = rng.normal(0, 0.12) + 0.2 * value_regime
            size = rng.normal(10, 1.1)
            beta = rng.normal(1.0, 0.2)
            volatility = abs(rng.normal(0.22, 0.06))
            value = rng.normal(0.0, 1.0)
            quality = rng.normal(0.0, 1.0)
            profitability = rng.normal(0.05, 0.03)
            leverage = abs(rng.normal(0.35, 0.12))
            investment = rng.normal(0.03, 0.02)
            illiquidity = abs(rng.normal(0.015, 0.008))
            reversal = rng.normal(0.0, 0.07)
            sentiment = rng.normal(0.0, 0.9)
            sic2 = int(rng.choice([10, 20, 35, 50, 60, 70]))

            signal = (
                0.20 * momentum_1m
                + 0.12 * momentum_6m
                + 0.015 * value
                + 0.01 * quality
                - 0.02 * volatility
                - 0.015 * illiquidity
                + 0.01 * profitability
                - 0.008 * leverage
            )
            ret = market + signal + rng.normal(0, 0.06)

            rows.append(
                {
                    "permno": permno,
                    "DATE": date.strftime("%Y-%m-%d"),
                    "RET": ret,
                    "momentum_1m": momentum_1m,
                    "momentum_6m": momentum_6m,
                    "size_ln_mktcap": size,
                    "beta_12m": beta,
                    "volatility_12m": volatility,
                    "book_to_market": value,
                    "quality_score": quality,
                    "profitability": profitability,
                    "asset_growth": investment,
                    "leverage_ratio": leverage,
                    "illiquidity": illiquidity,
                    "short_term_reversal": reversal,
                    "sentiment_score": sentiment,
                    "sic2": sic2,
                }
            )

    df = pd.DataFrame(rows)
    for col in ["momentum_6m", "book_to_market", "quality_score", "profitability", "illiquidity"]:
        mask = rng.random(len(df)) < 0.03
        df.loc[mask, col] = np.nan
    return df


def load_dataset(config: RunConfig) -> Tuple[pd.DataFrame, str]:
    if config.data_path:
        data_path = Path(config.data_path).expanduser().resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        return pd.read_csv(data_path), "real"
    return generate_demo_dataset(config.demo_months, config.demo_stocks, config.random_seed), "demo"


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str], str]:
    id_cols = ["permno", "DATE"]
    if "name" in df.columns:
        id_cols.append("name")

    target = "RET"
    feature_cols = [col for col in df.columns if col not in id_cols + [target]]

    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.sort_values(["permno", "DATE"])
    df[feature_cols] = df.groupby("permno")[feature_cols].ffill()

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=[target])

    if "sic2" in df.columns and "sic2" in feature_cols:
        sic2_dummies = pd.get_dummies(df["sic2"], prefix="sic2", drop_first=True)
        df = pd.concat([df.drop(columns=["sic2"]), sic2_dummies], axis=1)
        feature_cols = [col for col in df.columns if col not in id_cols + [target]]

    return df.reset_index(drop=True), id_cols, feature_cols, target


def train_val_test_split(
    df: pd.DataFrame, feature_cols: List[str], target: str
) -> Dict[str, pd.DataFrame | np.ndarray | List[str] | StandardScaler]:
    df = df.sort_values("DATE").reset_index(drop=True)
    unique_dates = sorted(df["DATE"].unique())
    n_dates = len(unique_dates)
    train_end = max(int(0.6 * n_dates), 1)
    val_end = max(int(0.8 * n_dates), train_end + 1)

    train_dates = unique_dates[:train_end]
    val_dates = unique_dates[train_end:val_end]
    test_dates = unique_dates[val_end:]

    train_df = df[df["DATE"].isin(train_dates)].copy()
    val_df = df[df["DATE"].isin(val_dates)].copy()
    test_df = df[df["DATE"].isin(test_dates)].copy()

    x_train = train_df[feature_cols]
    x_val = val_df[feature_cols]
    x_test = test_df[feature_cols]
    y_train = train_df[target]
    y_val = val_df[target]
    y_test = test_df[target]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_dates": train_dates,
        "val_dates": val_dates,
        "test_dates": test_dates,
        "X_train": x_train,
        "X_val": x_val,
        "X_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_scaled": x_train_scaled,
        "X_val_scaled": x_val_scaled,
        "X_test_scaled": x_test_scaled,
        "scaler": scaler,
    }


def train_models(split: Dict[str, object], config: RunConfig) -> Tuple[Dict[str, dict], Dict[str, float], pd.DataFrame, RandomForestRegressor]:
    import time

    x_train = split["X_train"]
    x_val = split["X_val"]
    x_test = split["X_test"]
    y_train = split["y_train"]
    y_val = split["y_val"]
    x_train_scaled = split["X_train_scaled"]
    x_val_scaled = split["X_val_scaled"]
    x_test_scaled = split["X_test_scaled"]

    results: Dict[str, dict] = {}
    training_times: Dict[str, float] = {}

    start = time.time()
    ols = LinearRegression()
    ols.fit(x_train_scaled, y_train)
    training_times["OLS"] = time.time() - start
    results["OLS"] = {"train": ols.predict(x_train_scaled), "val": ols.predict(x_val_scaled), "test": ols.predict(x_test_scaled)}

    start = time.time()
    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train_scaled, y_train)
    training_times["Ridge"] = time.time() - start
    results["Ridge"] = {"train": ridge.predict(x_train_scaled), "val": ridge.predict(x_val_scaled), "test": ridge.predict(x_test_scaled)}

    start = time.time()
    lasso = Lasso(alpha=0.001, max_iter=5000)
    lasso.fit(x_train_scaled, y_train)
    training_times["Lasso"] = time.time() - start
    results["Lasso"] = {"train": lasso.predict(x_train_scaled), "val": lasso.predict(x_val_scaled), "test": lasso.predict(x_test_scaled)}

    start = time.time()
    rf = RandomForestRegressor(
        n_estimators=config.rf_estimators,
        max_depth=8,
        min_samples_split=50,
        n_jobs=1,
        random_state=config.random_seed,
    )
    rf.fit(x_train, y_train)
    training_times["RandomForest"] = time.time() - start
    results["RandomForest"] = {"train": rf.predict(x_train), "val": rf.predict(x_val), "test": rf.predict(x_test)}

    start = time.time()
    gb = GradientBoostingRegressor(
        n_estimators=config.gb_estimators,
        max_depth=3,
        learning_rate=0.05,
        random_state=config.random_seed,
    )
    gb.fit(x_train, y_train)
    training_times["GradientBoosting"] = time.time() - start
    results["GradientBoosting"] = {"train": gb.predict(x_train), "val": gb.predict(x_val), "test": gb.predict(x_test)}

    start = time.time()
    nn = models.Sequential(
        [
            layers.Input(shape=(x_train_scaled.shape[1],)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.15),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    history = nn.fit(
        x_train_scaled,
        y_train,
        validation_data=(x_val_scaled, y_val),
        epochs=config.nn_epochs,
        batch_size=config.nn_batch_size,
        callbacks=[early_stop],
        verbose=0,
    )
    training_times["NeuralNetwork"] = time.time() - start
    results["NeuralNetwork"] = {
        "train": nn.predict(x_train_scaled, verbose=0).flatten(),
        "val": nn.predict(x_val_scaled, verbose=0).flatten(),
        "test": nn.predict(x_test_scaled, verbose=0).flatten(),
    }

    return results, training_times, pd.DataFrame(history.history), rf


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, dataset_name: str, model_name: str) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "Model": model_name,
        "Dataset": dataset_name,
        "R2": r2_score(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
    }


def portfolio_performance(df_subset: pd.DataFrame, predictions: np.ndarray, dates_list: List[str], target: str) -> dict:
    df_temp = df_subset.copy()
    df_temp["Prediction"] = predictions
    monthly_returns: List[float] = []
    for date in dates_list:
        date_data = df_temp[df_temp["DATE"] == date]
        if len(date_data) < 20:
            continue
        top_n = min(100, len(date_data))
        top_bucket = date_data.nlargest(top_n, "Prediction")
        monthly_returns.append(float(top_bucket[target].mean()))

    volatility = float(np.std(monthly_returns)) if monthly_returns else 0.0
    avg_return = float(np.mean(monthly_returns)) if monthly_returns else 0.0
    sharpe = avg_return / volatility if volatility else 0.0
    return {
        "Avg_Return": avg_return,
        "Volatility": volatility,
        "Sharpe_Ratio": sharpe,
        "N_Months": len(monthly_returns),
        "Returns": monthly_returns,
    }


def save_target_distribution(df: pd.DataFrame, target: str, charts_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].hist(df[target], bins=100, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0, 0].set_title("Full Distribution of Stock Returns")
    axes[0, 0].set_xlabel("Monthly Return (RET)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(df[target].mean(), color="red", linestyle="--", label=f"Mean: {df[target].mean():.4f}")
    axes[0, 0].legend()

    axes[0, 1].hist(df[target], bins=100, edgecolor="black", alpha=0.7, color="darkgreen")
    axes[0, 1].set_title("Returns Distribution (Zoomed: -50% to +50%)")
    axes[0, 1].set_xlim(-0.5, 0.5)

    axes[1, 0].boxplot(df[target], vert=True)
    axes[1, 0].set_title("Boxplot of Stock Returns")
    stats.probplot(df[target], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot")

    plt.tight_layout()
    plt.savefig(charts_dir / "01_target_distribution.png", bbox_inches="tight")
    plt.close()


def save_returns_over_time(df: pd.DataFrame, target: str, charts_dir: Path) -> None:
    monthly_avg = df.groupby("DATE")[target].mean().reset_index()
    monthly_avg["DATE"] = pd.to_datetime(monthly_avg["DATE"])
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(monthly_avg["DATE"], monthly_avg[target], linewidth=1.5, color="darkblue")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.fill_between(monthly_avg["DATE"], monthly_avg[target], alpha=0.3, color="skyblue")
    ax.set_title("Average Stock Returns Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Monthly Return")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(charts_dir / "02_returns_over_time.png", bbox_inches="tight")
    plt.close()


def compute_correlations(df: pd.DataFrame, feature_cols: List[str], target: str) -> pd.Series:
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns[:100]
    return df[numeric_cols].corrwith(df[target]).abs().sort_values(ascending=False)


def save_feature_correlations(correlations: pd.Series, charts_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    top_20 = correlations.head(20)
    ax.barh(range(len(top_20)), top_20.values, color="teal")
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20.index)
    ax.set_xlabel("Absolute Correlation with Returns")
    ax.set_title("Top 20 Features by Correlation")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(charts_dir / "03_feature_correlations.png", bbox_inches="tight")
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame, correlations: pd.Series, target: str, charts_dir: Path) -> None:
    top_features = correlations.head(15).index.tolist()
    corr_matrix = df[top_features + [target]].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(charts_dir / "04_correlation_heatmap.png", bbox_inches="tight")
    plt.close()


def save_missing_data_pattern(raw_df: pd.DataFrame, charts_dir: Path) -> None:
    missing_counts = raw_df.isnull().sum().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(missing_counts)), missing_counts.values, color="coral")
    ax.set_xticks(range(len(missing_counts)))
    ax.set_xticklabels(missing_counts.index, rotation=45, ha="right")
    ax.set_ylabel("Number of Missing Values")
    ax.set_title("Top 20 Features with Missing Values")
    plt.tight_layout()
    plt.savefig(charts_dir / "05_missing_data_pattern.png", bbox_inches="tight")
    plt.close()


def save_data_split_chart(split: Dict[str, object], charts_dir: Path) -> None:
    split_data = pd.DataFrame(
        {
            "Period": ["Train", "Validation", "Test"],
            "Observations": [len(split["train_df"]), len(split["val_df"]), len(split["test_df"])],
            "Months": [len(split["train_dates"]), len(split["val_dates"]), len(split["test_dates"])],
        }
    )
    x = np.arange(len(split_data))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, split_data["Observations"], width, label="Observations", color="steelblue")
    ax2 = ax.twinx()
    ax2.bar(x + width / 2, split_data["Months"], width, label="Months", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(split_data["Period"])
    ax.set_xlabel("Dataset Split")
    ax.set_ylabel("Number of Observations", color="steelblue")
    ax2.set_ylabel("Number of Months", color="coral")
    ax.set_title("Train-Validation-Test Split")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(charts_dir / "06_data_split.png", bbox_inches="tight")
    plt.close()


def save_nn_history(history_df: pd.DataFrame, charts_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history_df["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history_df["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_title("Neural Network Training: Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history_df["mae"], label="Train MAE", linewidth=2)
    axes[1].plot(history_df["val_mae"], label="Val MAE", linewidth=2)
    axes[1].set_title("Neural Network Training: MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(charts_dir / "07_nn_training_history.png", bbox_inches="tight")
    plt.close()


def save_training_times(training_times: Dict[str, float], charts_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    models_list = list(training_times.keys())
    times = list(training_times.values())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    ax.barh(models_list, times, color=colors)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Model Training Time Comparison")
    ax.grid(True, alpha=0.3, axis="x")
    for idx, value in enumerate(times):
        ax.text(value, idx, f" {value:.2f}s", va="center")
    plt.tight_layout()
    plt.savefig(charts_dir / "08_training_times.png", bbox_inches="tight")
    plt.close()


def save_performance_charts(perf_df: pd.DataFrame, results: Dict[str, dict], split: Dict[str, object], charts_dir: Path) -> str:
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    test_perf = perf_df[perf_df["Dataset"] == "Test"].reset_index(drop=True)
    x = np.arange(len(test_perf))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, test_perf["R2"], color=colors)
    ax.set_xlabel("Model")
    ax.set_ylabel("R-squared")
    ax.set_title("Model Performance: R2 on Test Set")
    ax.set_xticks(x)
    ax.set_xticklabels(test_perf["Model"], rotation=15)
    for idx, value in enumerate(test_perf["R2"]):
        ax.text(idx, value, f"{value:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(charts_dir / "09_r2_comparison.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, test_perf["MSE"], color=colors)
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Model Performance: MSE on Test Set")
    ax.set_xticks(x)
    ax.set_xticklabels(test_perf["Model"], rotation=15)
    plt.tight_layout()
    plt.savefig(charts_dir / "10_mse_comparison.png", bbox_inches="tight")
    plt.close()

    best_model = test_perf.loc[test_perf["R2"].idxmax(), "Model"]
    y_test = split["y_test"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(y_test, results[best_model]["test"], alpha=0.3, s=10, color="steelblue")
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    axes[0].set_title(f"Actual vs Predicted: {best_model}")
    axes[0].set_xlabel("Actual Returns")
    axes[0].set_ylabel("Predicted Returns")

    residuals = y_test - results[best_model]["test"]
    axes[1].scatter(results[best_model]["test"], residuals, alpha=0.3, s=10, color="coral")
    axes[1].axhline(y=0, color="r", linestyle="--")
    axes[1].set_title(f"Residual Plot: {best_model}")
    axes[1].set_xlabel("Predicted Returns")
    axes[1].set_ylabel("Residuals")
    plt.tight_layout()
    plt.savefig(charts_dir / "11_actual_vs_predicted.png", bbox_inches="tight")
    plt.close()

    return best_model


def save_portfolio_outputs(results: Dict[str, dict], split: Dict[str, object], target: str, charts_dir: Path) -> Tuple[pd.DataFrame, Dict[str, List[float]]]:
    portfolio_metrics = []
    portfolio_returns: Dict[str, List[float]] = {}
    train_df = split["train_df"]
    test_df = split["test_df"]
    train_dates = split["train_dates"]
    test_dates = split["test_dates"]

    for model_name, preds in results.items():
        port_train = portfolio_performance(train_df, preds["train"], train_dates, target)
        port_test = portfolio_performance(test_df, preds["test"], test_dates, target)
        portfolio_metrics.append({"Model": model_name, "Dataset": "Train", "Avg_Return": port_train["Avg_Return"], "Volatility": port_train["Volatility"], "Sharpe_Ratio": port_train["Sharpe_Ratio"]})
        portfolio_metrics.append({"Model": model_name, "Dataset": "Test", "Avg_Return": port_test["Avg_Return"], "Volatility": port_test["Volatility"], "Sharpe_Ratio": port_test["Sharpe_Ratio"]})
        portfolio_returns[model_name] = port_test["Returns"]

    port_df = pd.DataFrame(portfolio_metrics)
    port_test_df = port_df[port_df["Dataset"] == "Test"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    axes[0].barh(port_test_df["Model"], port_test_df["Avg_Return"], color=colors)
    axes[0].set_title("Portfolio Average Return")
    axes[1].barh(port_test_df["Model"], port_test_df["Volatility"], color=colors)
    axes[1].set_title("Portfolio Volatility")
    axes[2].barh(port_test_df["Model"], port_test_df["Sharpe_Ratio"], color=colors)
    axes[2].set_title("Portfolio Sharpe Ratio")
    plt.tight_layout()
    plt.savefig(charts_dir / "12_portfolio_metrics.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 7))
    for model_name, returns in portfolio_returns.items():
        if returns:
            cumulative = np.cumprod(1 + np.array(returns)) - 1
            ax.plot(range(len(cumulative)), cumulative, label=model_name, linewidth=2)
    ax.set_title("Cumulative Portfolio Returns")
    ax.set_xlabel("Months (Test Period)")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(charts_dir / "13_cumulative_returns.png", bbox_inches="tight")
    plt.close()

    return port_df, portfolio_returns


def save_feature_importance(rf_model: RandomForestRegressor, feature_cols: List[str], charts_dir: Path) -> pd.DataFrame:
    rf_importance = pd.DataFrame({"Feature": feature_cols, "Importance": rf_model.feature_importances_}).sort_values("Importance", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 8))
    top_20 = rf_importance.head(20)
    ax.barh(range(len(top_20)), top_20["Importance"], color="forestgreen")
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20["Feature"])
    ax.set_title("Top 20 Feature Importances")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(charts_dir / "14_feature_importance.png", bbox_inches="tight")
    plt.close()
    return rf_importance


def save_train_vs_test_chart(perf_df: pd.DataFrame, charts_dir: Path) -> None:
    train_perf = perf_df[perf_df["Dataset"] == "Train"].reset_index(drop=True)
    test_perf = perf_df[perf_df["Dataset"] == "Test"].reset_index(drop=True)
    x = np.arange(len(train_perf))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(x - width / 2, train_perf["R2"], width, label="Train", color="steelblue", alpha=0.8)
    axes[0].bar(x + width / 2, test_perf["R2"], width, label="Test", color="coral", alpha=0.8)
    axes[0].set_title("R2 Comparison: Train vs Test")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(train_perf["Model"], rotation=15, ha="right")
    axes[0].legend()

    axes[1].bar(x - width / 2, train_perf["MSE"], width, label="Train", color="steelblue", alpha=0.8)
    axes[1].bar(x + width / 2, test_perf["MSE"], width, label="Test", color="coral", alpha=0.8)
    axes[1].set_title("MSE Comparison: Train vs Test")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(train_perf["Model"], rotation=15, ha="right")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "15_train_vs_test.png", bbox_inches="tight")
    plt.close()


def save_risk_return_tradeoff(port_df: pd.DataFrame, charts_dir: Path) -> None:
    port_test = port_df[port_df["Dataset"] == "Test"]
    color_map = {"OLS": "#1f77b4", "Ridge": "#ff7f0e", "Lasso": "#2ca02c", "RandomForest": "#d62728", "GradientBoosting": "#9467bd", "NeuralNetwork": "#8c564b"}
    fig, ax = plt.subplots(figsize=(10, 8))
    for _, row in port_test.iterrows():
        ax.scatter(row["Volatility"], row["Avg_Return"], s=240, alpha=0.8, color=color_map[row["Model"]], label=row["Model"], edgecolors="black", linewidth=1.0)
    ax.set_xlabel("Portfolio Volatility")
    ax.set_ylabel("Average Return")
    ax.set_title("Risk-Return Tradeoff")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(charts_dir / "16_risk_return_tradeoff.png", bbox_inches="tight")
    plt.close()


def run_project(config: RunConfig) -> dict:
    set_plot_style()
    set_random_seeds(config.random_seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = config.output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    raw_df, data_mode = load_dataset(config)
    processed_df, _, feature_cols, target = preprocess(raw_df)
    split = train_val_test_split(processed_df, feature_cols, target)

    save_target_distribution(processed_df, target, charts_dir)
    save_returns_over_time(processed_df, target, charts_dir)
    correlations = compute_correlations(processed_df, feature_cols, target)
    save_feature_correlations(correlations, charts_dir)
    save_correlation_heatmap(processed_df, correlations, target, charts_dir)
    save_missing_data_pattern(raw_df, charts_dir)
    save_data_split_chart(split, charts_dir)

    results, training_times, history_df, rf_model = train_models(split, config)
    save_nn_history(history_df, charts_dir)
    save_training_times(training_times, charts_dir)

    performance_metrics = []
    for model_name, preds in results.items():
        performance_metrics.append(evaluate_predictions(split["y_train"], preds["train"], "Train", model_name))
        performance_metrics.append(evaluate_predictions(split["y_val"], preds["val"], "Validation", model_name))
        performance_metrics.append(evaluate_predictions(split["y_test"], preds["test"], "Test", model_name))
    perf_df = pd.DataFrame(performance_metrics)
    perf_df.to_csv(config.output_dir / "predictive_performance_detailed.csv", index=False)
    best_model = save_performance_charts(perf_df, results, split, charts_dir)

    port_df, _ = save_portfolio_outputs(results, split, target, charts_dir)
    port_df.to_csv(config.output_dir / "portfolio_performance.csv", index=False)

    rf_importance = save_feature_importance(rf_model, feature_cols, charts_dir)
    rf_importance.to_csv(config.output_dir / "feature_importance.csv", index=False)

    save_train_vs_test_chart(perf_df, charts_dir)
    save_risk_return_tradeoff(port_df, charts_dir)

    best_r2 = float(perf_df.query("Dataset == 'Test'").sort_values("R2", ascending=False).iloc[0]["R2"])
    best_port = port_df.query("Dataset == 'Test'").sort_values("Sharpe_Ratio", ascending=False).iloc[0]

    summary = {
        "data_mode": data_mode,
        "rows": int(processed_df.shape[0]),
        "features": int(len(feature_cols)),
        "date_range": [str(processed_df["DATE"].min()), str(processed_df["DATE"].max())],
        "best_model_by_r2": best_model,
        "best_test_r2": best_r2,
        "best_portfolio_by_sharpe": str(best_port["Model"]),
        "best_test_sharpe": float(best_port["Sharpe_Ratio"]),
        "output_dir": str(config.output_dir),
        "charts_dir": str(charts_dir),
        "chart_count": 16,
    }
    (config.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
