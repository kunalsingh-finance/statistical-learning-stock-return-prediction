"""Monthly target alignment and preprocessing without held-out sample statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

METADATA = {"permno", "DATE", "target_DATE", "RET", "name", "target_source"}


def align_next_month_target(df: pd.DataFrame) -> pd.DataFrame:
    """Align same-month RET to next month, or validate explicitly prepared targets.

    DATE is the feature month end. Gaps are never treated as one-month forecasts.
    A prepared input must contain target_DATE exactly one month after DATE.
    """
    missing = {"permno", "DATE", "RET"} - set(df)
    if missing:
        raise ValueError(f"Missing required panel columns: {sorted(missing)}")
    panel = df.copy()
    dates = pd.to_datetime(panel["DATE"], errors="coerce", format="mixed")
    if dates.isna().any() or panel["permno"].isna().any():
        raise ValueError("Security identifiers and feature dates must be present and valid")
    panel["DATE"] = dates.dt.to_period("M").dt.to_timestamp("M")
    if panel.duplicated(["permno", "DATE"]).any():
        raise ValueError("Duplicate security-month keys")
    panel["RET"] = pd.to_numeric(panel["RET"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    panel = panel.sort_values(["permno", "DATE"]).reset_index(drop=True)
    expected = panel["DATE"] + pd.offsets.MonthEnd(1)
    if "target_DATE" in panel:
        actual = pd.to_datetime(panel["target_DATE"], errors="coerce", format="mixed")
        if actual.isna().any() or not actual.eq(expected).all():
            raise ValueError("Prepared target_DATE must be exactly the next calendar month end")
        panel["target_DATE"] = actual
    else:
        groups = panel.groupby("permno", sort=False)
        panel["target_DATE"] = groups["DATE"].shift(-1)
        panel["RET"] = groups["RET"].shift(-1)
        panel = panel[panel["target_DATE"].eq(expected)].copy()
    panel = panel.dropna(subset=["RET"])
    if panel.empty:
        raise ValueError("No consecutive next-month targets remain")
    return panel.reset_index(drop=True)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Only past-value forward fills occur here; medians/encoding belong to fitting."""
    panel = df.sort_values(["permno", "DATE"]).copy()
    features = [col for col in panel if col not in METADATA]
    for col in features:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    panel[features] = panel.groupby("permno", sort=False)[features].ffill()
    return panel.reset_index(drop=True), features


def chronological_split(panel: pd.DataFrame, features: list[str]) -> dict:
    dates = sorted(panel["DATE"].unique())
    if len(dates) < 10:
        raise ValueError("At least ten feature months are required for three nonempty purged splits")
    train_end, val_end = int(0.6 * len(dates)), int(0.8 * len(dates))
    train_dates, val_dates, test_dates = dates[:train_end], dates[train_end:val_end], dates[val_end:]
    frames = {
        "train": panel[panel["DATE"].isin(train_dates) & panel["target_DATE"].lt(val_dates[0])].copy(),
        "val": panel[panel["DATE"].isin(val_dates) & panel["target_DATE"].lt(test_dates[0])].copy(),
        "test": panel[panel["DATE"].isin(test_dates)].copy(),
    }
    if any(frame.empty for frame in frames.values()):
        raise ValueError("Chronological target purging left an empty split")
    numeric = [col for col in features if col != "sic2"]
    medians = frames["train"][numeric].median().dropna()
    matrices = {key: frame[list(medians.index)].fillna(medians).astype(float) for key, frame in frames.items()}
    categories: list = []
    if "sic2" in features:
        categories = sorted(frames["train"]["sic2"].dropna().unique())
        for key, frame in frames.items():
            encoded = pd.get_dummies(pd.Categorical(frame["sic2"], categories=categories), prefix="sic2", dtype=float)
            encoded.index = frame.index
            # Unknown/missing industries map to all-zero indicators.
            matrices[key] = pd.concat([matrices[key], encoded], axis=1)
    if matrices["train"].shape[1] == 0:
        raise ValueError("No usable training features")
    scaler = StandardScaler().fit(matrices["train"])
    result = {"scaler": scaler, "feature_cols": list(matrices["train"]), "imputation_medians": medians.to_dict(),
              "industry_categories": categories, "purged_boundary_labels": True}
    for key, frame in frames.items():
        result[f"{key}_df"] = frame
        result[f"{key}_dates"] = sorted(frame["DATE"].unique())
        result[f"X_{key}"] = matrices[key]
        result[f"X_{key}_scaled"] = scaler.transform(matrices[key])
        result[f"y_{key}"] = frame["RET"]
    return result


def select_model(performance: pd.DataFrame) -> str:
    """Choose only by validation MSE; tie order is deterministic."""
    validation = performance[performance["Dataset"].eq("Validation")]
    if validation.empty or not np.isfinite(validation["MSE"]).all():
        raise ValueError("Finite validation scores are required for model selection")
    return str(validation.sort_values(["MSE", "Model"]).iloc[0]["Model"])
