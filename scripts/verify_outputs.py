"""Independently reconcile saved scores and selection to dated predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def verify(directory: Path):
    status = json.loads((directory / "run_status.json").read_text())
    assert status["status"] == "SUCCESS", "Output generation is incomplete or failed"
    required = {"run_summary.json", "predictions.csv", "predictive_performance_detailed.csv",
                "portfolio_performance.csv", "feature_importance.csv"}
    hashes = status["artifact_sha256"]
    assert required.issubset(hashes), "Successful run record lacks required outputs"
    for name, expected_hash in hashes.items():
        path = directory / name
        assert path.resolve().is_relative_to(directory.resolve()), "Artifact escapes output directory"
        raw = path.read_bytes()
        if path.suffix in {".csv", ".json"}:
            raw = raw.replace(b"\r\n", b"\n")
        assert hashlib.sha256(raw).hexdigest() == expected_hash, f"Output changed since successful run: {name}"
    summary = json.loads((directory / "run_summary.json").read_text())
    predictions = pd.read_csv(directory / "predictions.csv", parse_dates=["DATE", "target_DATE"])
    scores = pd.read_csv(directory / "predictive_performance_detailed.csv")
    portfolios = pd.read_csv(directory / "portfolio_performance.csv")
    assert summary["configuration"]["neural_network_enabled"] or not (directory / "charts/07_nn_training_history.png").exists()
    assert not predictions.duplicated(["Model", "Dataset", "permno", "DATE"]).any()
    assert predictions["target_DATE"].eq(predictions["DATE"] + pd.offsets.MonthEnd(1)).all()
    assert np.isfinite(predictions[["RET", "Prediction"]]).all().all()
    for (model, dataset), rows in predictions.groupby(["Model", "Dataset"]):
        mse = np.square(rows["RET"] - rows["Prediction"]).mean()
        expected = scores[scores["Model"].eq(model) & scores["Dataset"].eq(dataset)].iloc[0]
        np.testing.assert_allclose(mse, expected["MSE"], rtol=1e-10, atol=1e-12)
        if dataset == "Test":
            mean_return = rows.groupby("DATE").apply(lambda group: group.nlargest(max(1, int(np.ceil(len(group) * 0.2))), "Prediction")["RET"].mean(), include_groups=False)
            portfolio = portfolios[portfolios["Model"].eq(model) & portfolios["Dataset"].eq(dataset)].iloc[0]
            assert portfolio["N_Months"] == len(mean_return)
            np.testing.assert_allclose(mean_return.mean(), portfolio["Avg_Return"], rtol=1e-10, atol=1e-12)
    validation = scores[scores["Dataset"].eq("Validation")].sort_values(["MSE", "Model"])
    assert summary["selected_model"] == validation.iloc[0]["Model"]
    selected_test = scores[scores["Dataset"].eq("Test") & scores["Model"].eq(summary["selected_model"])].iloc[0]
    np.testing.assert_allclose(summary["selected_test_r2"], selected_test["R2"], rtol=1e-10, atol=1e-12)
    train = predictions[predictions["Dataset"].eq("Train")]
    val = predictions[predictions["Dataset"].eq("Validation")]
    test = predictions[predictions["Dataset"].eq("Test")]
    assert train["target_DATE"].max() < val["DATE"].min()
    assert val["target_DATE"].max() < test["DATE"].min()
    print(json.dumps({"status": "passed", "prediction_rows": len(predictions), "models": len(validation), "selected_model": summary["selected_model"], "data_mode": summary["data_mode"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("sample_outputs"))
    verify(parser.parse_args().output_dir)
