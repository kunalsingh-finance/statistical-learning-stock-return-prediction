import tempfile
import json
import shutil
import contextlib
import io
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.fa590_stock_return_prediction.panel import align_next_month_target, prepare_features, chronological_split, select_model
from src.fa590_stock_return_prediction.pipeline import portfolio_performance, RunConfig, run_project
from scripts.verify_outputs import verify
from src.fa590_stock_return_prediction.prepare_wrds_merge import build_dataset


def prepared_panel():
    dates = pd.date_range("2020-01-31", periods=15, freq="ME")
    panel = pd.DataFrame({"permno": 1, "DATE": dates, "target_DATE": dates + pd.offsets.MonthEnd(1),
                          "RET": np.arange(15) / 100, "x": np.arange(15, dtype=float), "sic2": 10})
    panel.loc[0, "x"] = np.nan
    return panel


class ForecastValidityTests(unittest.TestCase):
    def test_target_moves_exactly_one_calendar_month(self):
        raw = pd.DataFrame({"permno": [1, 1, 1, 1], "DATE": ["2020-01-31", "2020-02-29", "2020-04-30", "2020-05-31"], "RET": [0.1, 0.2, 0.4, 0.5]})
        aligned = align_next_month_target(raw)
        self.assertEqual(aligned["RET"].tolist(), [0.2, 0.5])
        self.assertEqual(aligned["DATE"].dt.month.tolist(), [1, 4])

    def test_prepared_targets_are_not_shifted_again(self):
        panel = prepared_panel()
        assert_frame_equal(align_next_month_target(panel), panel)

    def test_same_month_or_missing_prepared_date_is_rejected(self):
        panel = prepared_panel()
        for invalid in [panel.loc[0, "DATE"], pd.NaT]:
            bad = panel.copy()
            bad.loc[0, "target_DATE"] = invalid
            with self.assertRaisesRegex(ValueError, "next calendar month"):
                align_next_month_target(bad)

    def test_future_features_cannot_change_training_preprocessing(self):
        panel, features = prepare_features(prepared_panel())
        split = chronological_split(panel, features)
        changed = prepared_panel()
        changed.loc[changed["DATE"] >= "2020-10-01", "x"] = 1e9
        changed.loc[changed["DATE"] >= "2020-10-01", "sic2"] = 99
        changed, other_features = prepare_features(changed)
        other = chronological_split(changed, other_features)
        assert_frame_equal(split["X_train"], other["X_train"])
        np.testing.assert_array_equal(split["X_train_scaled"], other["X_train_scaled"])
        self.assertEqual(split["imputation_medians"], other["imputation_medians"])
        self.assertNotIn(99, other["industry_categories"])

    def test_forward_labels_do_not_cross_split_origins(self):
        panel, features = prepare_features(prepared_panel())
        split = chronological_split(panel, features)
        self.assertLess(split["train_df"]["target_DATE"].max(), split["val_df"]["DATE"].min())
        self.assertLess(split["val_df"]["target_DATE"].max(), split["test_df"]["DATE"].min())

    def test_duplicate_and_short_panels_are_rejected(self):
        panel = prepared_panel()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            align_next_month_target(pd.concat([panel, panel.iloc[[0]]]))
        panel, features = prepare_features(panel.iloc[:5])
        with self.assertRaisesRegex(ValueError, "ten feature months"):
            chronological_split(panel, features)

    def test_selection_ignores_final_test_scores(self):
        scores = pd.DataFrame({"Model": ["A", "B", "A", "B"], "Dataset": ["Validation", "Validation", "Test", "Test"], "MSE": [2.0, 1.0, 0.0, 1e9]})
        self.assertEqual(select_model(scores), "B")
        scores.loc[scores["Dataset"].eq("Test"), "MSE"] = [1e9, 0.0]
        self.assertEqual(select_model(scores), "B")

    def test_wrds_merge_uses_next_month_not_same_month(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"permno": [1, 1], "DATE": ["20200131", "20200229"], "x": [1, 2]}).to_csv(root / "chars.csv", index=False)
            pd.DataFrame({"permno": [1, 1, 1], "DATE": ["20200131", "20200229", "20200331"], "RET": [0.1, 0.2, 0.3]}).to_csv(root / "returns.csv", index=False)
            merged = build_dataset(root / "chars.csv", root / "returns.csv", root / "out.csv")
            self.assertEqual(merged["RET"].tolist(), [0.2, 0.3])
            self.assertEqual(merged["target_DATE"].dt.month.tolist(), [2, 3])

    def test_top_quintile_is_a_ranked_subset(self):
        panel = pd.DataFrame({"DATE": ["2020-01-31"] * 20, "RET": np.arange(20) / 100})
        result = portfolio_performance(panel, np.arange(20), ["2020-01-31"], "RET")
        self.assertAlmostEqual(result["Avg_Return"], np.mean([0.16, 0.17, 0.18, 0.19]))
        self.assertEqual(result["N_Months"], 1)

    def test_small_monthly_population_is_not_reported_as_empty(self):
        panel = pd.DataFrame({"DATE": ["2020-01-31"] * 10, "RET": np.arange(10) / 100})
        result = portfolio_performance(panel, np.arange(10), ["2020-01-31"], "RET")
        self.assertAlmostEqual(result["Avg_Return"], 0.085)
        self.assertEqual(result["N_Months"], 1)

    def test_failed_rebuild_cannot_verify_previous_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_status.json").write_text(json.dumps({"status": "SUCCESS"}))
            (root / "run_summary.json").write_text(json.dumps({"selected_model": "STALE"}))
            with self.assertRaises(FileNotFoundError):
                run_project(RunConfig(data_path=str(root / "missing.csv"), output_dir=root, skip_neural_network=True))
            self.assertEqual(json.loads((root / "run_status.json").read_text())["status"], "ERROR")
            with self.assertRaisesRegex(AssertionError, "incomplete or failed"):
                verify(root)

    def test_saved_outputs_verify_across_checkout_line_endings_and_reject_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "outputs"
            shutil.copytree(Path(__file__).resolve().parents[1] / "sample_outputs", root)
            for path in root.glob("*.csv"):
                path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
            with contextlib.redirect_stdout(io.StringIO()):
                verify(root)
            for path in root.glob("*.csv"):
                path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
            with contextlib.redirect_stdout(io.StringIO()):
                verify(root)
            prediction_path = root / "predictions.csv"
            prediction_path.write_bytes(prediction_path.read_bytes() + b"changed")
            with self.assertRaisesRegex(AssertionError, "Output changed since successful run"):
                verify(root)


if __name__ == "__main__":
    unittest.main()
