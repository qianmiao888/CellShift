import unittest

import numpy as np

from virtual_cell.metrics import diagnostic_metrics, global_r2, log2_rmse


class MetricTest(unittest.TestCase):
    def test_missing_values_are_masked(self) -> None:
        truth = np.asarray([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
        prediction = np.asarray([[1.0, 99.0], [2.0, 4.0]], dtype=np.float32)
        self.assertAlmostEqual(log2_rmse(truth, prediction), np.sqrt(1.0 / 3.0))
        self.assertTrue(np.isfinite(global_r2(truth, prediction)))

    def test_perfect_prediction(self) -> None:
        truth = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        metrics = diagnostic_metrics(truth, truth.copy())
        self.assertEqual(metrics["log2_rmse"], 0.0)
        self.assertEqual(metrics["global_r2"], 1.0)
        self.assertEqual(metrics["protein_r2_median"], 1.0)


if __name__ == "__main__":
    unittest.main()

