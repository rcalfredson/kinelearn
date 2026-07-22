from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from KineLearn.core.hard_negatives import (
    match_hard_negative_pool,
    maximum_rolling_mean,
    score_fully_negative_windows,
    select_diverse_hard_negative_pool,
)


class _FakeModel:
    def predict_on_batch(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X[:, :, :1], dtype=np.float32)


class HardNegativeScoringTests(unittest.TestCase):
    def test_maximum_rolling_mean_rewards_sustained_confidence(self) -> None:
        probabilities = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.6, 0.6, 0.6, 0.6],
            ]
        )
        np.testing.assert_allclose(
            maximum_rolling_mean(probabilities, 2),
            [0.5, 0.6],
        )

    def test_only_fully_negative_windows_are_scored(self) -> None:
        mmX = np.array(
            [
                [[0.1], [0.1], [0.1], [0.1]],
                [[0.8], [0.8], [0.8], [0.8]],
                [[0.9], [0.9], [0.9], [0.9]],
            ],
            dtype=np.float32,
        )
        mmY = np.zeros((3, 4, 1), dtype=np.uint8)
        mmY[2, 1:3, 0] = 1
        scores = score_fully_negative_windows(
            _FakeModel(),
            mmX,
            mmY,
            np.array(["a", "a", "b"], dtype=object),
            np.array([0, 10, 0]),
            behavior_idx=0,
            rolling_frames=2,
            batch_size=2,
        )

        self.assertEqual(scores["source_window_index"].tolist(), [1, 0])
        self.assertEqual(scores["hardness"].round(3).tolist(), [0.8, 0.1])

    def test_pool_selection_removes_overlapping_near_duplicates(self) -> None:
        scores = pd.DataFrame(
            {
                "source_window_index": [0, 1, 2, 3],
                "stem": ["a", "a", "a", "b"],
                "start": [0, 10, 80, 0],
                "hardness": [0.9, 0.8, 0.7, 0.6],
                "max_probability": [0.9, 0.8, 0.7, 0.6],
            }
        )
        pool = select_diverse_hard_negative_pool(
            scores,
            pool_fraction=0.5,
            min_start_separation=60,
        )

        self.assertEqual(pool["source_window_index"].tolist(), [0, 2])

    def test_pool_matches_by_stem_and_start_and_rejects_positive_labels(self) -> None:
        vids = np.array(["a", "a", "b"], dtype=object)
        starts = np.array([0, 10, 0])
        mmY = np.zeros((3, 4, 1), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmpdir:
            pool_path = Path(tmpdir) / "pool.csv"
            pd.DataFrame({"stem": ["b", "a"], "start": [0, 10]}).to_csv(
                pool_path, index=False
            )
            indices = match_hard_negative_pool(
                pool_path, vids, starts, mmY, behavior_idx=0
            )
            np.testing.assert_array_equal(indices, [2, 1])

            mmY[1, 0, 0] = 1
            with self.assertRaisesRegex(ValueError, "positive labels"):
                match_hard_negative_pool(
                    pool_path, vids, starts, mmY, behavior_idx=0
                )


if __name__ == "__main__":
    unittest.main()
