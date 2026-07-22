from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from KineLearn.core import generators


@unittest.skipIf(generators.tf is None, "TensorFlow is not installed")
class StratifiedWindowGeneratorTests(unittest.TestCase):
    def make_generator(self, seed: int = 7) -> generators.StratifiedWindowGenerator:
        mmX = np.zeros((10, 4, 1), dtype=np.float32)
        for index in range(10):
            mmX[index, :, 0] = index
        mmY = np.zeros((10, 4, 1), dtype=np.uint8)
        mmY[0:2, 1:3, 0] = 1
        return generators.StratifiedWindowGenerator(
            mmX,
            mmY,
            behavior_idx=0,
            batch_size=8,
            hard_negative_indices=np.array([2, 3]),
            positive_per_batch=1,
            hard_negative_per_batch=3,
            random_negative_per_batch=4,
            seed=seed,
        )

    def test_every_batch_has_requested_composition(self) -> None:
        generator = self.make_generator()
        self.assertEqual(len(generator), 2)
        for indices in generator.indices:
            self.assertEqual(np.count_nonzero(np.isin(indices, [0, 1])), 1)
            self.assertEqual(np.count_nonzero(np.isin(indices, [2, 3])), 3)
            self.assertEqual(np.count_nonzero(np.isin(indices, range(4, 10))), 4)

    def test_sampling_is_reproducible_for_a_fixed_seed(self) -> None:
        first = self.make_generator(seed=9)
        second = self.make_generator(seed=9)
        np.testing.assert_array_equal(first.indices, second.indices)
        first.on_epoch_end()
        second.on_epoch_end()
        np.testing.assert_array_equal(first.indices, second.indices)

    def test_returned_tensors_preserve_window_shapes(self) -> None:
        generator = self.make_generator()
        X, y = generator[0]
        self.assertEqual(X.shape, (8, 4, 1))
        self.assertEqual(y.shape, (8, 4, 1))


if __name__ == "__main__":
    unittest.main()
