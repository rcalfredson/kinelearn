from __future__ import annotations

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from KineLearn.core import models
from KineLearn.core.models import residual_tcn_receptive_field


class ResidualTcnReceptiveFieldTests(unittest.TestCase):
    def test_proposed_architecture_covers_61_frames(self) -> None:
        self.assertEqual(
            residual_tcn_receptive_field(
                3,
                [1, 2, 4, 8],
                convolutions_per_block=2,
            ),
            61,
        )

    def test_invalid_receptive_field_settings_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "kernel_size"):
            residual_tcn_receptive_field(0, [1, 2])
        with self.assertRaisesRegex(ValueError, "dilations"):
            residual_tcn_receptive_field(3, [])
        with self.assertRaisesRegex(ValueError, "dilations"):
            residual_tcn_receptive_field(3, [1, 0])
        with self.assertRaisesRegex(ValueError, "convolutions_per_block"):
            residual_tcn_receptive_field(3, [1, 2], convolutions_per_block=0)


@unittest.skipIf(models.tf is None, "TensorFlow is not installed")
class ResidualTcnModelTests(unittest.TestCase):
    def test_model_preserves_time_axis_and_emits_one_probability(self) -> None:
        model = models.build_sequence_model(
            60,
            122,
            model_cfg={
                "variant": "residual_tcn",
                "residual_tcn": {
                    "channels": 64,
                    "kernel_size": 3,
                    "dilations": [1, 2, 4, 8],
                    "convolutions_per_block": 2,
                    "dropout": 0.15,
                    "activation": "relu",
                },
            },
        )

        self.assertEqual(model.input_shape, (None, 60, 122))
        self.assertEqual(model.output_shape, (None, 60, 1))
        self.assertEqual(model.name, "keypoint_residual_tcn")
        self.assertFalse(any(isinstance(layer, models.LSTM) for layer in model.layers))

        projection = model.get_layer("tcn_input_projection")
        self.assertEqual(projection.filters, 64)
        self.assertEqual(projection.kernel_size, (1,))

        observed_dilations = [
            model.get_layer(f"tcn_block_{block}_conv_1").dilation_rate
            for block in range(1, 5)
        ]
        self.assertEqual(observed_dilations, [(1,), (2,), (4,), (8,)])

    def test_invalid_model_settings_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "channels"):
            models.build_keypoint_residual_tcn(60, 122, channels=0)
        with self.assertRaisesRegex(ValueError, "dropout"):
            models.build_keypoint_residual_tcn(60, 122, dropout=1.0)


if __name__ == "__main__":
    unittest.main()
