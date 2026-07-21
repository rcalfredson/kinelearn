import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from KineLearn.scripts.calculate_features import (
    SCALED_FEATURE_TYPES,
    load_configured_scalers,
    select_feature_family,
)


class ConfiguredScalerTests(unittest.TestCase):
    def test_ignores_stale_coordinate_scaler_entry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            configured = {}
            for feature_type in SCALED_FEATURE_TYPES:
                path = root / f"{feature_type}.pkl"
                joblib.dump({"feature_type": feature_type}, path)
                configured[feature_type] = str(path)
            configured["coordinates"] = str(root / "missing_coordinates.pkl")

            loaded = load_configured_scalers({"scalers": configured})

            self.assertEqual(set(loaded), set(SCALED_FEATURE_TYPES))

    def test_requires_each_standardized_feature_family(self):
        with self.assertRaisesRegex(ValueError, "velocity"):
            load_configured_scalers({"scalers": {}})


class FeatureFamilySelectionTests(unittest.TestCase):
    def test_relational_names_do_not_leak_into_legacy_scaler_inputs(self):
        frame = pd.DataFrame(
            columns=[
                "abdomen_coord_x",
                "abdomen_velocity_y",
                "abdomen_acceleration_x",
                "angle_head_thorax_abdomen",
                "distance_head_thorax",
                "rel_back_legs_angle_asymmetry",
                "rel_back_legs_mean_origin_tip_distance",
                "rel_back_legs_distance_asymmetry",
            ]
        )

        self.assertEqual(
            list(select_feature_family(frame, "angles")),
            ["angle_head_thorax_abdomen"],
        )
        self.assertEqual(
            list(select_feature_family(frame, "distances")),
            ["distance_head_thorax"],
        )
        self.assertEqual(
            list(select_feature_family(frame, "coordinates")),
            ["abdomen_coord_x"],
        )


if __name__ == "__main__":
    unittest.main()
