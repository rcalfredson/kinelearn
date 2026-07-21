import unittest

import numpy as np
import pandas as pd

from KineLearn.core.features import (
    compute_bilateral_tip_features,
    compute_lagged_relational_features,
    compute_relational_features,
    select_behavior_feature_columns,
)


GROUP_CONFIG = {
    "type": "bilateral_tips",
    "behavior": "back_leg_together",
    "origin": "abdomen",
    "axis": ["thorax", "abdomen"],
    "left_tip": "left_back_leg_tip",
    "right_tip": "right_back_leg_tip",
}


def pose_frame(points):
    return pd.DataFrame(
        {
            f"{point}_{axis}": [coords[i]]
            for point, coords in points.items()
            for i, axis in enumerate(("x", "y"))
        }
    )


class BilateralRelationalFeatureTests(unittest.TestCase):
    def test_symmetric_pose_has_expected_body_frame_geometry(self):
        frame = pose_frame(
            {
                "head": (-1.0, 0.0),
                "thorax": (0.0, 0.0),
                "abdomen": (1.0, 0.0),
                "left_back_leg_tip": (1.0, 1.0),
                "right_back_leg_tip": (1.0, 1.0),
            }
        )

        result = compute_bilateral_tip_features(
            frame, "back_legs", GROUP_CONFIG, ["head", "thorax"]
        ).iloc[0]

        expected = {
            "rel_back_legs_left_posterior": 0.0,
            "rel_back_legs_left_lateral": 1.0,
            "rel_back_legs_right_posterior": 0.0,
            "rel_back_legs_right_lateral": 1.0,
            "rel_back_legs_tip_separation": 0.0,
            "rel_back_legs_midpoint_posterior": 0.0,
            "rel_back_legs_midpoint_lateral": 1.0,
            "rel_back_legs_mean_origin_tip_distance": 1.0,
            "rel_back_legs_distance_asymmetry": 0.0,
            "rel_back_legs_left_signed_angle": 0.5,
            "rel_back_legs_right_signed_angle": 0.5,
            "rel_back_legs_angle_asymmetry": 0.0,
        }
        for column, value in expected.items():
            self.assertAlmostEqual(result[column], value)

    def test_features_are_invariant_to_translation_rotation_and_scale(self):
        points = {
            "head": (-1.0, 0.0),
            "thorax": (0.0, 0.0),
            "abdomen": (1.0, 0.0),
            "left_back_leg_tip": (1.2, 1.1),
            "right_back_leg_tip": (1.8, -0.7),
        }
        angle = 0.73
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        transformed = {
            name: tuple(3.7 * rotation @ np.asarray(coords) + np.array([14.0, -9.0]))
            for name, coords in points.items()
        }

        original_features = compute_bilateral_tip_features(
            pose_frame(points), "back_legs", GROUP_CONFIG, ["head", "thorax"]
        )
        transformed_features = compute_bilateral_tip_features(
            pose_frame(transformed), "back_legs", GROUP_CONFIG, ["head", "thorax"]
        )

        np.testing.assert_allclose(
            original_features.to_numpy(),
            transformed_features.to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_zero_body_length_yields_nan_normalized_geometry(self):
        frame = pose_frame(
            {
                "head": (0.0, 0.0),
                "thorax": (0.0, 0.0),
                "abdomen": (1.0, 0.0),
                "left_back_leg_tip": (1.0, 1.0),
                "right_back_leg_tip": (1.0, -1.0),
            }
        )
        result = compute_bilateral_tip_features(
            frame, "back_legs", GROUP_CONFIG, ["head", "thorax"]
        )
        self.assertTrue(result.isna().all(axis=None))


class LaggedRelationalFeatureTests(unittest.TestCase):
    def test_dynamic_features_remain_translation_rotation_and_scale_invariant(self):
        poses = [
            {
                "head": (-1.0, 0.0),
                "thorax": (0.0, 0.0),
                "abdomen": (1.0, 0.0),
                "left_back_leg_tip": left_tip,
                "right_back_leg_tip": right_tip,
            }
            for left_tip, right_tip in [
                ((1.2, 1.1), (1.8, -0.7)),
                ((1.4, 0.8), (1.6, -0.9)),
                ((1.7, 0.5), (1.3, -1.0)),
            ]
        ]
        angle = 0.73
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        transformed_poses = [
            {
                name: tuple(
                    3.7 * rotation @ np.asarray(coords) + np.array([14.0, -9.0])
                )
                for name, coords in points.items()
            }
            for points in poses
        ]
        features_cfg = {
            "body_length_pts": ["head", "thorax"],
            "relational": {
                "back_legs": {
                    **GROUP_CONFIG,
                    "dynamics": {"lags": [1, 2]},
                }
            },
        }

        original = compute_relational_features(
            pd.concat([pose_frame(points) for points in poses], ignore_index=True),
            features_cfg,
        )
        transformed = compute_relational_features(
            pd.concat(
                [pose_frame(points) for points in transformed_poses],
                ignore_index=True,
            ),
            features_cfg,
        )

        np.testing.assert_allclose(
            original.to_numpy(),
            transformed.to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_causal_deltas_use_configured_lags_and_zero_leading_boundaries(self):
        frames = []
        for left_x in [1.0, 1.25, 1.5, 1.75]:
            frames.append(
                pose_frame(
                    {
                        "head": (-1.0, 0.0),
                        "thorax": (0.0, 0.0),
                        "abdomen": (1.0, 0.0),
                        "left_back_leg_tip": (left_x, 1.0),
                        "right_back_leg_tip": (1.0, -1.0),
                    }
                )
            )
        df_xy = pd.concat(frames, ignore_index=True)
        group_cfg = {
            **GROUP_CONFIG,
            "dynamics": {
                "lags": [1, 2],
                "features": ["left_posterior"],
            },
        }

        result = compute_relational_features(
            df_xy,
            {
                "body_length_pts": ["head", "thorax"],
                "relational": {"back_legs": group_cfg},
            },
        )

        np.testing.assert_allclose(
            result["rel_back_legs_left_posterior_delta_lag1"],
            [0.0, 0.25, 0.25, 0.25],
        )
        np.testing.assert_allclose(
            result["rel_back_legs_left_posterior_delta_lag2"],
            [0.0, 0.0, 0.5, 0.5],
        )
        self.assertNotIn("rel_back_legs_right_posterior_delta_lag1", result)

    def test_signed_angle_delta_wraps_across_pi_boundary(self):
        static = pd.DataFrame(
            {
                "rel_back_legs_left_signed_angle": [0.9, -0.9],
                "rel_back_legs_distance_asymmetry": [0.1, 0.2],
            }
        )
        result = compute_lagged_relational_features(
            static,
            "back_legs",
            {
                "dynamics": {
                    "lags": [1],
                    "features": [
                        "left_signed_angle",
                        "distance_asymmetry",
                    ],
                }
            },
        )

        self.assertAlmostEqual(
            result.loc[1, "rel_back_legs_left_signed_angle_delta_lag1"],
            0.2,
        )
        self.assertAlmostEqual(
            result.loc[1, "rel_back_legs_distance_asymmetry_delta_lag1"],
            0.1,
        )

    def test_missing_current_pose_is_not_replaced_by_boundary_zero(self):
        static = pd.DataFrame(
            {"rel_back_legs_tip_separation": [np.nan, 0.5, 0.7]}
        )
        result = compute_lagged_relational_features(
            static,
            "back_legs",
            {"dynamics": {"lags": [2]}},
        )

        self.assertTrue(
            np.isnan(result.loc[0, "rel_back_legs_tip_separation_delta_lag2"])
        )
        self.assertEqual(
            result.loc[1, "rel_back_legs_tip_separation_delta_lag2"], 0.0
        )
        self.assertTrue(
            np.isnan(result.loc[2, "rel_back_legs_tip_separation_delta_lag2"])
        )

    def test_invalid_lag_and_feature_settings_fail_loudly(self):
        static = pd.DataFrame({"rel_back_legs_tip_separation": [0.5]})
        for dynamics, message in [
            ({"lags": [0]}, "positive integers"),
            ({"lags": [1, 1]}, "duplicate lag"),
            ({"lags": [1], "features": ["unknown"]}, "unknown features"),
        ]:
            with self.subTest(dynamics=dynamics):
                with self.assertRaisesRegex(ValueError, message):
                    compute_lagged_relational_features(
                        static,
                        "back_legs",
                        {"dynamics": dynamics},
                    )


class BehaviorFeatureSelectionTests(unittest.TestCase):
    def test_only_keeps_relational_groups_assigned_to_behavior(self):
        columns = [
            "legacy_feature",
            "rel_back_legs_tip_separation",
            "rel_genitalia_extension_posterior",
            "rel_shared_posture_value",
        ]
        features_cfg = {
            "relational": {
                "back_legs": {"behavior": "back_leg_together"},
                "genitalia_extension": {"behavior": "genitalia_extension"},
                "shared_posture": {},
            }
        }

        selected = select_behavior_feature_columns(
            columns, features_cfg, "back_leg_together"
        )

        self.assertEqual(
            selected,
            [
                "legacy_feature",
                "rel_back_legs_tip_separation",
                "rel_shared_posture_value",
            ],
        )

    def test_relational_columns_are_opt_in(self):
        selected = select_behavior_feature_columns(
            ["legacy_feature", "rel_back_legs_tip_separation"],
            {},
            "back_leg_together",
        )
        self.assertEqual(selected, ["legacy_feature"])


if __name__ == "__main__":
    unittest.main()
