from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from KineLearn.core.manifests import recusal_stems
from KineLearn.scripts import eval as eval_script


class _FakeLoadedModel:
    def __init__(self) -> None:
        self.loaded_weights = None

    def load_weights(self, path: str) -> None:
        self.loaded_weights = path


class _FakeBatchModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict_on_batch(self, X: np.ndarray) -> np.ndarray:
        return np.full((X.shape[0], X.shape[1], 1), self.value, dtype=np.float32)


class EvalManifestTests(unittest.TestCase):
    def test_recusal_stems_defaults_to_train_plus_val(self) -> None:
        manifest = {
            "resolved_stems": {
                "train": ["video_a", "video_b"],
                "val": ["video_c"],
                "test": ["video_d"],
            }
        }

        self.assertEqual(recusal_stems(manifest), {"video_a", "video_b", "video_c"})
        self.assertEqual(recusal_stems(manifest, policy="train"), {"video_a", "video_b"})

    def test_build_loaded_model_uses_manifest_model_config(self) -> None:
        fake_model = _FakeLoadedModel()
        manifest = {
            "window": {"size": 90},
            "feature_selection": {"n_input_features": 110},
            "training": {
                "model": {
                    "variant": "conv_bilstm",
                    "conv_frontend": {"filters": [32, 32], "kernel_sizes": [5, 3]},
                }
            },
        }
        weights_path = Path("/tmp/fake.weights.h5")

        with patch.object(eval_script, "tf", object()):
            with patch.object(eval_script, "build_sequence_model", return_value=fake_model) as mock_build:
                model = eval_script.build_loaded_model(manifest, weights_path)

        self.assertIs(model, fake_model)
        mock_build.assert_called_once_with(
            90,
            110,
            model_cfg={
                "variant": "conv_bilstm",
                "conv_frontend": {"filters": [32, 32], "kernel_sizes": [5, 3]},
            },
        )
        self.assertEqual(fake_model.loaded_weights, str(weights_path))

    def test_evaluate_prediction_source_recuses_member_on_train_and_val_stems(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            weights_a = root / "member_a.weights.h5"
            weights_b = root / "member_b.weights.h5"
            weights_a.write_text("a\n")
            weights_b.write_text("b\n")

            mmX_path = root / "test_features.fp32"
            mmY_path = root / "test_labels.u8"
            vids_path = root / "test_vids.npy"
            starts_path = root / "test_starts.npy"

            mmX = np.memmap(mmX_path, mode="w+", dtype="float32", shape=(2, 3, 2))
            mmX[:] = 1.0
            del mmX

            mmY = np.memmap(mmY_path, mode="w+", dtype="uint8", shape=(2, 3, 1))
            mmY[:] = 1
            del mmY

            np.save(vids_path, np.array(["video_a", "video_b"], dtype=object), allow_pickle=True)
            np.save(starts_path, np.array([0, 0], dtype=np.int32), allow_pickle=True)

            eval_manifest = {
                "kl_config": "/tmp/config.yaml",
                "behavior": "genitalia_extension",
                "behavior_idx": 0,
                "label_columns": ["genitalia_extension"],
                "feature_columns": ["feat_1", "feat_2"],
                "window": {"size": 3, "stride": 1},
                "feature_selection": {
                    "include_absolute_coordinates": False,
                    "n_input_features": 2,
                },
                "training": {"final_zero_fill": False, "batch_size": 2},
                "artifacts": {
                    "test": {
                        "count": 2,
                        "X_path": str(mmX_path),
                        "Y_path": str(mmY_path),
                        "vids_path": str(vids_path),
                        "starts_path": str(starts_path),
                        "X_dtype": "float32",
                        "Y_dtype": "uint8",
                        "X_shape": [2, 3, 2],
                        "Y_shape": [2, 3, 1],
                    }
                },
            }
            eval_manifest_path = root / "eval_manifest.yml"
            with open(eval_manifest_path, "w") as f:
                yaml.safe_dump(eval_manifest, f, sort_keys=False)

            source = {
                "manifest_kind": "ensemble",
                "manifest_path": root / "ensemble_manifest.yml",
                "kl_config": "/tmp/config.yaml",
                "behavior": "genitalia_extension",
                "behavior_idx": 0,
                "label_columns": ["genitalia_extension"],
                "feature_columns": ["feat_1", "feat_2"],
                "window": {"size": 3, "stride": 1},
                "feature_selection": {
                    "include_absolute_coordinates": False,
                    "n_input_features": 2,
                },
                "training": {"final_zero_fill": False},
                "aggregation": {"method": "mean_probability", "n_members": 2},
                "members": [
                    {
                        "manifest": {
                            "resolved_stems": {
                                "train": ["video_a"],
                                "val": [],
                            }
                        },
                        "weights_path": weights_a,
                    },
                    {
                        "manifest": {
                            "resolved_stems": {
                                "train": [],
                                "val": ["video_b"],
                            }
                        },
                        "weights_path": weights_b,
                    },
                ],
            }

            models = [_FakeBatchModel(0.2), _FakeBatchModel(0.8)]
            with patch.object(eval_script, "build_loaded_model", side_effect=models):
                frame_df, metric_rows, _error_rows = eval_script.evaluate_prediction_source(
                    source,
                    eval_manifest,
                    eval_manifest_path,
                    subset="test",
                    threshold=0.5,
                    batch_size=2,
                    level="frame",
                    episode_min_frames=1,
                    episode_max_gap=0,
                    episode_overlap_threshold=0.2,
                    ensemble_recusal_policy="train_val",
                )

        self.assertEqual(len(frame_df), 6)
        probs_a = frame_df.loc[frame_df["__stem__"] == "video_a", "prob_genitalia_extension"].tolist()
        probs_b = frame_df.loc[frame_df["__stem__"] == "video_b", "prob_genitalia_extension"].tolist()
        self.assertTrue(np.allclose(probs_a, [0.8, 0.8, 0.8]))
        self.assertTrue(np.allclose(probs_b, [0.2, 0.2, 0.2]))
        self.assertEqual(metric_rows[0]["frame_coverage"], 1.0)
        self.assertEqual(metric_rows[0]["recusal_policy"], "train_val")


if __name__ == "__main__":
    unittest.main()
