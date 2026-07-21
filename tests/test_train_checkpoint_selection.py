from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from KineLearn.scripts import train as train_script
from KineLearn.scripts.train import (
    EpisodeCheckpointSelector,
    checkpoint_candidate_rank,
    checkpoint_thresholds,
    resolve_checkpoint_selection_config,
    resolve_execution_settings,
    select_checkpoint_candidate,
)


class _FakeEpochModel:
    def __init__(self, predictions: np.ndarray) -> None:
        self.predictions = predictions
        self.save_count = 0

    def predict_on_batch(self, X: np.ndarray) -> np.ndarray:
        return np.broadcast_to(
            self.predictions,
            (X.shape[0], *self.predictions.shape[1:]),
        ).copy()

    def save_weights(self, path: str) -> None:
        Path(path).write_text(f"saved {self.save_count}\n")
        self.save_count += 1


class CheckpointSelectionConfigTests(unittest.TestCase):
    def test_default_grid_is_inclusive_and_stable(self) -> None:
        thresholds = checkpoint_thresholds(
            {"thresholds": {"start": 0.35, "stop": 0.38, "step": 0.01}}
        )
        self.assertEqual(thresholds, [0.35, 0.36, 0.37, 0.38])

    def test_explicit_thresholds_are_sorted_and_deduplicated(self) -> None:
        self.assertEqual(
            checkpoint_thresholds({"thresholds": [0.6, 0.4, 0.6]}),
            [0.4, 0.6],
        )

    def test_invalid_checkpoint_settings_fail_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "step must be positive"):
            checkpoint_thresholds(
                {"thresholds": {"start": 0.4, "stop": 0.6, "step": 0.0}}
            )
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            checkpoint_thresholds({"thresholds": [0.0, 0.5]})
        with self.assertRaisesRegex(ValueError, "episode_min_frames"):
            resolve_checkpoint_selection_config(
                {"checkpoint_selection": {"enabled": True, "episode_min_frames": 0}}
            )

    def test_disabled_config_is_normalized_for_manifest_recording(self) -> None:
        cfg = resolve_checkpoint_selection_config({})
        self.assertFalse(cfg["enabled"])
        self.assertEqual(cfg["metric"], "episode_f1")
        self.assertEqual(cfg["thresholds"][0], 0.35)
        self.assertEqual(cfg["thresholds"][-1], 0.75)


class ExecutionSettingsTests(unittest.TestCase):
    def test_defaults_preserve_existing_batching_behavior(self) -> None:
        self.assertEqual(resolve_execution_settings({}), (8, 1, 8))
        self.assertEqual(
            resolve_execution_settings({"batch_size": 16}),
            (16, 1, 16),
        )

    def test_independent_execution_and_inference_settings(self) -> None:
        self.assertEqual(
            resolve_execution_settings(
                {
                    "batch_size": 8,
                    "steps_per_execution": 32,
                    "inference_batch_size": 256,
                }
            ),
            (8, 32, 256),
        )

    def test_nonpositive_or_noninteger_settings_are_rejected(self) -> None:
        for name, value in [
            ("batch_size", 0),
            ("steps_per_execution", -1),
            ("inference_batch_size", 0),
            ("steps_per_execution", True),
            ("inference_batch_size", 8.5),
        ]:
            with self.subTest(name=name, value=value):
                with self.assertRaisesRegex(ValueError, name):
                    resolve_execution_settings({name: value})


class CheckpointCandidateRankingTests(unittest.TestCase):
    @staticmethod
    def candidate(
        *, threshold: float, f1: float, precision: float, recall: float
    ) -> dict:
        return {
            "threshold": threshold,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

    def test_f1_is_primary_selection_metric(self) -> None:
        balanced = self.candidate(
            threshold=0.5, f1=0.80, precision=0.80, recall=0.80
        )
        higher_f1 = self.candidate(
            threshold=0.6, f1=0.81, precision=0.90, recall=0.74
        )
        self.assertIs(select_checkpoint_candidate([balanced, higher_f1]), higher_f1)

    def test_minimum_precision_recall_breaks_f1_ties(self) -> None:
        imbalanced = self.candidate(
            threshold=0.5, f1=0.80, precision=0.95, recall=0.70
        )
        balanced = self.candidate(
            threshold=0.6, f1=0.80, precision=0.82, recall=0.78
        )
        self.assertGreater(
            checkpoint_candidate_rank(balanced), checkpoint_candidate_rank(imbalanced)
        )
        self.assertIs(select_checkpoint_candidate([imbalanced, balanced]), balanced)

    def test_threshold_closest_to_half_is_final_tie_breaker(self) -> None:
        farther = self.candidate(
            threshold=0.7, f1=0.80, precision=0.80, recall=0.80
        )
        closer = self.candidate(
            threshold=0.55, f1=0.80, precision=0.80, recall=0.80
        )
        self.assertIs(select_checkpoint_candidate([farther, closer]), closer)

    def test_lower_threshold_breaks_an_exact_distance_tie(self) -> None:
        lower = self.candidate(
            threshold=0.45, f1=0.80, precision=0.80, recall=0.80
        )
        upper = self.candidate(
            threshold=0.55, f1=0.80, precision=0.80, recall=0.80
        )
        self.assertIs(select_checkpoint_candidate([upper, lower]), lower)

    def test_empty_candidate_list_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            select_checkpoint_candidate([])


@unittest.skipIf(train_script.tf is None, "TensorFlow is not installed")
class EpisodeCheckpointSelectorTests(unittest.TestCase):
    def test_selector_keeps_best_epoch_and_writes_audit_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            checkpoint_path = output_dir / "best_model.weights.h5"
            mmX = np.ones((1, 6, 2), dtype=np.float32)
            mmY = np.array([[[0], [1], [1], [1], [0], [0]]], dtype=np.uint8)
            vids = np.array(["video_a"], dtype=object)
            starts = np.array([0], dtype=np.int32)
            selector = EpisodeCheckpointSelector(
                mmX=mmX,
                mmY=mmY,
                vids=vids,
                starts=starts,
                behavior="back_leg_together",
                behavior_idx=0,
                window_size=6,
                batch_size=1,
                selection_cfg={
                    "metric": "episode_f1",
                    "thresholds": [0.5, 0.9],
                    "episode_min_frames": 2,
                    "episode_max_gap": 0,
                    "episode_overlap_threshold": 0.2,
                },
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
            )
            model = _FakeEpochModel(
                np.array([[[0.1], [0.8], [0.8], [0.8], [0.1], [0.1]]])
            )
            selector.set_model(model)
            first_logs = {}
            selector.on_epoch_end(0, first_logs)

            model.predictions = np.full((1, 6, 1), 0.1, dtype=np.float32)
            second_logs = {}
            selector.on_epoch_end(1, second_logs)

            self.assertEqual(selector.best_candidate["epoch"], 1)
            self.assertEqual(selector.best_candidate["threshold"], 0.5)
            self.assertEqual(selector.best_candidate["f1"], 1.0)
            self.assertEqual(model.save_count, 1)
            self.assertEqual(first_logs["val_selected_episode_f1"], 1.0)
            self.assertEqual(second_logs["val_selected_episode_f1"], 0.0)
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(selector.candidates_path.exists())
            self.assertTrue(selector.summary_path.exists())
            self.assertTrue(selector.predictions_path.exists())


if __name__ == "__main__":
    unittest.main()
