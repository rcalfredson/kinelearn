from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from KineLearn.scripts.compare_sweeps import (
    load_batch,
    pairwise_deltas,
    validate_compatibility,
)
from KineLearn.scripts.select_threshold_map import select_threshold_rows


class ThresholdMapSelectionTests(unittest.TestCase):
    def test_selects_best_threshold_with_balanced_tie_breaking(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "outer_split": "outer_seed0",
                    "inner_split": "inner_seed0",
                    "threshold": 0.4,
                    "f1": 0.8,
                    "precision": 0.95,
                    "recall": 0.7,
                },
                {
                    "outer_split": "outer_seed0",
                    "inner_split": "inner_seed0",
                    "threshold": 0.6,
                    "f1": 0.8,
                    "precision": 0.82,
                    "recall": 0.78,
                },
            ]
        )

        selected = select_threshold_rows(frame)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["outer_id"], "outer_seed0")
        self.assertEqual(selected.iloc[0]["inner_seed"], "0")
        self.assertEqual(selected.iloc[0]["threshold"], 0.6)


class SweepComparisonTests(unittest.TestCase):
    def write_batch(
        self,
        root: Path,
        *,
        label: str,
        f1_values: tuple[float, float],
        threshold_mode: str,
    ) -> Path:
        sweep_dir = root / f"{label}_sweep"
        eval_dir = root / f"{label}_eval"
        split_dir = sweep_dir / "splits"
        split_dir.mkdir(parents=True)
        eval_dir.mkdir()
        rows = []
        for outer_seed, f1 in enumerate(f1_values):
            outer_dir = split_dir / f"outer_seed{outer_seed}"
            outer_dir.mkdir()
            split_path = outer_dir / "train_test_split.yaml"
            val_path = outer_dir / "train_val_split_seed0.yaml"
            split_path.write_text(
                yaml.safe_dump({"train": ["a", "b"], "test": [f"t{outer_seed}"]})
            )
            val_path.write_text(yaml.safe_dump({"train": ["a"], "val": ["b"]}))
            rows.append(
                {
                    "outer_id": f"outer_seed{outer_seed}",
                    "outer_seed": outer_seed,
                    "inner_seed": 0,
                    "behavior": "back_leg_together",
                    "subset": "test",
                    "level": "episode",
                    "eval_returncode": 0,
                    "f1": f1,
                    "precision": f1 - 0.05,
                    "recall": f1 + 0.05,
                    "threshold": 0.5,
                    "split_path": str(split_path),
                    "val_split_path": str(val_path),
                }
            )
        pd.DataFrame(rows).to_csv(eval_dir / "batch_eval_summary.csv", index=False)
        config = {
            "sweep_dir": str(sweep_dir),
            "threshold_mode": threshold_mode,
            "threshold": 0.5 if threshold_mode == "fixed" else None,
            "episode_matching_method": "one_to_one_max_cardinality",
            "episode_overlap_denominator": "predicted_episode_length",
        }
        (eval_dir / "batch_eval_config.yml").write_text(
            yaml.safe_dump(config, sort_keys=False)
        )
        return eval_dir

    def test_loads_policies_and_computes_paired_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            historical_dir = self.write_batch(
                root,
                label="historical",
                f1_values=(0.6, 0.7),
                threshold_mode="external_map",
            )
            new_dir = self.write_batch(
                root,
                label="new",
                f1_values=(0.7, 0.9),
                threshold_mode="selected_checkpoint",
            )
            historical, historical_meta = load_batch(
                "historical",
                historical_dir,
                checkpoint_policy="val_loss",
                behavior=None,
            )
            new, new_meta = load_batch(
                "new",
                new_dir,
                checkpoint_policy="episode_f1",
                behavior=None,
            )

            validate_compatibility(
                [historical, new], [historical_meta, new_meta]
            )
            deltas, summary = pairwise_deltas(
                [("historical", historical), ("new", new)]
            )

            self.assertEqual(
                historical_meta["threshold_policy"],
                "validation_selected_posthoc_map",
            )
            self.assertEqual(
                new_meta["threshold_policy"],
                "validation_selected_from_checkpoint_training",
            )
            self.assertEqual(deltas["delta_f1"].round(6).tolist(), [0.1, 0.2])
            self.assertAlmostEqual(summary.iloc[0]["mean_delta_f1"], 0.15)
            self.assertEqual(summary.iloc[0]["wins_f1"], 2)


if __name__ == "__main__":
    unittest.main()
