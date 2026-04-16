from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


TRAIN_MANIFEST_REQUIRED_KEYS = [
    "behavior",
    "behavior_idx",
    "label_columns",
    "feature_columns",
    "window",
    "artifacts",
    "feature_selection",
    "training_run",
]

ENSEMBLE_MANIFEST_REQUIRED_KEYS = [
    "manifest_type",
    "behavior",
    "behavior_idx",
    "label_columns",
    "feature_columns",
    "window",
    "feature_selection",
    "training",
    "aggregation",
    "members",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return payload


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def require_keys(d: dict[str, Any], keys: list[str], where: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Missing keys {missing} in {where}")


def as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def resolve_recorded_path(value: str | Path, manifest_path: Path) -> Path:
    recorded = as_path(value)
    if recorded.is_absolute():
        if recorded.exists():
            return recorded
        adjacent = manifest_path.parent / recorded.name
        if adjacent.exists():
            return adjacent.resolve()
        return recorded

    relative = (manifest_path.parent / recorded).resolve()
    if relative.exists():
        return relative

    adjacent = (manifest_path.parent / recorded.name).resolve()
    if adjacent.exists():
        return adjacent

    return relative


def load_train_manifest(path: Path) -> dict[str, Any]:
    manifest = load_yaml(path)
    if manifest.get("manifest_type") == "ensemble":
        raise ValueError(
            f"{path} is an ensemble manifest. This command expects a train_manifest.yml file."
        )
    require_keys(manifest, TRAIN_MANIFEST_REQUIRED_KEYS, f"manifest {path}")
    return manifest


def validate_train_manifests(manifests: list[dict[str, Any]], subset: str) -> None:
    if not manifests:
        raise ValueError("At least one manifest is required.")

    behaviors = [m["behavior"] for m in manifests]
    dupes = sorted({b for b in behaviors if behaviors.count(b) > 1})
    if dupes:
        raise ValueError(f"Duplicate behaviors in evaluation set: {dupes}")

    base = manifests[0]
    shared_fields = [
        ("kl_config", "KineLearn config"),
        ("split", "split file"),
        ("label_columns", "label columns"),
    ]
    for field, label in shared_fields:
        base_val = base.get(field)
        for manifest in manifests[1:]:
            if manifest.get(field) != base_val:
                raise ValueError(f"All manifests must share the same {label}.")

    base_window = base["window"]
    for manifest in manifests[1:]:
        if manifest["window"] != base_window:
            raise ValueError("All manifests must share the same window size/stride.")

    if subset in {"train", "val"}:
        base_training = base.get("training", {})
        for manifest in manifests[1:]:
            training = manifest.get("training", {})
            for field in ("val_fraction", "seed"):
                if training.get(field) != base_training.get(field):
                    raise ValueError(
                        f"All manifests must share training.{field} when evaluating '{subset}'."
                    )


def resolve_weights_path(manifest: dict[str, Any], manifest_path: Path) -> Path:
    training_run = manifest.get("training_run", {})
    candidates = [
        training_run.get("evaluation_weights"),
        training_run.get("checkpoint_best_model"),
        training_run.get("checkpoint_interrupted_model"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = resolve_recorded_path(candidate, manifest_path)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No usable weights file found for manifest {manifest_path}."
    )


def load_ensemble_manifest(path: Path) -> dict[str, Any]:
    manifest = load_yaml(path)
    if manifest.get("manifest_type") != "ensemble":
        raise ValueError(f"{path} is not an ensemble manifest.")
    require_keys(manifest, ENSEMBLE_MANIFEST_REQUIRED_KEYS, f"ensemble manifest {path}")

    if manifest["aggregation"].get("method") != "mean_probability":
        raise ValueError(
            f"Unsupported ensemble aggregation method: {manifest['aggregation'].get('method')}"
        )
    members = manifest.get("members") or []
    if not isinstance(members, list) or len(members) < 2:
        raise ValueError(f"Ensemble manifest {path} must list at least two members.")
    return manifest


def inference_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "behavior": manifest["behavior"],
        "behavior_idx": int(manifest["behavior_idx"]),
        "label_columns": list(manifest["label_columns"]),
        "feature_columns": list(manifest["feature_columns"]),
        "window": dict(manifest["window"]),
        "feature_selection": dict(manifest["feature_selection"]),
        "training": {
            "final_zero_fill": bool(
                (manifest.get("training") or {}).get("final_zero_fill", False)
            )
        },
    }


def recusal_stems(manifest: dict[str, Any], *, policy: str = "train_val") -> set[str]:
    resolved = manifest.get("resolved_stems") or {}
    train_stems = {str(stem) for stem in (resolved.get("train") or [])}
    val_stems = {str(stem) for stem in (resolved.get("val") or [])}

    if policy == "train":
        return train_stems
    if policy == "train_val":
        return train_stems | val_stems

    raise ValueError(f"Unsupported recusal policy: {policy}")


def selection_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    training_cfg = dict((manifest.get("training") or {}))
    training_cfg.pop("seed", None)
    training_cfg.pop("val_fraction", None)
    return {
        **inference_signature(manifest),
        "kl_config": manifest.get("kl_config"),
        "training_recipe": training_cfg,
        "focal": dict(manifest.get("focal") or {}),
    }


def validate_ensemble_member_manifests(
    member_manifests: list[dict[str, Any]],
    member_paths: list[Path],
) -> dict[str, Any]:
    if len(member_manifests) < 2:
        raise ValueError(
            "At least two train manifests are required to create an ensemble."
        )

    base_signature = inference_signature(member_manifests[0])
    for manifest, path in zip(member_manifests[1:], member_paths[1:]):
        sig = inference_signature(manifest)
        if sig["behavior"] != base_signature["behavior"]:
            raise ValueError(
                f"Ensemble members must share the same behavior. "
                f"Expected '{base_signature['behavior']}', got '{sig['behavior']}' in {path}."
            )
        for field in (
            "behavior_idx",
            "label_columns",
            "feature_columns",
            "window",
            "feature_selection",
            "training",
        ):
            if sig[field] != base_signature[field]:
                raise ValueError(
                    f"Ensemble members must share the same {field}. "
                    f"Mismatch found in {path}."
                )
    return base_signature


def validate_selection_candidate_manifests(
    member_manifests: list[dict[str, Any]],
    member_paths: list[Path],
) -> dict[str, Any]:
    if len(member_manifests) < 2:
        raise ValueError("At least two compatible candidate manifests are required.")

    base_signature = selection_signature(member_manifests[0])
    for manifest, path in zip(member_manifests[1:], member_paths[1:]):
        sig = selection_signature(manifest)
        for field in (
            "behavior",
            "behavior_idx",
            "label_columns",
            "feature_columns",
            "window",
            "feature_selection",
            "training",
            "kl_config",
            "training_recipe",
            "focal",
        ):
            if sig[field] != base_signature[field]:
                raise ValueError(
                    f"Selection candidates must share the same {field}. "
                    f"Mismatch found in {path}."
                )
    return base_signature


def build_ensemble_manifest_payload(
    member_paths: list[Path],
    member_manifests: list[dict[str, Any]],
    *,
    name: str | None,
) -> dict[str, Any]:
    shared = validate_ensemble_member_manifests(member_manifests, member_paths)
    member_rows = []
    for member_path, member_manifest in zip(member_paths, member_manifests):
        weights_path = resolve_weights_path(member_manifest, member_path)
        member_rows.append(
            {
                "manifest_path": str(member_path.resolve()),
                "run_dir": str(as_path(member_manifest["run_dir"]).resolve()),
                "weights_path": str(weights_path.resolve()),
                "split": member_manifest.get("split"),
                "val_split": member_manifest.get("val_split"),
            }
        )

    return {
        "manifest_type": "ensemble",
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ensemble_name": name,
        **shared,
        "aggregation": {
            "method": "mean_probability",
            "n_members": len(member_rows),
        },
        "members": member_rows,
    }


def load_prediction_source(path: Path) -> dict[str, Any]:
    raw = load_yaml(path)
    if raw.get("manifest_type") == "ensemble":
        ensemble_manifest = load_ensemble_manifest(path)
        member_paths = [
            as_path(member["manifest_path"]).resolve()
            for member in ensemble_manifest["members"]
        ]
        member_manifests = [
            load_train_manifest(member_path) for member_path in member_paths
        ]
        shared = validate_ensemble_member_manifests(member_manifests, member_paths)
        recorded = inference_signature(ensemble_manifest)
        if recorded != shared:
            raise ValueError(
                f"Ensemble manifest {path} no longer matches its member manifests. "
                "Recreate the ensemble manifest from the current members."
            )
        return {
            "manifest_kind": "ensemble",
            "manifest_path": path.resolve(),
            "kl_config": member_manifests[0].get("kl_config"),
            **recorded,
            "aggregation": dict(ensemble_manifest["aggregation"]),
            "members": [
                {
                    "manifest_path": member_path,
                    "manifest": member_manifest,
                    "weights_path": resolve_weights_path(member_manifest, member_path),
                }
                for member_path, member_manifest in zip(member_paths, member_manifests)
            ],
        }

    manifest = load_train_manifest(path)
    return {
        "manifest_kind": "train",
        "manifest_path": path.resolve(),
        "kl_config": manifest.get("kl_config"),
        **inference_signature(manifest),
        "aggregation": {"method": "mean_probability", "n_members": 1},
        "members": [
            {
                "manifest_path": path.resolve(),
                "manifest": manifest,
                "weights_path": resolve_weights_path(manifest, path),
            }
        ],
    }
