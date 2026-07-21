import numpy as np
import pandas as pd

from KineLearn.core.geometry import compute_angle, compute_distance

# Path: src/KineLearn/core/features.py


RELATIONAL_FEATURE_PREFIX = "rel_"
_CIRCULAR_RELATIONAL_SUFFIXES = {"left_signed_angle", "right_signed_angle"}


def _normalized_vector(dx, dy, scale, *, epsilon):
    """Return component arrays normalized by a positive per-frame scale."""
    scale = np.asarray(scale, dtype=float)
    valid = np.isfinite(scale) & (np.abs(scale) > epsilon)
    x = np.full(scale.shape, np.nan, dtype=float)
    y = np.full(scale.shape, np.nan, dtype=float)
    np.divide(np.asarray(dx, dtype=float), scale, out=x, where=valid)
    np.divide(np.asarray(dy, dtype=float), scale, out=y, where=valid)
    return x, y


def compute_bilateral_tip_features(df_xy, group_name, group_cfg, default_scale_pts):
    """Compute body-centric static geometry for a bilateral pair of tracked tips."""
    origin = group_cfg["origin"]
    anterior, posterior = group_cfg["axis"]
    left_tip = group_cfg["left_tip"]
    right_tip = group_cfg["right_tip"]
    scale_pts = group_cfg.get("body_length_pts", default_scale_pts)
    if len(scale_pts) != 2:
        raise ValueError(
            f"Relational group '{group_name}' body_length_pts must contain two points."
        )
    epsilon = float(group_cfg.get("epsilon", 1e-8))
    if epsilon <= 0:
        raise ValueError(f"Relational group '{group_name}' epsilon must be positive.")

    scale = compute_distance(df_xy, scale_pts[0], scale_pts[1]).to_numpy(dtype=float)
    axis_x = (df_xy[f"{posterior}_x"] - df_xy[f"{anterior}_x"]).to_numpy(dtype=float)
    axis_y = (df_xy[f"{posterior}_y"] - df_xy[f"{anterior}_y"]).to_numpy(dtype=float)
    axis_norm = np.hypot(axis_x, axis_y)
    unit_x, unit_y = _normalized_vector(axis_x, axis_y, axis_norm, epsilon=epsilon)

    left_x, left_y = _normalized_vector(
        df_xy[f"{left_tip}_x"] - df_xy[f"{origin}_x"],
        df_xy[f"{left_tip}_y"] - df_xy[f"{origin}_y"],
        scale,
        epsilon=epsilon,
    )
    right_x, right_y = _normalized_vector(
        df_xy[f"{right_tip}_x"] - df_xy[f"{origin}_x"],
        df_xy[f"{right_tip}_y"] - df_xy[f"{origin}_y"],
        scale,
        epsilon=epsilon,
    )

    # The lateral basis is the counter-clockwise perpendicular (-y, x).
    left_posterior = left_x * unit_x + left_y * unit_y
    left_lateral = -left_x * unit_y + left_y * unit_x
    right_posterior = right_x * unit_x + right_y * unit_y
    right_lateral = -right_x * unit_y + right_y * unit_x

    left_distance = np.hypot(left_x, left_y)
    right_distance = np.hypot(right_x, right_y)
    distance_sum = left_distance + right_distance
    distance_asymmetry = np.full(distance_sum.shape, np.nan, dtype=float)
    np.divide(
        np.abs(left_distance - right_distance),
        distance_sum + epsilon,
        out=distance_asymmetry,
        where=np.isfinite(distance_sum),
    )

    left_angle = np.arctan2(left_lateral, left_posterior)
    right_angle = np.arctan2(right_lateral, right_posterior)
    left_angle[left_distance <= epsilon] = np.nan
    right_angle[right_distance <= epsilon] = np.nan
    angle_error = np.arctan2(
        np.sin(left_angle - right_angle),
        np.cos(left_angle - right_angle),
    )

    prefix = f"{RELATIONAL_FEATURE_PREFIX}{group_name}_"
    return pd.DataFrame(
        {
            f"{prefix}left_posterior": left_posterior,
            f"{prefix}left_lateral": left_lateral,
            f"{prefix}right_posterior": right_posterior,
            f"{prefix}right_lateral": right_lateral,
            f"{prefix}tip_separation": np.hypot(left_x - right_x, left_y - right_y),
            f"{prefix}midpoint_posterior": (left_posterior + right_posterior) / 2.0,
            f"{prefix}midpoint_lateral": (left_lateral + right_lateral) / 2.0,
            f"{prefix}mean_origin_tip_distance": distance_sum / 2.0,
            f"{prefix}distance_asymmetry": distance_asymmetry,
            f"{prefix}left_signed_angle": left_angle / np.pi,
            f"{prefix}right_signed_angle": right_angle / np.pi,
            f"{prefix}angle_asymmetry": np.abs(angle_error) / np.pi,
        },
        index=df_xy.index,
    )


def _resolve_relational_dynamics(group_name, group_cfg, available_suffixes):
    """Validate and normalize one relational group's optional lag settings."""
    dynamics_cfg = group_cfg.get("dynamics")
    if dynamics_cfg is None:
        return [], []
    if not isinstance(dynamics_cfg, dict):
        raise ValueError(
            f"Relational group '{group_name}' dynamics must be a mapping."
        )
    if not dynamics_cfg.get("enabled", True):
        return [], []

    raw_lags = dynamics_cfg.get("lags")
    if not isinstance(raw_lags, (list, tuple)) or not raw_lags:
        raise ValueError(
            f"Relational group '{group_name}' dynamics.lags must be a non-empty list."
        )
    lags = []
    for lag in raw_lags:
        if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag <= 0:
            raise ValueError(
                f"Relational group '{group_name}' dynamics.lags must contain "
                "positive integers."
            )
        lag = int(lag)
        if lag in lags:
            raise ValueError(
                f"Relational group '{group_name}' dynamics.lags contains duplicate "
                f"lag {lag}."
            )
        lags.append(lag)

    selected_suffixes = dynamics_cfg.get("features")
    if selected_suffixes is None:
        selected_suffixes = list(available_suffixes)
    if not isinstance(selected_suffixes, (list, tuple)) or not selected_suffixes:
        raise ValueError(
            f"Relational group '{group_name}' dynamics.features must be a "
            "non-empty list when provided."
        )
    if any(not isinstance(suffix, str) for suffix in selected_suffixes):
        raise ValueError(
            f"Relational group '{group_name}' dynamics.features must contain strings."
        )
    if len(set(selected_suffixes)) != len(selected_suffixes):
        raise ValueError(
            f"Relational group '{group_name}' dynamics.features contains duplicates."
        )
    unknown = sorted(set(selected_suffixes) - set(available_suffixes))
    if unknown:
        raise ValueError(
            f"Relational group '{group_name}' dynamics.features contains unknown "
            f"features: {unknown}"
        )
    return lags, list(selected_suffixes)


def compute_lagged_relational_features(static_features, group_name, group_cfg):
    """Compute causal deltas from one group's body-centric static features."""
    prefix = f"{RELATIONAL_FEATURE_PREFIX}{group_name}_"
    group_columns = [
        column for column in static_features.columns if column.startswith(prefix)
    ]
    available_suffixes = [column[len(prefix) :] for column in group_columns]
    lags, selected_suffixes = _resolve_relational_dynamics(
        group_name, group_cfg, available_suffixes
    )
    if not lags:
        return pd.DataFrame(index=static_features.index)

    dynamic_features = {}
    for lag in lags:
        for suffix in selected_suffixes:
            column = f"{prefix}{suffix}"
            current = static_features[column].astype(float)
            previous = current.shift(lag)
            delta = current - previous
            if suffix in _CIRCULAR_RELATIONAL_SUFFIXES:
                delta = np.arctan2(
                    np.sin(np.pi * delta),
                    np.cos(np.pi * delta),
                ) / np.pi

            # A missing prior frame at the start of a video means no observed
            # change yet. Preserve NaN when the current pose itself is missing.
            boundary = np.arange(len(current)) < lag
            delta = pd.Series(delta, index=static_features.index, dtype=float)
            delta.loc[boundary & current.notna().to_numpy()] = 0.0
            dynamic_features[f"{column}_delta_lag{lag}"] = delta

    return pd.DataFrame(dynamic_features, index=static_features.index)


def compute_relational_features(df_xy, features_cfg):
    """Compute all enabled config-defined relational feature groups."""
    groups = features_cfg.get("relational", {}) or {}
    frames = []
    default_scale_pts = features_cfg.get("body_length_pts", ["head", "thorax"])
    for group_name, group_cfg in groups.items():
        if not group_cfg.get("enabled", True):
            continue
        group_type = group_cfg.get("type")
        if group_type == "bilateral_tips":
            static_features = compute_bilateral_tip_features(
                df_xy,
                group_name,
                group_cfg,
                default_scale_pts,
            )
            frames.extend(
                [
                    static_features,
                    compute_lagged_relational_features(
                        static_features,
                        group_name,
                        group_cfg,
                    ),
                ]
            )
        else:
            raise ValueError(
                f"Unsupported relational feature type for group '{group_name}': {group_type!r}"
            )
    if not frames:
        return pd.DataFrame(index=df_xy.index)
    return pd.concat(frames, axis=1)


def select_behavior_feature_columns(columns, features_cfg, behavior):
    """Keep legacy features plus relational groups assigned to this behavior."""
    groups = features_cfg.get("relational", {}) or {}
    allowed_prefixes = []
    for group_name, group_cfg in groups.items():
        if not group_cfg.get("enabled", True):
            continue
        assigned = group_cfg.get("behaviors", group_cfg.get("behavior"))
        if isinstance(assigned, str):
            assigned = [assigned]
        if assigned is None or behavior in assigned:
            allowed_prefixes.append(f"{RELATIONAL_FEATURE_PREFIX}{group_name}_")

    return [
        col
        for col in columns
        if not col.startswith(RELATIONAL_FEATURE_PREFIX)
        or any(col.startswith(prefix) for prefix in allowed_prefixes)
    ]


def extract_features(dlc_file, kl_config):
    """
    Extract motion and geometry features from a DeepLabCut CSV file using parameters from KineLearn config.

    Parameters
    ----------
    dlc_file : Path or str
        Path to the DeepLabCut CSV file.
    kl_config : dict
        Parsed KineLearn YAML configuration (contains feature definitions).

    Returns
    -------
    df_combined : pd.DataFrame
        All derived features including coordinates, relative, velocity, acceleration, angles, distances.
    df_xy : pd.DataFrame
        Absolute X/Y coordinates.
    df_p : pd.DataFrame
        Likelihood / probability columns.
    """
    df = pd.read_csv(dlc_file)
    df.columns = df.columns.str.strip()

    # --- Basic column parsing ---
    keypoints = sorted(set(col[:-2] for col in df.columns if col.endswith("_x")))
    xy_cols = [f"{kp}_{ax}" for kp in keypoints for ax in ("x", "y")]
    p_cols = [f"{kp}_p" for kp in keypoints]

    df_xy = df[xy_cols].copy().apply(pd.to_numeric, errors="coerce")
    df_p = df[p_cols].copy().apply(pd.to_numeric, errors="coerce")
    df_p.columns = [col[:-2] + "_probability" for col in df_p.columns]

    # --- Load geometric parameters from config ---
    features_cfg = kl_config["features"]
    ref_pt = features_cfg.get("ref_pt", "thorax")
    bl_pts = features_cfg.get("body_length_pts", ["head", "thorax"])

    # --- Compute body length normalization vector ---
    body_length = compute_distance(df_xy, bl_pts[0], bl_pts[1]).to_numpy()

    # --- Compute relative positions (normalized to ref_pt & body_length) ---
    df_relative = df_xy.copy()
    for col in df_xy.columns:
        if "_x" in col:
            df_relative[col] = (df_xy[col] - df_xy[f"{ref_pt}_x"]) / body_length
        elif "_y" in col:
            df_relative[col] = (df_xy[col] - df_xy[f"{ref_pt}_y"]) / body_length

    df_relative.columns = [
        col.replace("_x", "_coord_x").replace("_y", "_coord_y")
        for col in df_relative.columns
    ]

    # --- Compute velocity & acceleration (framewise differences) ---
    df_velocity = df_xy.diff().fillna(0) / body_length[:, np.newaxis]
    df_velocity.columns = [
        col.replace("_x", "_velocity_x").replace("_y", "_velocity_y")
        for col in df_velocity.columns
    ]

    df_acceleration = df_velocity.diff().fillna(0)
    df_acceleration.columns = [
        col.replace("_velocity_x", "_acceleration_x").replace(
            "_velocity_y", "_acceleration_y"
        )
        for col in df_velocity.columns
    ]

    # --- Compute angles dynamically ---
    angle_defs = features_cfg.get("angles", [])
    angle_features = {}
    for pts in angle_defs:
        a, b, c = pts
        name = f"angle_{a}_{b}_{c}"
        angle_features[name] = compute_angle(df_xy, a, b, c)
    df_angles = pd.DataFrame(angle_features)

    # --- Compute distances dynamically ---
    dist_defs = features_cfg.get("distances", [])
    dist_features = {}
    for p1, p2 in dist_defs:
        name = f"distance_{p1}_{p2}"
        dist_features[name] = compute_distance(df_xy, p1, p2)
    df_distances = pd.DataFrame(dist_features)

    # --- Compute normalized, behavior-scoped relational geometry ---
    df_relational = compute_relational_features(df_xy, features_cfg)

    # --- Combine all derived features ---
    df_derived = pd.concat(
        [
            df_relative,
            df_velocity,
            df_acceleration,
            df_angles,
            df_distances,
            df_relational,
        ],
        axis=1,
    )
    df_combined = pd.concat([df_xy, df_derived], axis=1)

    return df_combined, df_xy, df_p
