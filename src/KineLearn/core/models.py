from __future__ import annotations

try:
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Input,
        Dense,
        LSTM,
        Bidirectional,
        TimeDistributed,
        Conv1D,
    )
    from tensorflow.keras.models import Model
except Exception:
    tf = None
    Input = Dense = LSTM = Bidirectional = TimeDistributed = Conv1D = Model = None

# Path: src/KineLearn/core/models.py


def build_keypoint_bilstm(window_size: int, derived_dim: int) -> "tf.keras.Model":
    """
    Keypoints-only sequence model.
    Output is per-timestep sigmoid for ONE behavior: shape (B, T, 1)
    """
    if tf is None:
        raise ImportError("TensorFlow is required to build models.")

    inp = Input(shape=(int(window_size), int(derived_dim)), name="keypoints")
    x = Bidirectional(LSTM(128, return_sequences=True), name="bilstm_128")(inp)
    x = Bidirectional(LSTM(64, return_sequences=True), name="bilstm_64")(x)
    x = TimeDistributed(Dense(32, activation="relu"), name="td_dense_32")(x)
    x = TimeDistributed(Dense(64, activation="relu"), name="td_dense_64")(x)
    out = TimeDistributed(Dense(1, activation="sigmoid"), name="y")(x)
    return Model(inp, out, name="kinelearn_keypoints_bilstm")


def build_keypoint_conv_bilstm(window_size, derived_dim, *, conv_filters=(32, 32), conv_kernel_sizes=(5, 3)):
    if tf is None:
        raise ImportError("TensorFlow is required to build models.")

    if len(conv_filters) != len(conv_kernel_sizes):
        raise ValueError("conv_frontend.filters and conv_frontend.kernel_sizes must have the same length")

    inp = Input(shape=(window_size, derived_dim), name="derived_seq")
    x = inp
    for idx, (filters, kernel_size) in enumerate(zip(conv_filters, conv_kernel_sizes), start=1):
        x = Conv1D(
            filters=int(filters),
            kernel_size=int(kernel_size),
            padding="same",
            activation="relu",
            name=f"conv_frontend_{idx}",
        )(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = TimeDistributed(Dense(32, activation="relu"))(x)
    x = TimeDistributed(Dense(64, activation="relu"))(x)
    y = TimeDistributed(Dense(1, activation="sigmoid"), name="frame_prob")(x)
    return Model(inp, y, name="keypoint_conv_bilstm")


def build_sequence_model(window_size, derived_dim, *, model_cfg=None):
    model_cfg = dict(model_cfg or {})
    variant = str(model_cfg.get("variant", "bilstm")).strip().lower()

    if variant == "bilstm":
        return build_keypoint_bilstm(window_size, derived_dim)

    if variant == "conv_bilstm":
        conv_cfg = dict(model_cfg.get("conv_frontend") or {})
        conv_filters = conv_cfg.get("filters", [32, 32])
        conv_kernel_sizes = conv_cfg.get("kernel_sizes", [5, 3])
        return build_keypoint_conv_bilstm(
            window_size,
            derived_dim,
            conv_filters=tuple(conv_filters),
            conv_kernel_sizes=tuple(conv_kernel_sizes),
        )

    raise ValueError(f"Unknown training.model.variant: {variant}")
