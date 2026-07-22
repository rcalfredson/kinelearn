from __future__ import annotations

try:
    import tensorflow as tf
    from tensorflow.keras.layers import (
        Add,
        Activation,
        Input,
        Dense,
        Dropout,
        LSTM,
        Bidirectional,
        LayerNormalization,
        TimeDistributed,
        Conv1D,
    )
    from tensorflow.keras.models import Model
except Exception:
    tf = None
    Add = Activation = Input = Dense = Dropout = LSTM = Bidirectional = None
    LayerNormalization = TimeDistributed = Conv1D = Model = None

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


def residual_tcn_receptive_field(
    kernel_size: int,
    dilations,
    *,
    convolutions_per_block: int = 2,
) -> int:
    """Return the theoretical temporal receptive field of a residual TCN."""
    kernel_size = int(kernel_size)
    convolutions_per_block = int(convolutions_per_block)
    dilations = tuple(int(dilation) for dilation in dilations)

    if kernel_size <= 0:
        raise ValueError("residual_tcn.kernel_size must be positive")
    if not dilations or any(dilation <= 0 for dilation in dilations):
        raise ValueError("residual_tcn.dilations must contain positive integers")
    if convolutions_per_block <= 0:
        raise ValueError("residual_tcn.convolutions_per_block must be positive")

    return 1 + (kernel_size - 1) * convolutions_per_block * sum(dilations)


def build_keypoint_residual_tcn(
    window_size: int,
    derived_dim: int,
    *,
    channels: int = 64,
    kernel_size: int = 3,
    dilations=(1, 2, 4, 8),
    convolutions_per_block: int = 2,
    dropout: float = 0.15,
    activation: str = "relu",
) -> "tf.keras.Model":
    """Build a noncausal residual TCN with one probability per input frame."""
    if tf is None:
        raise ImportError("TensorFlow is required to build models.")

    channels = int(channels)
    kernel_size = int(kernel_size)
    dilations = tuple(int(dilation) for dilation in dilations)
    convolutions_per_block = int(convolutions_per_block)
    dropout = float(dropout)

    if channels <= 0:
        raise ValueError("residual_tcn.channels must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("residual_tcn.dropout must be in [0, 1)")
    residual_tcn_receptive_field(
        kernel_size,
        dilations,
        convolutions_per_block=convolutions_per_block,
    )

    inp = Input(shape=(int(window_size), int(derived_dim)), name="derived_seq")
    x = Conv1D(
        filters=channels,
        kernel_size=1,
        padding="same",
        name="tcn_input_projection",
    )(inp)

    for block_idx, dilation in enumerate(dilations, start=1):
        residual = x
        for conv_idx in range(1, convolutions_per_block + 1):
            x = Conv1D(
                filters=channels,
                kernel_size=kernel_size,
                dilation_rate=dilation,
                padding="same",
                name=f"tcn_block_{block_idx}_conv_{conv_idx}",
            )(x)
            x = LayerNormalization(
                name=f"tcn_block_{block_idx}_norm_{conv_idx}"
            )(x)
            x = Activation(
                activation,
                name=f"tcn_block_{block_idx}_activation_{conv_idx}",
            )(x)
            x = Dropout(
                dropout,
                name=f"tcn_block_{block_idx}_dropout_{conv_idx}",
            )(x)

        x = Add(name=f"tcn_block_{block_idx}_residual_add")([residual, x])
        x = Activation(
            activation,
            name=f"tcn_block_{block_idx}_output_activation",
        )(x)

    out = Conv1D(
        filters=1,
        kernel_size=1,
        padding="same",
        activation="sigmoid",
        name="frame_prob",
    )(x)
    return Model(inp, out, name="keypoint_residual_tcn")


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

    if variant == "residual_tcn":
        tcn_cfg = dict(model_cfg.get("residual_tcn") or {})
        return build_keypoint_residual_tcn(
            window_size,
            derived_dim,
            channels=tcn_cfg.get("channels", 64),
            kernel_size=tcn_cfg.get("kernel_size", 3),
            dilations=tcn_cfg.get("dilations", [1, 2, 4, 8]),
            convolutions_per_block=tcn_cfg.get("convolutions_per_block", 2),
            dropout=tcn_cfg.get("dropout", 0.15),
            activation=tcn_cfg.get("activation", "relu"),
        )

    raise ValueError(f"Unknown training.model.variant: {variant}")
