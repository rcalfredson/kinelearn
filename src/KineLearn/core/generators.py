from __future__ import annotations

import numpy as np

try:
    import tensorflow as tf

    SequenceBase = tf.keras.utils.Sequence
except Exception:
    tf = None
    SequenceBase = object

# Path: src/KineLearn/core/generators.py


class KeypointWindowGenerator(SequenceBase):
    """
    Memmap-backed generator yielding:
      X: (B, T, D) float32
      y: (B, T, 1) float32   for a single behavior_idx

    Notes:
    - Uses window indices (0..N-1) as the sampling unit.
    - Works with np.memmap or ndarray.
    """

    def __init__(
        self,
        mmX: np.ndarray,
        mmY: np.ndarray,
        *,
        behavior_idx: int,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        noise_std: float = 0.0,
    ):
        if tf is None:
            raise ImportError(
                "TensorFlow is required for KeypointWindowGenerator "
                "(tf.keras.utils.Sequence)."
            )

        self.mmX = mmX
        self.mmY = mmY
        self.behavior_idx = int(behavior_idx)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.noise_std = float(noise_std)
        self.rng = (
            np.random.default_rng(self.seed) if (self.shuffle or self.noise_std > 0.0) else None
        )

        self.n = int(mmX.shape[0])
        if int(mmY.shape[0]) != self.n:
            raise ValueError(
                f"mmX/mmY count mismatch: {mmX.shape[0]} vs {mmY.shape[0]}"
            )
        if not (0 <= self.behavior_idx < int(mmY.shape[-1])):
            raise ValueError(f"behavior_idx out of range: {self.behavior_idx}")

        self.indices = np.arange(self.n, dtype=np.int64)
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(self.n / self.batch_size))

    def __getitem__(self, i: int):
        sl = slice(i * self.batch_size, (i + 1) * self.batch_size)
        idx = self.indices[sl]

        Xb = np.asarray(self.mmX[idx], dtype=np.float32)  # (B,T,D)
        yb = np.asarray(self.mmY[idx, :, self.behavior_idx], dtype=np.float32)  # (B,T)
        yb = yb[..., None]  # (B,T,1)

        if self.noise_std > 0.0:
            Xb = Xb + self.rng.normal(0.0, self.noise_std, size=Xb.shape).astype(
                np.float32
            )

        return Xb, yb

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


class StratifiedWindowGenerator(SequenceBase):
    """Memmap generator with a fixed positive/hard/random batch composition."""

    def __init__(
        self,
        mmX: np.ndarray,
        mmY: np.ndarray,
        *,
        behavior_idx: int,
        batch_size: int,
        hard_negative_indices: np.ndarray,
        positive_per_batch: int,
        hard_negative_per_batch: int,
        random_negative_per_batch: int,
        seed: int = 0,
        noise_std: float = 0.0,
    ):
        if tf is None:
            raise ImportError(
                "TensorFlow is required for StratifiedWindowGenerator "
                "(tf.keras.utils.Sequence)."
            )

        self.mmX = mmX
        self.mmY = mmY
        self.behavior_idx = int(behavior_idx)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.noise_std = float(noise_std)
        self.sampling_rng = np.random.default_rng(self.seed)
        self.noise_rng = np.random.default_rng(self.seed + 1)
        self.n = int(mmX.shape[0])
        if int(mmY.shape[0]) != self.n:
            raise ValueError(
                f"mmX/mmY count mismatch: {mmX.shape[0]} vs {mmY.shape[0]}"
            )
        if not 0 <= self.behavior_idx < int(mmY.shape[-1]):
            raise ValueError(f"behavior_idx out of range: {self.behavior_idx}")

        self.positive_per_batch = int(positive_per_batch)
        self.hard_negative_per_batch = int(hard_negative_per_batch)
        self.random_negative_per_batch = int(random_negative_per_batch)
        counts = (
            self.positive_per_batch,
            self.hard_negative_per_batch,
            self.random_negative_per_batch,
        )
        if any(count < 0 for count in counts):
            raise ValueError("Stratified per-batch counts cannot be negative")
        if sum(counts) != self.batch_size:
            raise ValueError(
                "Stratified per-batch counts must sum to batch_size; "
                f"got {counts} for batch_size={self.batch_size}"
            )

        labels = np.asarray(
            mmY[:, :, self.behavior_idx], dtype=np.uint8
        ).sum(axis=1)
        self.positive_indices = np.flatnonzero(labels > 0).astype(np.int64)
        negative_indices = np.flatnonzero(labels == 0).astype(np.int64)
        hard_negative_indices = np.asarray(
            hard_negative_indices, dtype=np.int64
        ).reshape(-1)
        if len(np.unique(hard_negative_indices)) != len(hard_negative_indices):
            raise ValueError("hard_negative_indices contains duplicates")
        if np.any(hard_negative_indices < 0) or np.any(hard_negative_indices >= self.n):
            raise ValueError("hard_negative_indices contains out-of-range values")
        if np.any(labels[hard_negative_indices] != 0):
            raise ValueError("hard_negative_indices must refer only to negative windows")

        self.hard_negative_indices = hard_negative_indices
        self.random_negative_indices = np.setdiff1d(
            negative_indices,
            hard_negative_indices,
            assume_unique=False,
        )
        required_pools = [
            ("positive", self.positive_per_batch, self.positive_indices),
            ("hard-negative", self.hard_negative_per_batch, self.hard_negative_indices),
            ("random-negative", self.random_negative_per_batch, self.random_negative_indices),
        ]
        for name, count, pool in required_pools:
            if count > 0 and len(pool) == 0:
                raise ValueError(f"No {name} windows are available for stratified sampling")

        self.steps_per_epoch = int(np.ceil(self.n / self.batch_size))
        self.indices = np.empty(
            (self.steps_per_epoch, self.batch_size), dtype=np.int64
        )
        self.on_epoch_end()

    def __len__(self) -> int:
        return self.steps_per_epoch

    def _draw(self, pool: np.ndarray, count: int) -> np.ndarray:
        total = self.steps_per_epoch * int(count)
        if total == 0:
            return np.empty((self.steps_per_epoch, 0), dtype=np.int64)
        pieces = []
        remaining = total
        while remaining > 0:
            shuffled = self.sampling_rng.permutation(pool)
            take = min(remaining, len(shuffled))
            pieces.append(shuffled[:take])
            remaining -= take
        return np.concatenate(pieces).reshape(self.steps_per_epoch, int(count))

    def on_epoch_end(self):
        positive = self._draw(self.positive_indices, self.positive_per_batch)
        hard = self._draw(
            self.hard_negative_indices, self.hard_negative_per_batch
        )
        random_negative = self._draw(
            self.random_negative_indices, self.random_negative_per_batch
        )
        indices = np.concatenate([positive, hard, random_negative], axis=1)
        for batch in indices:
            self.sampling_rng.shuffle(batch)
        self.indices = indices

    def __getitem__(self, i: int):
        idx = self.indices[int(i)]
        Xb = np.asarray(self.mmX[idx], dtype=np.float32)
        yb = np.asarray(
            self.mmY[idx, :, self.behavior_idx], dtype=np.float32
        )[..., None]
        if self.noise_std > 0.0:
            Xb = Xb + self.noise_rng.normal(
                0.0, self.noise_std, size=Xb.shape
            ).astype(np.float32)
        return Xb, yb
