"""Sequence padding, warmup masking, and feature normalization alignment utilities."""

import numpy as np
import torch


def collate_padded_sequences(
    sequences: list[np.ndarray],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length `[T_i, C]` arrays into `(x_pad [B, T_max, C], mask [B, T_max])`."""
    if not sequences:
        raise ValueError("Cannot collate an empty sequence list.")
    batch_size = len(sequences)
    n_channels = sequences[0].shape[-1]
    max_len = max(seq.shape[0] for seq in sequences)

    x_pad = torch.zeros((batch_size, max_len, n_channels), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for idx, seq in enumerate(sequences):
        t_len = seq.shape[0]
        x_pad[idx, :t_len] = torch.from_numpy(np.asarray(seq, dtype=np.float32))
        mask[idx, :t_len] = True
    return x_pad, mask


def make_deam_warmup_mask(times_s: np.ndarray, warmup_sec: float = 15.0) -> np.ndarray:
    """Return boolean mask `True` for `t >= warmup_sec` (`15.0 s` DEAM annotation start)."""
    t = np.asarray(times_s, dtype=np.float32)
    return t >= (warmup_sec - 1e-4)
