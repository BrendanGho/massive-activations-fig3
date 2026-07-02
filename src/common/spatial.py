"""Spatial-map + stability helpers for the channel-stability experiment.

Pure ``numpy`` (plus ``clustering.minmax_normalize_channels`` for the aggregated
heatmap, which is itself pure numpy) so this module imports and unit-tests on CPU
with no torch/diffusers/matplotlib.

Two groups of helpers:

* **spatial maps** — reshape a single channel's per-token activations (or the
  Fig-3B–style aggregate over a set of channels) to the latent grid ``(h_lat, w_lat)``.
  Reshape is row-major, matching ``clustering.build_mask`` so a per-channel map and
  the aggregated heatmap align pixel-for-pixel.
* **stability metrics** — set-overlap between the top-k channel *identities* of two
  scenarios (Jaccard), and how often each channel is selected across scenarios.
"""

from __future__ import annotations

import numpy as np

from src.common import clustering


def _check_grid(n_tokens: int, h_lat: int, w_lat: int) -> None:
    if n_tokens != h_lat * w_lat:
        raise ValueError(f"token count {n_tokens} != h_lat*w_lat {h_lat}*{w_lat}={h_lat * w_lat}")


def channel_spatial_map(
    image_stream: np.ndarray,
    channel: int,
    h_lat: int,
    w_lat: int,
    normalize: bool = True,
) -> np.ndarray:
    """One channel's per-token activations reshaped to the latent grid.

    image_stream: (N_tokens, D). Returns (h_lat, w_lat) float32, row-major (matching
    ``clustering.build_mask``). ``normalize`` applies per-map min-max to [0, 1] for
    display (a constant map -> all zeros).
    """
    image_stream = np.asarray(image_stream)
    if image_stream.ndim != 2:
        raise ValueError(f"image_stream must be 2D (N_tokens, D), got {image_stream.shape}")
    n_tokens = image_stream.shape[0]
    _check_grid(n_tokens, h_lat, w_lat)

    vals = image_stream[:, int(channel)].astype(np.float64)
    if normalize:
        lo = float(vals.min())
        hi = float(vals.max())
        span = hi - lo
        vals = (vals - lo) / span if span != 0 else np.zeros_like(vals)
    return vals.reshape(h_lat, w_lat).astype(np.float32)


def aggregated_topk_heatmap(
    image_stream: np.ndarray,
    channel_idx: np.ndarray,
    h_lat: int,
    w_lat: int,
) -> np.ndarray:
    """Fig-3B aggregate heatmap over a set of channels, reshaped to the latent grid.

    Min-max normalizes each selected channel across tokens, then sums across channels
    -> per-token scalar ``s[n]`` (identical to ``clustering.build_mask(...).heatmap``,
    but without running K-means since we only want the heatmap). Returns (h_lat, w_lat)
    float32.
    """
    image_stream = np.asarray(image_stream)
    channel_idx = np.asarray(channel_idx, dtype=int).ravel()
    n_tokens = image_stream.shape[0]
    _check_grid(n_tokens, h_lat, w_lat)

    feat = image_stream[:, channel_idx]  # (N_tokens, k)
    normalized = clustering.minmax_normalize_channels(feat)
    s = normalized.sum(axis=1)  # (N_tokens,)
    return s.reshape(h_lat, w_lat).astype(np.float32)


def topk_jaccard(a_idx: np.ndarray, b_idx: np.ndarray) -> float:
    """Jaccard overlap of two channel-id sets. Two empty sets -> 1.0 (perfect match)."""
    a = set(int(x) for x in np.asarray(a_idx).ravel())
    b = set(int(x) for x in np.asarray(b_idx).ravel())
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def pairwise_jaccard_matrix(topk_sets: list[np.ndarray]) -> np.ndarray:
    """(S, S) symmetric matrix of pairwise ``topk_jaccard``; diagonal == 1.0."""
    s = len(topk_sets)
    mat = np.ones((s, s), dtype=np.float64)
    for i in range(s):
        for j in range(i + 1, s):
            val = topk_jaccard(topk_sets[i], topk_sets[j])
            mat[i, j] = mat[j, i] = val
    return mat


def selection_frequency(topk_sets: list[np.ndarray], d: int) -> np.ndarray:
    """Per-channel count of how many of the given top-k sets include that channel.

    Returns an int array of length ``d`` (channel dim); index = channel id.
    """
    counts = np.zeros(int(d), dtype=np.int64)
    for idx in topk_sets:
        for c in set(int(x) for x in np.asarray(idx).ravel()):
            if 0 <= c < d:
                counts[c] += 1
    return counts
