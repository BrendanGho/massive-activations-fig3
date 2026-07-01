"""Stage 3 numeric core: normalize -> K-means(2) on k-dim token vectors -> mask.

Pure ``numpy`` + ``scikit-learn`` so it runs (and is tested) on CPU with no model.

The ordering here is load-bearing (see spec invariants):
1. Min-max normalize EACH selected channel independently across tokens FIRST
   (channels differ wildly in raw scale).
2. K-means with K=2 on the resulting **k-dim per-token feature vectors** — not on
   any collapsed scalar.
3. Separately, per-token scalar ``s[n] = normalized[n, :].sum()`` is the Fig 3B heatmap.
4. Foreground = the K-means cluster with the higher mean ``s``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # sklearn is in the dev + fig3 deps; import at module load is fine (lightweight).
    from sklearn.cluster import KMeans
except Exception as exc:  # pragma: no cover - only hit if deps missing
    KMeans = None  # type: ignore[assignment]
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None


@dataclass
class MaskResult:
    mask: np.ndarray  # (H_lat, W_lat) bool foreground mask
    heatmap: np.ndarray  # (H_lat, W_lat) float32 per-token scalar s (Fig 3B)
    labels: np.ndarray  # (N_tokens,) int cluster id per token
    fg_cluster: int  # which cluster id was chosen as foreground
    n_channels: int  # k actually used


def minmax_normalize_channels(feat: np.ndarray) -> np.ndarray:
    """Per-channel (axis=0, across tokens) min-max normalization to [0, 1].

    feat: (N_tokens, k). Constant channels (max == min) map to 0 to avoid div-by-zero.
    """
    feat = np.asarray(feat, dtype=np.float64)
    if feat.ndim != 2:
        raise ValueError(f"feat must be 2D (N_tokens, k), got shape {feat.shape}")
    lo = feat.min(axis=0, keepdims=True)
    hi = feat.max(axis=0, keepdims=True)
    span = hi - lo
    span[span == 0] = 1.0  # constant channel -> all zeros after subtraction
    return (feat - lo) / span


def kmeans_foreground(normalized: np.ndarray, seed: int = 0):
    """K-means(2) on k-dim token vectors; foreground = higher-mean-``s`` cluster.

    Returns (labels, fg_cluster, s). Uses sklearn library defaults for init/n_init
    (logged in run_metadata) and a derived ``random_state`` for reproducibility.
    """
    if KMeans is None:  # pragma: no cover
        raise ImportError(f"scikit-learn is required for clustering: {_SKLEARN_IMPORT_ERROR}")
    normalized = np.asarray(normalized, dtype=np.float64)
    n_tokens = normalized.shape[0]
    s = normalized.sum(axis=1)  # (N_tokens,) Fig 3B heatmap scalar

    if n_tokens < 2:
        return np.zeros(n_tokens, dtype=int), 0, s

    km = KMeans(n_clusters=2, random_state=int(seed) % (2**31 - 1))
    labels = km.fit_predict(normalized)

    # Foreground = cluster with the higher mean s. Guard against an empty cluster.
    means = {}
    for c in (0, 1):
        members = labels == c
        if members.any():
            means[c] = float(s[members].mean())
    fg_cluster = max(means, key=means.get)
    return labels, fg_cluster, s


def build_mask(
    image_stream: np.ndarray,
    channel_idx: np.ndarray,
    h_lat: int,
    w_lat: int,
    seed: int = 0,
) -> MaskResult:
    """Full Stage 3 for one (layer, strategy): selected channels -> binary mask.

    image_stream: (N_tokens, D) image-stream activations for one layer.
    channel_idx:  (k,) indices of the selected channels.
    Returns a MaskResult with mask/heatmap reshaped to (h_lat, w_lat) row-major.
    """
    image_stream = np.asarray(image_stream)
    channel_idx = np.asarray(channel_idx, dtype=int).ravel()
    n_tokens = image_stream.shape[0]
    if n_tokens != h_lat * w_lat:
        raise ValueError(f"token count {n_tokens} != h_lat*w_lat {h_lat}*{w_lat}={h_lat * w_lat}")

    feat = image_stream[:, channel_idx]  # (N_tokens, k)
    normalized = minmax_normalize_channels(feat)
    labels, fg_cluster, s = kmeans_foreground(normalized, seed=seed)

    mask_tokens = labels == fg_cluster  # (N_tokens,) bool
    mask = mask_tokens.reshape(h_lat, w_lat)
    heatmap = s.reshape(h_lat, w_lat).astype(np.float32)
    return MaskResult(
        mask=mask,
        heatmap=heatmap,
        labels=labels,
        fg_cluster=fg_cluster,
        n_channels=int(channel_idx.size),
    )
