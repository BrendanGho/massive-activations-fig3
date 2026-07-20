"""Numeric core for the high-norm-token / massive-activation overlap experiment.

Pure ``numpy`` + stdlib ``math`` (no scipy, no sklearn, no torch), so this module
imports and unit-tests on CPU. See ``SPEC_highnorm.md`` for the question and the
decision rule; ``src/experiments/highnorm_tokens.py`` is the GPU runner on top.

**The invariant this module exists to protect:** ``‖x‖² = Σ_d x[d]²``, so a token with a
massive value in one channel has its L2 norm dominated by that channel. Comparing a
massive-channel score against the *full* token norm is therefore circular. Every overlap
statistic here is meant to be fed ``token_norms(X, exclude=C)`` — the norm with the
massive channels excised — never the full norm.
"""

from __future__ import annotations

import math

import numpy as np

from src.stage2_channel_ranking import channel_scores

# --- channel selection --------------------------------------------------------


def top_channels(image_stream: np.ndarray, k: int) -> np.ndarray:
    """Top-k channels by ``mean(abs(activations))`` — abs then mean (stage-2 invariant).

    Stable sort => deterministic tie-breaking by ascending channel index, matching
    ``stage2_channel_ranking.rank_channels``. Returns (k,) int32, rank order (best first).
    """
    score = channel_scores(image_stream)
    d = score.shape[0]
    k = int(k)
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k > d:
        raise ValueError(f"k={k} exceeds channel dim D={d}")
    return np.argsort(-score, kind="stable")[:k].astype(np.int32)


def scale_matched_channels(
    image_stream: np.ndarray,
    k: int,
    exclude: np.ndarray,
    rng: np.random.Generator,
    rank_lo: int = 50,
    rank_hi: int = 500,
) -> np.ndarray:
    """``k`` random channels from descending-rank window ``[rank_lo, rank_hi)``, minus ``exclude``.

    The *scale matching* matters: drawing uniformly over all D channels mostly hits
    near-dead channels, against which any real effect looks significant. Sampling from a
    mid-rank window compares the massive channels against merely *active* ones.
    """
    score = channel_scores(image_stream)
    d = score.shape[0]
    order = np.argsort(-score, kind="stable")
    lo = max(0, int(rank_lo))
    hi = min(d, int(rank_hi))
    if hi <= lo:
        raise ValueError(f"empty rank window [{rank_lo}, {rank_hi}) for D={d}")

    banned = {int(c) for c in np.asarray(exclude).ravel()}
    pool = np.array([c for c in order[lo:hi] if int(c) not in banned], dtype=np.int32)
    if pool.size < int(k):
        raise ValueError(
            f"rank window [{rank_lo}, {rank_hi}) has {pool.size} eligible channels, need {k}"
        )
    return rng.choice(pool, size=int(k), replace=False).astype(np.int32)


# --- per-token quantities -----------------------------------------------------


def massive_score(image_stream: np.ndarray, channel_idx: np.ndarray) -> np.ndarray:
    """``m[n] = Σ_{c∈C} |x[n,c]|`` — the score whose top tail makes the speckles. (N,) float64."""
    acts = np.asarray(image_stream, dtype=np.float64)
    if acts.ndim != 2:
        raise ValueError(f"image_stream must be 2D (N_tokens, D), got {acts.shape}")
    idx = np.asarray(channel_idx, dtype=int).ravel()
    if idx.size == 0:
        return np.zeros(acts.shape[0], dtype=np.float64)
    return np.abs(acts[:, idx]).sum(axis=1)


def token_norms(image_stream: np.ndarray, exclude: np.ndarray | None = None) -> np.ndarray:
    """Per-token L2 norm, optionally over the complement of ``exclude``. (N,) float64.

    ``exclude=None`` gives Darcet's ``N_full``; ``exclude=C_k`` gives the deconfounded
    ``N_ex`` that every overlap statistic in this experiment must use.
    """
    acts = np.asarray(image_stream, dtype=np.float64)
    if acts.ndim != 2:
        raise ValueError(f"image_stream must be 2D (N_tokens, D), got {acts.shape}")
    if exclude is not None:
        keep = np.ones(acts.shape[1], dtype=bool)
        keep[np.asarray(exclude, dtype=int).ravel()] = False
        acts = acts[:, keep]
    if acts.shape[1] == 0:
        return np.zeros(image_stream.shape[0], dtype=np.float64)
    return np.linalg.norm(acts, axis=1)


def norm_fraction(image_stream: np.ndarray, channel_idx: np.ndarray) -> np.ndarray:
    """``ρ[n] = ‖x[n,C]‖² / ‖x[n,:]‖²`` — share of squared norm owned by ``C``. (N,) in [0,1].

    Zero-norm tokens yield 0.0 rather than NaN, so downstream medians stay finite.
    """
    acts = np.asarray(image_stream, dtype=np.float64)
    if acts.ndim != 2:
        raise ValueError(f"image_stream must be 2D (N_tokens, D), got {acts.shape}")
    idx = np.asarray(channel_idx, dtype=int).ravel()
    total = np.square(acts).sum(axis=1)
    part = np.square(acts[:, idx]).sum(axis=1) if idx.size else np.zeros_like(total)
    out = np.zeros_like(total)
    nz = total > 0
    out[nz] = part[nz] / total[nz]
    return np.clip(out, 0.0, 1.0)


def top_fraction_indices(values: np.ndarray, frac: float) -> np.ndarray:
    """Indices of the top ``frac`` of ``values`` (at least 1). Ties broken by ascending index."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    n_take = max(1, int(round(v.size * float(frac))))
    return np.sort(np.argsort(-v, kind="stable")[:n_take]).astype(np.int64)


# --- E1: variance-explained curve ---------------------------------------------


def variance_explained_curve(
    image_stream: np.ndarray,
    ks: list[int],
    base_k: int = 2,
    outlier_frac: float = 0.01,
) -> dict[str, list[float]]:
    """How much of a token's squared norm the top-k channels own — outliers vs typical.

    The outlier token set is fixed **once** at ``base_k`` (the channels that produce the
    speckles) and then held constant while ``k`` sweeps, so the curve reads as "for the
    tokens that make the speckles, how much of their norm do the top-k channels explain".
    Letting the token set move with ``k`` would conflate two effects.

    Returns ``{"ks", "rho_outlier", "rho_typical"}`` — medians over each token group.
    This is E1, the crux: ``rho_outlier → 1`` at small k with ``rho_typical`` near zero
    is H1 (one mechanism); a mid-range ``rho_outlier`` is H2 (a distinct register token).
    """
    base = top_channels(image_stream, base_k)
    outliers = top_fraction_indices(massive_score(image_stream, base), outlier_frac)
    mask = np.zeros(np.asarray(image_stream).shape[0], dtype=bool)
    mask[outliers] = True

    out: dict[str, list[float]] = {"ks": [], "rho_outlier": [], "rho_typical": []}
    for k in ks:
        rho = norm_fraction(image_stream, top_channels(image_stream, k))
        out["ks"].append(int(k))
        out["rho_outlier"].append(float(np.median(rho[mask])))
        out["rho_typical"].append(float(np.median(rho[~mask])) if (~mask).any() else float("nan"))
    return out


# --- E2: set overlap ----------------------------------------------------------


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_sf(obs: int, n_pop: int, n_success: int, n_draw: int) -> float:
    """``P(X >= obs)`` for X ~ Hypergeometric(n_pop, n_success, n_draw). Pure stdlib.

    Exact via log-gamma; the sums here are over at most a few thousand terms.
    """
    obs, n_pop, n_success, n_draw = int(obs), int(n_pop), int(n_success), int(n_draw)
    lo = max(obs, max(0, n_draw + n_success - n_pop))
    hi = min(n_draw, n_success)
    if lo > hi:
        return 0.0 if obs > hi else 1.0
    denom = _log_comb(n_pop, n_draw)
    total = 0.0
    for i in range(lo, hi + 1):
        total += math.exp(
            _log_comb(n_success, i) + _log_comb(n_pop - n_success, n_draw - i) - denom
        )
    return float(min(1.0, max(0.0, total)))


def overlap_stats(a_idx: np.ndarray, b_idx: np.ndarray, n_tokens: int) -> dict[str, float]:
    """Intersection of two token sets vs chance: IoU, observed/expected, hypergeometric p.

    Feed this the massive-score top set and the **deconfounded** ``N_ex`` top set. Both
    empty -> IoU 1.0 (vacuously identical), matching ``spatial.topk_jaccard``.
    """
    a = {int(x) for x in np.asarray(a_idx).ravel()}
    b = {int(x) for x in np.asarray(b_idx).ravel()}
    union = a | b
    inter = a & b
    iou = 1.0 if not union else len(inter) / len(union)
    expected = (len(a) * len(b) / n_tokens) if n_tokens > 0 else 0.0
    p = hypergeom_sf(len(inter), int(n_tokens), len(a), len(b)) if union else 1.0
    return {
        "iou": float(iou),
        "n_intersection": float(len(inter)),
        "expected_intersection": float(expected),
        "p_value": float(p),
    }


# --- rank statistics ----------------------------------------------------------


def _average_ranks(v: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties averaged (the tie handling Spearman and Mann-Whitney need)."""
    v = np.asarray(v, dtype=np.float64).ravel()
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    sorted_v = v[order]
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC of ``scores`` predicting boolean ``labels``, via the Mann-Whitney statistic.

    Ties contribute 0.5. Degenerate (all-one-class) input -> NaN.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (Pearson on average-ranks). Constant input -> NaN."""
    ra, rb = _average_ranks(a), _average_ranks(b)
    if ra.size != rb.size:
        raise ValueError(f"length mismatch: {ra.size} vs {rb.size}")
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt(float((ra @ ra) * (rb @ rb)))
    return float(ra @ rb / denom) if denom > 0 else float("nan")


# --- E3: norm distribution shape ----------------------------------------------


#: Sarle's bimodality coefficient for a normal distribution; the conventional
#: "suggests bimodality" cutoff is 5/9 (the value for a uniform distribution).
BC_GAUSSIAN = 1.0 / 3.0
BC_BIMODAL_CUTOFF = 5.0 / 9.0


def bimodality_coefficient(values: np.ndarray) -> float:
    """Sarle's ``BC = (skew² + 1) / (excess_kurtosis + 3(n-1)²/((n-2)(n-3)))``.

    Deliberately NOT a 2-means split quality: 2-means splits *any* distribution, scoring
    ~0.64 on both a single Gaussian and a heavy-tailed lognormal, so it cannot tell
    Darcet's separated high-norm mode from a mere long tail. BC does — empirically ~0.33
    for both a Gaussian and a lognormal tail, but 0.84-0.96 for a 97/3 two-mode mixture.
    Needs n > 3; returns NaN below that or on constant input.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 3:
        return float("nan")
    sd = x.std(ddof=1)
    if sd == 0:
        return float("nan")
    centered = x - x.mean()
    skew = float((centered**3).mean() / sd**3)
    kurt = float((centered**4).mean() / sd**4 - 3.0)
    denom = kurt + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((skew**2 + 1.0) / denom) if denom > 0 else float("nan")


def bimodality_split(norms: np.ndarray, n_iter: int = 100) -> dict[str, float]:
    """Derive Darcet's hand-picked high-norm cutoff per run, plus whether it is justified.

    Darcet reads the threshold (150 for DINOv2) off a visibly bimodal histogram and notes
    it is model-specific, so we derive it by 1-D 2-means on ``log10(norm)`` instead of
    hard-coding. ``bc`` is the independent check that a two-mode reading is warranted at
    all — compare against ``BC_BIMODAL_CUTOFF``; if it sits near ``BC_GAUSSIAN`` the
    threshold is an artifact of forcing a split and the high-norm population is not
    separable in this model.

    Returns ``{"threshold", "frac_above", "bc"}``; ``threshold`` is in linear norm units
    (midpoint of the two centroids in log space).
    """
    v = np.asarray(norms, dtype=np.float64).ravel()
    v = v[np.isfinite(v) & (v > 0)]
    nan = {"threshold": float("nan"), "frac_above": float("nan"), "bc": float("nan")}
    if v.size < 2:
        return nan
    x = np.log10(v)
    bc = bimodality_coefficient(x)

    c_lo, c_hi = np.percentile(x, 25.0), np.percentile(x, 75.0)
    if c_lo == c_hi:
        return {"threshold": float(10.0**c_hi), "frac_above": 0.0, "bc": bc}
    for _ in range(int(n_iter)):
        hi_mask = np.abs(x - c_hi) < np.abs(x - c_lo)
        if not hi_mask.any() or hi_mask.all():
            break
        new_lo, new_hi = float(x[~hi_mask].mean()), float(x[hi_mask].mean())
        if new_lo == c_lo and new_hi == c_hi:
            break
        c_lo, c_hi = new_lo, new_hi

    hi_mask = np.abs(x - c_hi) < np.abs(x - c_lo)
    return {
        "threshold": float(10.0 ** (0.5 * (c_lo + c_hi))),
        "frac_above": float(hi_mask.mean()),
        "bc": bc,
    }


# --- Darcet's positive control: neighbour redundancy --------------------------


def neighbor_cosine_similarity(image_stream: np.ndarray, h_lat: int, w_lat: int) -> np.ndarray:
    """Mean cosine similarity of each token to its 4-neighbourhood on the latent grid.

    Darcet's Fig. 5a: high-norm tokens sit on patches highly redundant with their
    neighbours. Edge/corner tokens average over the 3/2 neighbours they have. Row-major
    reshape, matching ``spatial.channel_spatial_map`` and ``clustering.build_mask``.
    Zero-norm tokens contribute 0 similarity. Returns (N,) float64.
    """
    acts = np.asarray(image_stream, dtype=np.float64)
    if acts.ndim != 2:
        raise ValueError(f"image_stream must be 2D (N_tokens, D), got {acts.shape}")
    n_tokens = acts.shape[0]
    if n_tokens != h_lat * w_lat:
        raise ValueError(f"token count {n_tokens} != {h_lat}*{w_lat}={h_lat * w_lat}")

    norms = np.linalg.norm(acts, axis=1)
    unit = np.zeros_like(acts)
    nz = norms > 0
    unit[nz] = acts[nz] / norms[nz, None]
    grid = unit.reshape(h_lat, w_lat, -1)

    total = np.zeros((h_lat, w_lat), dtype=np.float64)
    count = np.zeros((h_lat, w_lat), dtype=np.float64)
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        rolled = np.roll(grid, shift, axis=axis)
        sim = (grid * rolled).sum(axis=2)
        valid = np.ones((h_lat, w_lat), dtype=bool)
        # np.roll wraps; drop the wrapped edge so neighbours are true grid neighbours.
        if axis == 0:
            valid[0 if shift == 1 else h_lat - 1, :] = False
        else:
            valid[:, 0 if shift == 1 else w_lat - 1] = False
        total += np.where(valid, sim, 0.0)
        count += valid
    return (total / np.maximum(count, 1.0)).reshape(-1)
