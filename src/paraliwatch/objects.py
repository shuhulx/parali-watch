"""Fields burn as units; object voting reduces speckle and edge omission."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def segment_fields(
    refl: np.ndarray,
    valid: np.ndarray,
    *,
    edge_q: float = 0.80,
    min_px: int = 6,
    max_px: int = 400,
) -> np.ndarray:
    """Find field interiors in one clear pre-season scene, then recover their edges."""
    if refl.ndim != 3 or refl.shape[0] != 6 or valid.shape != refl.shape[1:]:
        raise ValueError("expected reflectance [6, H, W] and matching valid mask")
    if not 0 <= edge_q <= 1 or min_px < 1 or max_px < min_px:
        raise ValueError("invalid edge quantile or component size limits")
    valid = np.asarray(valid, dtype=bool) & np.all(np.isfinite(refl[:3]), axis=0)
    labels = np.zeros(valid.shape, dtype=np.int32)
    if not np.any(valid):
        return labels

    # Extend clear data across missing pixels so nodata does not create edges.
    nearest = None
    if not np.all(valid):
        nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    gradient = np.zeros(valid.shape, dtype=np.float32)
    for band in refl[:3]:
        values = band[valid]
        lo, hi = np.percentile(values, [2, 98])
        scale = hi - lo
        if scale == 0:
            scale = np.ptp(values)
        if scale == 0:
            continue
        scaled = ((band - lo) / scale).astype(np.float32)
        if nearest is not None:
            scaled = scaled[tuple(nearest)]
        dx = ndimage.sobel(scaled, axis=1, mode="reflect")
        dy = ndimage.sobel(scaled, axis=0, mode="reflect")
        np.maximum(gradient, np.hypot(dx, dy), out=gradient)
    edges = valid & (gradient > np.quantile(gradient[valid], edge_q))
    ndimage.label(valid & ~edges, structure=ndimage.generate_binary_structure(2, 1), output=labels)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_px
    keep[0] = False
    labels[~keep[labels]] = 0

    large = (labels > 0) & (sizes[labels] > max_px)
    if np.any(large):
        rows, cols = np.nonzero(large)
        grid_cols = (valid.shape[1] + 19) // 20
        grid_size = ((valid.shape[0] + 19) // 20) * grid_cols
        keys = labels[large].astype(np.int64) * grid_size + (rows // 20) * grid_cols + cols // 20
        _, inverse = np.unique(keys, return_inverse=True)
        labels[large] = inverse + sizes.size

    # Compact ids after dropping and splitting components; reserve zero for background.
    ids = np.unique(labels)
    ids = ids[ids > 0]
    if not ids.size:
        return labels
    remap = np.zeros(int(ids[-1]) + 1, dtype=np.int32)
    remap[ids] = np.arange(1, ids.size + 1, dtype=np.int32)
    labels = remap[labels]
    distance, nearest = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    grow = edges & (distance <= 1)
    labels[grow] = labels[nearest[0][grow], nearest[1][grow]]
    return labels


def field_vote(
    burn: np.ndarray,
    candidate: np.ndarray,
    fields: np.ndarray,
    *,
    min_frac: float = 0.5,
    min_px: int = 3,
) -> np.ndarray:
    """Vote within fields to reduce speckle and fill omitted candidate edges."""
    n_fields = int(fields.max(initial=0)) + 1
    burnt = np.bincount(fields[burn], minlength=n_fields)
    eligible = np.bincount(fields[candidate], minlength=n_fields)
    passes = (eligible > 0) & (burnt >= min_frac * eligible) & (burnt >= min_px)
    return np.where(fields > 0, candidate & passes[fields], burn).astype(bool)


def field_majority_day(burn_day: np.ndarray, voted: np.ndarray, fields: np.ndarray) -> np.ndarray:
    """Fill newly burnt pixels with their field's median day (rounded down)."""
    result = np.where(voted, burn_day, 0).astype(np.int16)
    original = (burn_day != 0) & (fields > 0)
    if not np.any(original):
        return result
    ids = fields[original]
    days = burn_day[original].astype(np.int64)
    order = np.lexsort((days, ids))
    days = days[order]
    counts = np.bincount(ids, minlength=int(fields.max(initial=0)) + 1)
    present = np.flatnonzero(counts)
    starts = np.cumsum(counts) - counts
    lower = starts[present] + (counts[present] - 1) // 2
    upper = starts[present] + counts[present] // 2
    medians = np.zeros(counts.size, dtype=np.int16)
    medians[present] = (days[lower] + days[upper]) // 2
    newly_burnt = voted & (burn_day == 0) & (fields > 0)
    result[newly_burnt] = medians[fields[newly_burnt]]
    return result
