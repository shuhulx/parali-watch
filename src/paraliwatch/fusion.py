"""Sentinel-2 x MODIS fusion as a teacher -> student pipeline.

Stage 1 — **Teacher (MODIS supervises Sentinel-2).** The MCD64A1 idea moved
from 500 m to 20 m: fold the S2 ARD stack in date order, keep each pixel's last
clear observation, and emit every clear-to-clear *candidate transition* that
loses NBR or darkens in SWIR. MODIS active fires weakly label them (positive =
fire in the pixel's own 1 km cell inside the interval; negative = no fire within
±2 cells / ±2 days while MODIS could see the cell). A logistic regression on S2
features only learns P(fire-neighbourhood | spectra). The weak labels are noisy
— farmers till within days of burning, so "fire cells" also hold tillage — so
the teacher is good at *ranking* but its decision boundary is biased.

Stage 2 — **Student (label-efficient fusion).** A few hundred transitions,
drawn away from the test points and stratified over the teacher's score, are
photo-interpreted. Because these labels no longer come from MODIS, MODIS can
finally enter as *features* (fire counts in/near the cell, MODIS visibility,
the teacher score). A gradient-boosted model is trained with spatial
cross-validation over 1 km cells, and its threshold maximises out-of-fold F1.

Attribution — the first transition over threshold is the pixel's burn; a
MODIS-confirmed burn is dated to the last MODIS fire inside its S2 interval
("S2 says which field, MODIS says when"); otherwise the S2 date is kept.
Both maps are written so the gain from the student is measurable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pystac
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .ard import _write, read_s2_ard
from .catalog import LocalCatalog
from .config import Settings
from .grid import Grid

log = logging.getLogger(__name__)

FEATURES = [
    "dnbr", "nbr_post", "b8a_post", "b11_post", "b12_post", "d_b12", "nbr2_post", "d_nbr2", "mirbi_post",
    "d_b11", "nbr_pre", "b11_pre", "vis_post", "d_vis", "gap_days",
    "dnbr_ctx5", "d_b11_ctx5", "b12_post_ctx5",
]  # fmt: skip
# Student-only features: may be NaN (no later clear view), which the gradient-boosted student handles
TEMPORAL_FEATURES = ["b12_next", "nbr_next", "next_gap", "hot_post", "hot_pre"]
MODIS_FEATURES = ["modis_own", "modis_near", "modis_seen_frac", "modis_high", "modis_frp_max", "teacher_p"]
EPS = 1e-6
MAX_PER_CLASS = 100_000
SEED = 20241101


def indices(refl: np.ndarray) -> dict[str, np.ndarray]:
    b8a, b11, b12 = refl[:3]
    return {
        "nbr": (b8a - b12) / (b8a + b12 + EPS),
        "nbr2": (b11 - b12) / (b11 + b12 + EPS),
        "mirbi": 10 * b12 - 9.8 * b11 + 2,  # Trigg & Flasse (2001): char-sensitive
        "b8a": b8a,
        "b11": b11,
        "b12": b12,
    }


@dataclass
class Transitions:
    pix: np.ndarray  # flat pixel index
    day_prev: np.ndarray  # ordinal of previous clear observation
    day: np.ndarray  # ordinal of this observation
    X: np.ndarray  # [n, len(FEATURES)] float32


@dataclass
class Fold:
    transitions: Transitions
    seen: dict[date, np.ndarray]  # S2 date -> pixels clear on that date
    n_obs: np.ndarray  # clear observations per pixel over the season


def _local_mean(values: np.ndarray, where: np.ndarray, shape: tuple[int, int], size: int = 5) -> np.ndarray:
    """Mean of a per-pixel signal over its size x size neighbourhood, ignoring unobserved pixels.

    Burns are field-sized (5-10 px); an isolated dark pixel among unchanged neighbours is
    more likely noise. Context is a soft feature, unlike field voting which erased small scars.
    """
    grid = np.zeros(shape, np.float32)
    grid[where] = values
    num = ndimage.uniform_filter(grid, size)
    den = ndimage.uniform_filter(where.astype(np.float32), size)
    return (num / np.maximum(den, 1e-6))[where]


def fold(scenes: list[tuple[date, Path]], shape: tuple[int, int], min_dnbr: float, min_dswir: float) -> Fold:
    """Stream scenes in time order; emit candidate transitions.

    A candidate loses NBR (green -> harvested/burnt) *or* darkens in SWIR
    (bright stubble -> char). The second rule matters: stubble already has low
    NBR, so burning it barely moves NBR but halves SWIR reflectance.
    """
    prev = None
    prev_day = np.full(shape, -1, np.int32)
    n_obs = np.zeros(shape, np.int16)
    parts: list[Transitions] = []
    seen = {}
    for d, path in sorted(scenes):
        refl, valid = read_s2_ard(path)
        if prev is None:
            prev = np.full(refl.shape, np.nan, np.float32)
        seen[d] = valid
        n_obs += valid
        have = valid & (prev_day >= 0)
        if have.any():
            r1, r0 = refl[:, have], prev[:, have]
            now, pre = indices(r1), indices(r0)
            dnbr = pre["nbr"] - now["nbr"]
            dswir = pre["b11"] - now["b11"]
            keep = (dnbr > min_dnbr) | (dswir > min_dswir)
            vis_now = r1[3:6].mean(axis=0) if r1.shape[0] >= 6 else np.zeros_like(dnbr)
            vis_pre = r0[3:6].mean(axis=0) if r0.shape[0] >= 6 else np.zeros_like(dnbr)
            gap = d.toordinal() - prev_day[have]
            ctx = [_local_mean(v, have, shape) for v in (dnbr, dswir, now["b12"])]
            X = np.column_stack(
                [
                    dnbr,
                    now["nbr"],
                    now["b8a"],
                    now["b11"],
                    now["b12"],
                    now["b12"] - pre["b12"],
                    now["nbr2"],
                    pre["nbr2"] - now["nbr2"],
                    now["mirbi"],
                    dswir,
                    pre["nbr"],
                    pre["b11"],
                    vis_now,
                    vis_pre - vis_now,
                    gap,
                    *ctx,
                ]
            )[keep].astype(np.float32)
            pix = np.flatnonzero(have)[keep]
            parts.append(Transitions(pix, prev_day.ravel()[pix].copy(), np.full(pix.size, d.toordinal(), np.int32), X))
        prev[:, valid] = refl[:, valid]
        prev_day[valid] = d.toordinal()
    if not parts:
        parts.append(
            Transitions(
                np.zeros(0, np.int64),
                np.zeros(0, np.int32),
                np.zeros(0, np.int32),
                np.zeros((0, len(FEATURES)), np.float32),
            )
        )
    t = Transitions(
        np.concatenate([p.pix for p in parts]),
        np.concatenate([p.day_prev for p in parts]),
        np.concatenate([p.day for p in parts]),
        np.concatenate([p.X for p in parts]),
    )
    return Fold(t, seen, n_obs)


class FireIndex:
    """O(1) 'any fire in cell c between days a..b' via cumulative sums over days."""

    def __init__(self, days: np.ndarray, fire_cls: np.ndarray, observed: np.ndarray, min_class: int, buffer: int):
        self.day0 = int(days[0])
        self.n = len(days)
        fire = fire_cls >= min_class
        size = (1, 2 * buffer + 1, 2 * buffer + 1)
        near = ndimage.maximum_filter(fire, size=size) if buffer else fire
        self._own = _cum(fire)
        self._near = _cum(near)
        self._obs = _cum(observed)
        self.shape = fire.shape[1:]

    def _count(self, cum, cell, a, b):
        """Events on days a..b inclusive (ordinals); ranges outside the stack count zero."""
        lo = np.clip(np.asarray(a) - self.day0, 0, self.n)
        hi = np.clip(np.asarray(b) - self.day0 + 1, 0, self.n)
        hi = np.maximum(hi, lo)
        flat = cum.reshape(cum.shape[0], -1)
        return flat[hi, cell] - flat[lo, cell]

    def own(self, cell, a, b):
        return self._count(self._own, cell, a, b)

    def near(self, cell, a, b):
        return self._count(self._near, cell, a, b)

    def observed(self, cell, a, b):
        return self._count(self._obs, cell, a, b)

    def last_near_day(self, cell, a, b):
        """Latest day in a..b with a fire in/near the cell (ordinal), or -1."""
        a, b = np.asarray(a), np.asarray(b)
        out = np.full(np.broadcast(cell, b).shape, -1, np.int64)
        for k in range(int((b - a).max(initial=0)) + 1):
            d = b - k
            undecided = (out < 0) & (d >= a)
            hit = undecided & (self.near(cell, d, d) > 0)
            out[hit] = d[hit]
        return out


def _cum(x: np.ndarray) -> np.ndarray:
    c = np.zeros((x.shape[0] + 1, *x.shape[1:]), np.int32)
    np.cumsum(x, axis=0, out=c[1:])
    return c


def coarse_cell(pix: np.ndarray, grid: Grid) -> np.ndarray:
    r, c = np.divmod(pix, grid.shape[1])
    return (r // grid.factor) * grid.coarse.shape[1] + (c // grid.factor)


def weak_labels(t: Transitions, cells: np.ndarray, fi: FireIndex, pad: int) -> np.ndarray:
    """1 = fire-positive, 0 = confident negative, -1 = unlabelled."""
    # A transition covers (day_prev, day]: the fire must come after the last clear view.
    y = np.full(t.pix.size, -1, np.int8)
    pos = fi.own(cells, t.day_prev + 1, t.day) > 0
    quiet = fi.near(cells, t.day_prev + 1 - pad, t.day + pad) == 0
    seen = fi.observed(cells, t.day_prev + 1, t.day) > 0
    y[quiet & seen] = 0
    y[pos] = 1
    return y


def fit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cv: bool = True):
    if min((y == 0).sum(), (y == 1).sum()) < 50 or np.unique(groups[y >= 0]).size < 5:
        raise RuntimeError(
            f"not enough weak labels to train (pos={(y == 1).sum()}, neg={(y == 0).sum()}); "
            "extend the season or relax fusion.negative_buffer_cells"
        )
    rng = np.random.default_rng(SEED)
    idx = []
    for k in (0, 1):
        cls = np.flatnonzero(y == k)
        idx.append(rng.choice(cls, min(cls.size, MAX_PER_CLASS), replace=False))
    idx = np.concatenate(idx)
    Xs, ys, gs = X[idx], y[idx], groups[idx]

    def model():
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))

    aucs = []
    for tr, te in GroupKFold(n_splits=5).split(Xs, ys, gs) if cv else []:  # spatial CV: whole 1 km cells held out
        m = model().fit(Xs[tr], ys[tr])
        aucs.append(roc_auc_score(ys[te], m.predict_proba(Xs[te])[:, 1]))
    m = model().fit(Xs, ys)
    coefs = dict(zip(FEATURES, m[-1].coef_[0].round(3).tolist()))
    diag = {
        "n_pos_available": int((y == 1).sum()),
        "n_neg_available": int((y == 0).sum()),
        "n_unlabelled": int((y == -1).sum()),
        "n_train_per_class": [int((ys == 0).sum()), int((ys == 1).sum())],
        "weak_label_auc_spatial_cv": [round(float(np.mean(aucs)), 3), round(float(np.std(aucs)), 3)] if aucs else None,
        "standardised_coefficients": coefs,
        "intercept": round(float(m[-1].intercept_[0]), 3),
    }
    return m, diag


REF = Path("reference")
BIN_SHARE = [0.5, 0.3, 0.15, 0.04, 0.01]  # population share of each teacher-score bin (quantile bins)


def modis_features(t: Transitions, cells: np.ndarray, fs, fire_min_class: int, teacher_p: np.ndarray):
    """MODIS evidence inside each transition's (day_prev, day] window."""
    fi_own = FireIndex(fs["days"], fs["fire"], fs["observed"], fire_min_class, 0)
    fi_near = FireIndex(fs["days"], fs["fire"], fs["observed"], fire_min_class, 1)
    fi_high = FireIndex(fs["days"], fs["fire"], fs["observed"], 9, 0)  # high-confidence detections only
    a, b = t.day_prev + 1, t.day
    span = np.maximum(b - a + 1, 1)
    return np.column_stack(
        [
            fi_own.own(cells, a, b),
            fi_near.near(cells, a, b),
            fi_own.observed(cells, a, b) / span,
            fi_high.own(cells, a, b),
            window_max(fs["frp"], int(fs["days"][0]), cells, a, b) if "frp" in fs else np.zeros(t.pix.size),
            teacher_p,
        ]
    ).astype(np.float32)


def window_max(daily: np.ndarray, day0: int, cells: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Max of a daily (D, h, w) MODIS layer over each [a, b] day window at each cell."""
    flat = daily.reshape(daily.shape[0], -1)
    out = np.zeros(cells.size, np.float32)
    for k in range(int((b - a).max(initial=0)) + 1):
        d = b - k - day0
        ok = (b - k >= a) & (d >= 0) & (d < flat.shape[0])
        out[ok] = np.maximum(out[ok], flat[d[ok], cells[ok]])
    return out


def temporal_features(t: Transitions, scenes: list[tuple[date, Path]]) -> np.ndarray:
    """Second pass over the scenes: what the pixel looked like at the *next* clear view, and haze.

    Char that is still dark at the next clear observation is more likely a burn; a transient dip
    is more likely haze or shadow. Burnt fields are often tilled within days, so this stays a soft
    feature (NaN when there is no later clear view) rather than a rule.
    """
    from .ard import read_s2_ard as _read

    n = t.pix.size
    out = np.full((n, len(TEMPORAL_FEATURES)), np.nan, np.float32)
    filled = np.zeros(n, bool)
    for d, path in sorted(scenes):
        refl, valid = _read(path)
        o = d.toordinal()
        flat = refl.reshape(refl.shape[0], -1)
        hot = flat[3] - 0.5 * flat[5] if flat.shape[0] >= 6 else np.zeros(flat.shape[1])
        on_post, on_pre = t.day == o, t.day_prev == o
        out[on_post, 3] = hot[t.pix[on_post]]
        out[on_pre, 4] = hot[t.pix[on_pre]]
        nxt = ~filled & (t.day < o)
        nxt[nxt] = valid.ravel()[t.pix[nxt]]
        px = t.pix[nxt]
        idx = indices(flat[:, px])
        out[nxt, 0], out[nxt, 1], out[nxt, 2] = idx["b12"], idx["nbr"], o - t.day[nxt]
        filled |= nxt
    return out


def first_over(t: Transitions, p: np.ndarray, thr: float) -> np.ndarray:
    """Index of each pixel's first transition with p > thr."""
    hit = np.flatnonzero(p > thr)  # strict: tied negatives cannot breach a cap
    hit = hit[np.lexsort((t.day[hit], t.pix[hit]))]
    _, first = np.unique(t.pix[hit], return_index=True)
    return hit[first]


def write_burn_map(path: Path, t: Transitions, sel: np.ndarray, p: np.ndarray, cells, fi_near, settings, grid):
    start = settings.season[0].toordinal()
    H, W = grid.shape
    burn_day = np.zeros(H * W, np.int16)  # days since season start + 1; 0 = no burn
    prev_day = np.zeros(H * W, np.int16)
    prob = np.zeros(H * W, np.uint8)
    evidence = np.zeros(H * W, np.uint8)  # 0 none, 1 S2-only, 2 MODIS-confirmed; +10 long-gap
    obs_day = np.zeros(H * W, np.int16)
    # Sentinel-2 says *which field*; MODIS says *when*. A confirmed burn is dated to the
    # last MODIS fire in its neighbourhood inside the S2 interval; an S2-only burn keeps
    # the S2 observation date (an upper bound). This undoes the smog-gap pile-up, where
    # burns from the blind week would otherwise all land on the first clear day.
    fire_day = fi_near.last_near_day(cells[sel], t.day_prev[sel] + 1, t.day[sel])
    confirmed = fire_day > 0
    dated = np.where(confirmed, fire_day, t.day[sel])
    burn_day[t.pix[sel]] = dated - start + 1
    obs_day[t.pix[sel]] = t.day[sel] - start + 1
    prev_day[t.pix[sel]] = t.day_prev[sel] - start + 1
    prob[t.pix[sel]] = np.round(p[sel] * 100).astype(np.uint8)
    long_gap = (t.day[sel] - t.day_prev[sel]) > settings["fusion"]["max_gap_days"]
    evidence[t.pix[sel]] = np.where(confirmed, 2, 1) + 10 * long_gap
    stack = np.stack([burn_day, prev_day, prob.astype(np.int16), evidence.astype(np.int16), obs_day])
    _write(
        path,
        stack.reshape(5, H, W),
        grid.fine,
        nodata=None,
        descriptions=("burn_day", "prev_day", "prob_pct", "evidence", "s2_obs_day"),
    )
    return {
        "n_burn_pixels": int(sel.size),
        "burnt_ha": round(sel.size * grid.pixel_area_ha, 1),
        "share_modis_confirmed": round(float(confirmed.mean()), 3) if sel.size else None,
        "median_days_redated_by_modis": float(np.median((t.day[sel] - dated)[confirmed])) if confirmed.any() else None,
        "share_long_gap": round(float(long_gap.mean()), 3) if sel.size else None,
    }


def field_smooth(pixel_path: Path, out: Path, t: Transitions, scenes, settings: Settings, grid: Grid) -> dict:
    """Fields burn as units: segment fields on one clear scene and vote within each."""
    import rasterio

    from .objects import field_majority_day, field_vote, segment_fields

    want = date.fromisoformat(settings["fusion"]["fields_scene"])
    d, path = min(scenes, key=lambda x: abs((x[0] - want).days))
    refl, valid = read_s2_ard(path)
    fields = segment_fields(refl, valid)
    with rasterio.open(pixel_path) as src:
        bands = src.read()
    burn = bands[0] > 0
    candidate = np.zeros(grid.shape[0] * grid.shape[1], bool)
    candidate[t.pix] = True
    voted = field_vote(burn, candidate.reshape(grid.shape), fields, min_frac=settings["fusion"]["field_min_frac"])
    smoothed = np.stack([field_majority_day(b, voted, fields) for b in bands]).astype(np.int16)
    _write(out, smoothed, grid.fine, nodata=None,
           descriptions=("burn_day", "prev_day", "prob_pct", "evidence", "s2_obs_day"))  # fmt: skip
    n = int(voted.sum())
    return {
        "fields_scene": d.isoformat(),
        "n_fields": int(fields.max()),
        "n_burn_pixels": n,
        "burnt_ha": round(n * grid.pixel_area_ha, 1),
        "pixels_added": int((voted & ~burn).sum()),
        "pixels_removed": int((burn & ~voted).sum()),
    }


def make_training_sample(t: Transitions, cells: np.ndarray, teacher_p: np.ndarray, grid: Grid, n_per_bin: int):
    """Transitions to photo-interpret, stratified over the teacher score, away from test points."""
    import pandas as pd

    test = pd.read_csv(REF / "sample.csv")
    f, w = grid.factor, grid.coarse.shape[1]
    test_cells = set(((test["row"] // f) * w + test["col"] // f).tolist())
    ok = ~np.isin(cells, list(test_cells))
    edges = np.quantile(teacher_p[ok], np.concatenate([[0], np.cumsum(BIN_SHARE)]))
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for k in range(5):
        lo, hi = edges[k], edges[k + 1]
        idx = np.flatnonzero(ok & (teacher_p >= lo) & ((teacher_p < hi) if k < 4 else True))
        for i in rng.choice(idx, min(n_per_bin, idx.size), replace=False):
            r, c = divmod(int(t.pix[i]), grid.shape[1])
            rows.append(
                {"bin": k, "row": r, "col": c, "pix": int(t.pix[i]),
                 "pre": date.fromordinal(int(t.day_prev[i])), "post": date.fromordinal(int(t.day[i]))}
            )  # fmt: skip
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "point_id", [f"T{i:03d}" for i in range(len(df))])
    df.to_csv(REF / "train_sample.csv", index=False)
    return df


def _hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
        l2_regularization=1.0, class_weight="balanced", random_state=SEED,
    )  # fmt: skip


def _student(Xs, ys, blocks, w, teacher_for, pseudo=None, scored=None):
    """Spatially nested validation of the whole teacher -> student stack.

    Folds hold out 3 km blocks (larger than any feature window). Inside each fold the teacher is
    refit without the held-out blocks and its score replaces the ``teacher_p`` column, so no
    held-out block influences its own features. ``pseudo`` = (X, y, w, blocks) of teacher-confident
    pseudo-labels, added to training folds only (never to the gold out-of-fold rows).
    ``scored`` marks the probability-sampled labels; actively-sampled ones train in every fold whose
    held-out blocks they lie outside, but are never scored, since they are not a random draw and would
    bias the F1 that sets the threshold. (They were chosen by a model fit on all labels, so the OOF
    estimate is mildly optimistic, as is reporting the F1 the threshold maximises; see MODEL_CARD.)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    oof = np.zeros(ys.size)
    for tr, te in GroupKFold(n_splits=5).split(Xs, ys, blocks):
        held = np.unique(blocks[te])
        tp, tp_pseudo = teacher_for(held, pseudo[4] if pseudo is not None else None)
        Xtr, Xte = Xs[tr].copy(), Xs[te].copy()
        Xtr[:, -1], Xte[:, -1] = tp[tr], tp[te]
        ytr, wtr = ys[tr], np.ones(tr.size)
        if pseudo is not None:
            # Pseudo-labels are re-derived with this fold's teacher: keep a row only if the fold
            # teacher agrees with its label, and give it the fold teacher's score.
            hi_t, lo_t = pseudo[5]
            agree = np.where(pseudo[1] == 1, tp_pseudo >= hi_t, tp_pseudo <= lo_t)
            keep = ~np.isin(pseudo[3], held) & agree
            Xp = pseudo[0][keep].copy()
            Xp[:, -1] = tp_pseudo[keep]
            Xtr = np.vstack([Xtr, Xp])
            ytr = np.concatenate([ytr, pseudo[1][keep]])
            wtr = np.concatenate([wtr, pseudo[2][keep]])
        oof[te] = _hgb().fit(Xtr, ytr, sample_weight=wtr).predict_proba(Xte)[:, 1]
    # Labels over-sample high teacher scores; design weights (bin share / labels in bin) make
    # the out-of-fold F1 a population estimate, and the threshold maximises that.
    sc = np.ones(ys.size, bool) if scored is None else scored
    yv, ov, wv = ys[sc], oof[sc], w[sc]
    grid_t = np.linspace(0.05, 0.99, 95)
    f1s = [f1_score(yv, ov > g, sample_weight=wv) for g in grid_t]
    thr = float(grid_t[int(np.argmax(f1s))])
    diag = {
        "n_labelled": int(ys.size),
        "n_labelled_burn": int(ys.sum()),
        "n_scored": int(sc.sum()),
        "student_oof_auc": round(float(roc_auc_score(yv, ov)), 3),
        "student_oof_f1": round(float(max(f1s)), 3),
        "student_oof_precision": round(float(precision_score(yv, ov > thr, sample_weight=wv)), 3),
        "student_oof_recall": round(float(recall_score(yv, ov > thr, sample_weight=wv)), 3),
        "student_oof_f1_unweighted": round(float(f1_score(yv, ov > thr)), 3),
        "student_threshold": thr,
    }
    Xf, yf, wf = Xs, ys, np.ones(ys.size)
    if pseudo is not None:
        Xf, yf, wf = np.vstack([Xs, pseudo[0]]), np.concatenate([ys, pseudo[1]]), np.concatenate([wf, pseudo[2]])
    return _hgb().fit(Xf, yf, sample_weight=wf), thr, diag


def extend_training_sample(t, cells, teacher_p, grid, seen: dict, per_bin: list[int]):
    """Round 2: more labels, only from haze-free scene pairs, weighted towards likely burns."""
    import pandas as pd

    train = pd.read_csv(REF / "train_sample.csv")
    clean = {d.toordinal() for d, v in seen.items() if v.mean() > 0.95}
    test = pd.read_csv(REF / "sample.csv")
    f, w = grid.factor, grid.coarse.shape[1]
    taken = set(((test["row"] // f) * w + test["col"] // f).tolist())
    ok = np.isin(t.day, list(clean)) & np.isin(t.day_prev, list(clean)) & ~np.isin(cells, list(taken))
    ok &= ~np.isin(t.pix, train["pix"].to_numpy())
    edges = np.quantile(teacher_p, np.concatenate([[0], np.cumsum(BIN_SHARE)]))
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for k, n in enumerate(per_bin):
        hi = teacher_p < edges[k + 1] if k < len(per_bin) - 1 else np.ones_like(ok)
        idx = np.flatnonzero(ok & (teacher_p >= edges[k]) & hi)
        for i in rng.choice(idx, min(n, idx.size), replace=False):
            r, c = divmod(int(t.pix[i]), grid.shape[1])
            pre, post = date.fromordinal(int(t.day_prev[i])), date.fromordinal(int(t.day[i]))
            rows.append({"bin": k, "row": r, "col": c, "pix": int(t.pix[i]), "pre": pre, "post": post})
    new = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    new.insert(0, "point_id", [f"T{len(train) + i:03d}" for i in range(len(new))])
    new["round"] = 2
    pd.concat([train.assign(round=1), new]).to_csv(REF / "train_sample.csv", index=False)
    return new


def active_sample(t, cells, p, thr, grid, n_uncertain: int, n_audit: int):
    """Round 3 (active learning): label where the student is least sure, plus an audit of mapped burns.

    Uncertainty = distance from the threshold in logit space; at most one point per 1 km cell so the
    batch is not spent on one field. The audit draw (random mapped burns) targets commission error.
    """
    import pandas as pd

    train = pd.read_csv(REF / "train_sample.csv")
    ok = ~np.isin(cells, sealed_cells(grid)) & ~np.isin(t.pix, train["pix"].to_numpy())
    logit = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))  # noqa: E731
    u = np.abs(logit(p) - logit(thr))
    idx = np.flatnonzero(ok)
    idx = idx[np.argsort(u[idx], kind="stable")]
    _, first = np.unique(cells[idx], return_index=True)
    picked = idx[np.sort(first)][:n_uncertain]
    rng = np.random.default_rng(SEED + 4)
    pool = np.setdiff1d(np.flatnonzero(ok & (p > thr)), picked)
    picked = np.concatenate([picked, rng.choice(pool, min(n_audit, pool.size), replace=False)])
    rows = []
    for k, i in enumerate(picked):
        r, c = divmod(int(t.pix[i]), grid.shape[1])
        rows.append({"bin": -1 if k < n_uncertain else -2, "row": r, "col": c, "pix": int(t.pix[i]),
                     "pre": date.fromordinal(int(t.day_prev[i])), "post": date.fromordinal(int(t.day[i]))})  # fmt: skip
    new = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    new.insert(0, "point_id", [f"T{len(train) + i:03d}" for i in range(len(new))])
    new["round"] = 3
    pd.concat([train, new]).to_csv(REF / "train_sample.csv", index=False)
    return new


def sealed_cells(grid: Grid) -> np.ndarray:
    """1 km cells holding a point of a sealed test set (reference/sample.csv, reference/testB/)."""
    import pandas as pd

    sets = [p for p in (REF / "sample.csv", REF / "testB" / "sample.csv") if p.exists()]
    if not sets:
        return np.zeros(0, int)
    test = pd.concat([pd.read_csv(p) for p in sets])
    f, w = grid.factor, grid.coarse.shape[1]
    return np.unique((test["row"] // f) * w + test["col"] // f)


def labelled_transitions(t: Transitions, cells: np.ndarray, grid: Grid):
    """All gold labels mapped to transition indices, matched on (pixel, pre, post).

    Sources: the training sample(s), plus agreed points of the retired test set A (its own
    (pre, post] interval is mapped to the pixel's transition inside it). Any label whose 1 km
    cell holds a point of the sealed test set is dropped.
    """
    import pandas as pd

    key = pd.Series(np.arange(t.pix.size), index=pd.MultiIndex.from_arrays([t.pix, t.day_prev, t.day]))
    f, w = grid.factor, grid.coarse.shape[1]
    sealed = set(sealed_cells(grid).tolist())
    out_idx, out = [], []

    train = pd.read_csv(REF / "train_sample.csv")
    y = _train_labels(train)
    m = key.reindex(pd.MultiIndex.from_arrays([
        train["pix"], [date.fromisoformat(d).toordinal() for d in train["pre"]],
        [date.fromisoformat(d).toordinal() for d in train["post"]]])).to_numpy()  # fmt: skip
    for i in np.flatnonzero((y >= 0) & ~np.isnan(m)):
        out_idx.append(int(m[i]))
        out.append({"y": int(y[i]), "source": f"train_r{int(train.get('round', pd.Series([1] * len(train)))[i])}"})

    ta = REF / "testA"
    if (ta / "labels_A.csv").exists():
        from .evaluate import _read_labels

        s = pd.read_csv(ta / "sample.csv")
        A, B = _read_labels(ta / "labels_A.csv"), _read_labels(ta / "labels_B.csv")
        s["la"], s["lb"] = s["point_id"].map(A), s["point_id"].map(B)
        s = s[(s["la"] == s["lb"]) & s["la"].isin([0, 1])]
        order = np.argsort(t.pix, kind="stable")
        sp = t.pix[order]
        for _, p in s.iterrows():
            pix = int(p["row"]) * grid.shape[1] + int(p["col"])
            cell = (int(p["row"]) // f) * w + int(p["col"]) // f
            if cell in sealed:
                continue
            lo, hi = np.searchsorted(sp, pix), np.searchsorted(sp, pix, side="right")
            cand = order[lo:hi]
            a, b = date.fromisoformat(p["pre"]).toordinal(), date.fromisoformat(p["post"]).toordinal()
            cand = cand[(t.day[cand] > a) & (t.day[cand] <= b) & (t.day_prev[cand] >= a)]
            if cand.size == 1:  # only when one transition unambiguously explains the labelled interval
                out_idx.append(int(cand[0]))
                out.append({"y": int(p["la"]), "source": "testA"})
    keep = [k for k, i in enumerate(out_idx) if cells[i] not in sealed]
    return np.array([out_idx[k] for k in keep], int), pd.DataFrame([out[k] for k in keep])


def _train_labels(train: "pd.DataFrame") -> np.ndarray:  # noqa: F821
    """Merge interpreter files (train_labels*.csv); 9 = unusable -> -1."""
    labs = {}
    for f in sorted(REF.glob("train_labels*.csv")):
        for ln in f.read_text().splitlines()[1:]:
            if ln.strip():
                pid, lab = ln.split(",", 2)[:2]
                labs[pid.strip()] = int(lab)
    y = train["point_id"].map(labs).fillna(9).astype(int).to_numpy()
    return np.where(np.isin(y, [0, 1]), y, -1)


def run(settings: Settings, cat: LocalCatalog, grid: Grid) -> dict:
    fz = settings["fusion"]
    s0, s1 = (d.isoformat() for d in settings.season)
    ard_items = [i for i in cat.items("ard-s2") if s0 <= i.properties["s2:acquisition_date"] <= s1]
    scenes = [
        (date.fromisoformat(i.properties["s2:acquisition_date"]), Path(i.assets["reflectance"].get_absolute_href()))
        for i in ard_items
    ]
    log.info("folding %d scenes", len(scenes))
    f = fold(scenes, grid.shape, fz["change_min_dnbr"], fz["change_min_dswir"])
    t = f.transitions

    fs = np.load(settings.path("ard", "modis_fire.npz"))
    fi = FireIndex(fs["days"], fs["fire"], fs["observed"], fz["fire_min_class"], fz["negative_buffer_cells"])
    fi_near = FireIndex(fs["days"], fs["fire"], fs["observed"], fz["fire_min_class"], 1)
    cells = coarse_cell(t.pix, grid)

    # ---- Stage 1: teacher (MODIS weak labels -> S2-only logistic regression)
    y = weak_labels(t, cells, fi, fz["negative_pad_days"])
    sealed = sealed_cells(grid)
    y[np.isin(cells, sealed)] = -1  # sealed test cells never shape the teacher or its threshold
    teacher, diag = fit(t.X, y, cells)
    p_t = teacher.predict_proba(t.X)[:, 1].astype(np.float32)
    # Neyman-Pearson threshold: weak labels make P uncalibrated, so cap the false-alarm
    # rate on MODIS-quiet changes (mostly harvest/tillage) instead of trusting 0.5.
    thr_t = float(np.quantile(p_t[y == 0], 1 - fz["max_false_alarm"], method="higher"))
    diag["threshold_p"] = round(thr_t, 4)
    diag["weak_label_recall_at_threshold"] = round(float((p_t[y == 1] > thr_t).mean()), 3)
    teacher_path = settings.path("products", "burn_season_teacher.tif")
    diag["teacher_map"] = write_burn_map(
        teacher_path, t, first_over(t, p_t, thr_t), p_t, cells, fi_near, settings, grid
    )

    # ---- Stage 2: student (photo-interpreted labels; MODIS now allowed as features)
    import pandas as pd

    out = settings.path("products", "burn_season.tif")
    if not (REF / "train_sample.csv").exists():
        make_training_sample(t, cells, p_t, grid, fz["train_per_bin"])
        from .evaluate import render_sheets

        render_sheets(settings, grid, pd.read_csv(REF / "train_sample.csv", parse_dates=["pre", "post"]),
                      REF / "train_chips")  # fmt: skip
    if fz.get("train_round2") and "round" not in pd.read_csv(REF / "train_sample.csv").columns:
        new = extend_training_sample(t, cells, p_t, grid, f.seen, fz["train_round2"])
        from .evaluate import render_sheets

        render_sheets(settings, grid, new.assign(pre=pd.to_datetime(new["pre"]), post=pd.to_datetime(new["post"])),
                      REF / "train_chips2")  # fmt: skip
    idx, y_lab = labelled_transitions(t, cells, grid)
    diag["label_sources"] = {k: int(v) for k, v in pd.Series(y_lab.source).value_counts().items()}
    if idx.size >= 60 and 0 < y_lab.y.sum() < idx.size:
        log.info("student: temporal + MODIS features")
        Xall = np.hstack([t.X, temporal_features(t, scenes), modis_features(t, cells, fs, fz["fire_min_class"], p_t)])
        ys = y_lab.y.to_numpy()
        scored = (y_lab.source != "train_r3").to_numpy()
        # Treat every probability-sampled transition as a draw from its teacher-score bin;
        # weight = bin share / n scored in bin. Active-learning rows get no weight (never scored).
        edges = np.quantile(p_t, np.concatenate([[0], np.cumsum(BIN_SHARE)]))
        b = np.clip(np.searchsorted(edges, p_t[idx], side="right") - 1, 0, len(BIN_SHARE) - 1)
        w = np.where(
            scored, np.array(BIN_SHARE)[b] / np.maximum(np.bincount(b[scored], minlength=len(BIN_SHARE))[b], 1), 0.0
        )
        bw = grid.coarse.shape[1] // 3 + 1
        block_of = lambda c: (c // grid.coarse.shape[1] // 3) * bw + (c % grid.coarse.shape[1]) // 3  # noqa: E731
        blocks_all = block_of(cells)

        def teacher_for(held_blocks, extra=None):
            m = ~np.isin(blocks_all, held_blocks)
            sub = np.flatnonzero(m & (y >= 0))
            tm, _ = fit(t.X[sub], y[sub], cells[sub], cv=False)
            p_extra = tm.predict_proba(t.X[extra])[:, 1] if extra is not None else None
            return tm.predict_proba(t.X[idx])[:, 1], p_extra

        base_model, thr_s, sd = _student(Xall[idx], ys, blocks_all[idx], w, teacher_for, scored=scored)
        student, chosen = base_model, "none"
        if fz.get("self_training"):
            # Capped, low-weight pseudo-labels: teacher-confident AND MODIS-consistent, away from gold blocks.
            # Excluded at 1 km-cell level (sealed test sets now touch every 3 km block); pseudo-labels carry
            # no reference information, and inside CV they are still dropped from held-out blocks.
            free = ~np.isin(cells, np.unique(cells[idx])) & ~np.isin(cells, sealed)
            hi_t, lo_t = np.quantile(p_t, [0.999, 0.2])
            rng = np.random.default_rng(SEED + 3)
            pos = np.flatnonzero(free & (p_t >= hi_t) & (Xall[:, -6] > 0))
            neg = np.flatnonzero(free & (p_t <= lo_t))
            cap = fz["self_training"]
            pi = np.concatenate([rng.choice(pos, min(cap, pos.size), replace=False),
                                 rng.choice(neg, min(cap, neg.size), replace=False)])  # fmt: skip
            pseudo = (Xall[pi], (p_t[pi] >= hi_t).astype(int), np.full(pi.size, 0.1), blocks_all[pi], pi, (hi_t, lo_t))
            st_model, st_thr, st_d = _student(Xall[idx], ys, blocks_all[idx], w, teacher_for, pseudo, scored)
            diag["self_training_trial"] = {"oof_f1": st_d["student_oof_f1"], "baseline_oof_f1": sd["student_oof_f1"],
                                           "n_pseudo": int(pi.size)}  # fmt: skip
            if st_d["student_oof_f1"] > sd["student_oof_f1"] + 0.01:  # adopt only on a real CV gain
                student, thr_s, sd, chosen = st_model, st_thr, st_d, "adopted"
            else:
                chosen = "rejected"
        diag["self_training"] = chosen
        diag.update(sd)
        p_s = student.predict_proba(Xall)[:, 1].astype(np.float32)
        al = fz.get("train_round3_active")
        train = pd.read_csv(REF / "train_sample.csv")
        if al and 3 not in set(train.get("round", pd.Series([1]))):
            active_sample(t, cells, p_s, thr_s, grid, *al)
            train = pd.read_csv(REF / "train_sample.csv")
        if al and not (REF / "train_chips3").exists():  # (re-)render if a previous run died after sampling
            from .evaluate import render_sheets

            r3 = train[train["round"] == 3]
            render_sheets(settings, grid, r3.assign(pre=pd.to_datetime(r3["pre"]), post=pd.to_datetime(r3["post"])),
                          REF / "train_chips3")  # fmt: skip
        pixel_path = settings.path("products", "burn_season_pixel.tif")
        diag["student_map"] = write_burn_map(
            pixel_path, t, first_over(t, p_s, thr_s), p_s, cells, fi_near, settings, grid
        )
        if fz.get("field_voting"):
            diag["field_map"] = field_smooth(pixel_path, out, t, scenes, settings, grid)
            diag["final_model"] = "student+fields"
        else:  # tried; see docs/MODEL_CARD.md — burns cover ~5% of a segment, voting erases them
            import shutil

            shutil.copyfile(pixel_path, out)
            diag["final_model"] = "student"
        joblib.dump(student, settings.path("products", "model.joblib"))
        final = diag.get("field_map", diag["student_map"])
    else:
        log.warning("no usable training labels yet (%d) — publishing the teacher map", int(idx.size))
        import shutil

        shutil.copyfile(teacher_path, out)
        diag["final_model"] = "teacher"
        joblib.dump(teacher, settings.path("products", "model.joblib"))
        final = diag["teacher_map"]

    seen_path = settings.path("products", "s2_seen.npz")
    np.savez_compressed(
        seen_path,
        days=np.array([d.toordinal() for d in f.seen]),
        seen=np.stack(list(f.seen.values())),
        n_obs=f.n_obs,
    )
    diag.update(
        {
            **final,
            "n_transitions": int(t.pix.size),
            "features": FEATURES + TEMPORAL_FEATURES + MODIS_FEATURES,
            "max_false_alarm": fz["max_false_alarm"],
            "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    settings.path("products", "model.json").write_text(json.dumps(diag, indent=2))
    cat.add_derived(
        "product-burn",
        "burn-season",
        datetime.combine(settings.season[0], datetime.min.time(), timezone.utc),
        settings["aoi"]["bbox"],
        {
            "burn": (out, pystac.MediaType.COG),
            "burn_teacher": (teacher_path, pystac.MediaType.COG),
            "s2_seen": (seen_path, "application/x-npz"),
            "model": (settings.path("products", "model.json"), pystac.MediaType.JSON),
        },
        ard_items + cat.items("ard-modis-fire"),
        {"fusion": fz, "features": FEATURES + TEMPORAL_FEATURES + MODIS_FEATURES},
        {"burnt_ha": diag["burnt_ha"], "final_model": diag["final_model"]},
    )
    cat.save()
    log.info("fusion: %s", {k: diag.get(k) for k in ("final_model", "burnt_ha", "student_oof_f1", "student_oof_auc")})
    return diag
