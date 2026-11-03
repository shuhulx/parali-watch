"""Honest evaluation: stratified reference sample + design-based accuracy.

Protocol (after Olofsson et al. 2014, *Good practices for estimating area and
assessing accuracy of land change*):

* **Population** — pixels clear in both scenes of a reference interval
  (pre, post]. Two intervals: one at peak burning before the smog, one that
  spans the smog gap (where Sentinel-2 is expected to struggle).
* **Strata** from the fused map: mapped-burn / changed-not-mapped / stable.
  Rare-class strata are over-sampled; stratum weights undo that.
* **Reference labels** are photo-interpreted from Sentinel-2 SWIR chips by two
  independent interpreters who see neither the stratum nor any model output
  (sheets are shuffled). Agreement (Cohen's kappa) is reported; points the two
  disagree on are excluded and counted.
* **Every method is scored on the same points** with stratum-weighted
  estimators (valid for any map, not just the one that defined the strata),
  with 95% CIs from a within-stratum bootstrap.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .ard import read_s2_ard
from .config import Settings
from .fusion import FireIndex, indices
from .grid import Grid

log = logging.getLogger(__name__)

REF = Path("reference")
SEED = 7
STRATA = {1: "mapped burn", 2: "teacher-only burn", 3: "other change", 4: "stable"}
METHODS = {
    "fusion": "S2 x MODIS student + field voting (final)",
    "fusion_pixel": "S2 x MODIS student, per pixel",
    "fusion_teacher": "MODIS-taught S2 teacher",
    "fusion_v1": "v1 teacher (NBR-only candidates)",
    "candidate": "endpoint change gate (diagnostic)",
    "s2_dnbr": "Sentinel-2 dNBR > 0.27 only",
    "modis_af": "MODIS active fire, 1 km cell",
    "mcd64": "MODIS MCD64A1 burned area",
}
DNBR_BASELINE = 0.27  # Key & Benson (2006) low/moderate severity break


def _scene_paths(settings: Settings) -> dict[date, Path]:
    """Active ARD per acquisition date, resolved through the catalogue (not the filesystem)."""
    from .catalog import LocalCatalog

    cat = LocalCatalog(settings.path("catalog", "catalog.json").parent)
    return {
        date.fromisoformat(i.properties["s2:acquisition_date"]): Path(i.assets["reflectance"].get_absolute_href())
        for i in cat.items("ard-s2")
    }


def _layers(settings: Settings, grid: Grid, pre: date, post: date) -> dict[str, np.ndarray]:
    """Per-pixel decisions of every method for one interval, plus strata."""
    scenes = _scene_paths(settings)
    r0, v0 = read_s2_ard(scenes[pre])
    r1, v1 = read_s2_ard(scenes[post])
    dnbr = indices(r0)["nbr"] - indices(r1)["nbr"]
    valid = v0 & v1
    s0 = settings.season[0].toordinal()

    def burnt_in(name: str) -> np.ndarray:
        path = settings.path("products", name)
        if not path.exists():
            return np.zeros(grid.shape, bool)
        with rasterio.open(path) as src:
            burn_day = src.read(1).astype(np.int32)
        burn_ord = np.where(burn_day > 0, burn_day + s0 - 1, 0)
        return (burn_ord > pre.toordinal()) & (burn_ord <= post.toordinal())

    fusion = burnt_in("burn_season.tif")
    mcd = np.load(settings.path("ard", "mcd64_burn_day.npz"))["burn_day"]
    mcd64 = (mcd > pre.toordinal()) & (mcd <= post.toordinal())
    fs = np.load(settings.path("ard", "modis_fire.npz"))
    fi = FireIndex(fs["days"], fs["fire"], fs["observed"], settings["fusion"]["fire_min_class"], 0)
    h, w = grid.coarse.shape
    cell_fire = fi.own(np.arange(h * w), np.full(h * w, pre.toordinal() + 1), np.full(h * w, post.toordinal()))
    modis_af = grid.upsample(cell_fire.reshape(h, w) > 0)
    teacher = burnt_in("burn_season_teacher.tif")
    dswir = r0[1] - r1[1]
    changed = (dnbr > settings["fusion"]["change_min_dnbr"]) | (dswir > settings["fusion"]["change_min_dswir"])
    strata = np.select([fusion, teacher, changed], [1, 2, 3], 4)
    strata[~valid] = 0
    return {
        "strata": strata,
        "fusion": fusion,
        "fusion_pixel": burnt_in("burn_season_pixel.tif"),
        "fusion_teacher": teacher,
        "fusion_v1": burnt_in("burn_season_v1.tif"),
        "s2_dnbr": dnbr > DNBR_BASELINE,
        "candidate": changed,  # endpoint-change diagnostic, not an exact recall ceiling
        "modis_af": modis_af,
        "mcd64": mcd64,
        "refl_pre": r0,
        "refl_post": r1,
    }


def make_sample(settings: Settings, grid: Grid, ev: dict | None = None, ref: Path = REF,
                prefix: str = "P", seed: int = SEED) -> pd.DataFrame:  # fmt: skip
    ev = ev or settings["evaluation"]
    rng = np.random.default_rng(seed)
    rows = []
    lon, lat = grid.fine_lonlat
    # Spatial independence: no test pixel inside a 1 km cell that holds a training point
    # (or, for a later test set, a point of any earlier one).
    train_cells = np.zeros(grid.shape, bool)
    f = grid.factor
    for prior in [REF / "train_sample.csv", REF / "testA" / "sample.csv", REF / "sample.csv"]:
        if prior.exists() and prior != ref / "sample.csv":
            tr = pd.read_csv(prior)
            for r, c in zip(tr["row"] // f, tr["col"] // f):
                train_cells[r * f : (r + 1) * f, c * f : (c + 1) * f] = True
    train_cells = train_cells.ravel()
    for k, (pre, post) in enumerate(ev["intervals"]):
        pre, post = date.fromisoformat(pre), date.fromisoformat(post)
        L = _layers(settings, grid, pre, post)
        for h, n in zip((1, 2, 3, 4), ev["n_per_stratum"]):
            idx = np.flatnonzero((L["strata"].ravel() == h) & ~train_cells)
            pick = rng.choice(idx, min(n, idx.size), replace=False)
            for p in pick:
                r, c = divmod(int(p), grid.shape[1])
                rows.append(
                    dict(
                        interval=k,
                        pre=pre,
                        post=post,
                        stratum=h,
                        N_h=idx.size,
                        row=r,
                        col=c,
                        lon=round(float(lon[r, c]), 6),
                        lat=round(float(lat[r, c]), 6),
                    )
                )
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)  # blind order
    df.insert(0, "point_id", [f"{prefix}{i:03d}" for i in range(len(df))])
    ref.mkdir(parents=True, exist_ok=True)
    df.to_csv(ref / "sample.csv", index=False)
    meta = {"grid": grid.signature, "seed": seed, "drawn": date.today().isoformat(), "intervals": ev["intervals"]}
    (ref / "sample_meta.json").write_text(json.dumps(meta))
    return df


def render_chips(settings: Settings, grid: Grid, ref: Path = REF) -> list[Path]:
    return render_sheets(settings, grid, pd.read_csv(ref / "sample.csv", parse_dates=["pre", "post"]), ref / "chips")


PANELS = [("TRUE", (5, 4, 3), 0.18, 0.7), ("SWIR", (2, 0, 1), 0.45, 0.8)]


class _Chips:
    """Reads ARD scenes (small LRU) and crops rendered chips around a pixel. Shared by sheets and the label page."""

    def __init__(self, settings: Settings, grid: Grid, half: int = 14):
        self.scenes, self.grid, self.half = _scene_paths(settings), grid, half
        self.cache: dict[date, np.ndarray] = {}

    def scene(self, d: date) -> np.ndarray:
        if d not in self.cache:
            if len(self.cache) > 6:
                self.cache.pop(next(iter(self.cache)))
            self.cache[d] = read_s2_ard(self.scenes[d])[0]
        return self.cache[d]

    def crop(self, d: date, r: int, c: int, bands, scale, gamma):
        H, W = self.grid.shape
        h = self.half
        r0, r1, c0, c1 = max(r - h, 0), min(r + h + 1, H), max(c - h, 0), min(c + h + 1, W)
        chip = np.stack([self.scene(d)[b, r0:r1, c0:c1] for b in bands], -1)
        return np.clip(chip / scale, 0, 1) ** gamma, (r - r0, c - c0)


def render_sheets(settings: Settings, grid: Grid, df: pd.DataFrame, out_dir: Path, per_sheet: int = 8,
                  half: int = 14) -> list[Path]:  # fmt: skip
    """Blind labelling sheets. Per point: true colour PRE | POST, SWIR PRE | POST. No model output shown."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    chips = _Chips(settings, grid, half)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    panels = PANELS
    df = df.sort_values("point_id").reset_index(drop=True)
    for s in range(0, len(df), per_sheet):
        sub = df.iloc[s : s + per_sheet]
        fig, axes = plt.subplots(4, 8, figsize=(16, 9.2))
        for ax in axes.ravel():
            ax.axis("off")
        for j, (_, p) in enumerate(sub.iterrows()):
            row, base = divmod(j, 2)
            for k, (name, bands, scale, gamma) in enumerate(panels):
                for side, d in enumerate((p["pre"].date(), p["post"].date())):
                    ax = axes[row, base * 4 + k * 2 + side]
                    img, (cy, cx) = chips.crop(d, int(p["row"]), int(p["col"]), bands, scale, gamma)
                    ax.imshow(img, interpolation="nearest")
                    ax.add_patch(Rectangle((cx - 1.5, cy - 1.5), 3, 3, fill=False, ec="cyan", lw=1.3))
                    ax.set_title(f"{p['point_id']} {name} {'PRE' if side == 0 else 'POST'} {d:%d %b}", fontsize=8.5)
        fig.suptitle(
            "Per point: TRUE colour (R=B04 G=B03 B=B02) pre|post, SWIR (R=B12 G=B8A B=B11) pre|post · 20 m px · "
            "chip 580 m · cyan = 3x3 px under test",
            fontsize=9,
        )
        fig.tight_layout()
        path = out_dir / f"sheet_{s // per_sheet:02d}.png"
        fig.savefig(path, dpi=85)
        plt.close(fig)
        paths.append(path)
    return paths


def render_label_page(settings: Settings, grid: Grid, ref: Path, out: Path | None = None, half: int = 20) -> Path:
    """Self-contained HTML for a human interpreter: same four chips per point, keys 0/1/9, CSV export.

    Labels autosave in the browser and export as ``labels_<name>.csv`` in the same format the
    pipeline reads, so a human set slots in as interpreter A, B or C. No model output is embedded.
    """
    import base64
    import io

    from PIL import Image

    df = pd.read_csv(ref / "sample.csv", parse_dates=["pre", "post"]).sort_values("point_id")
    chips, pts = _Chips(settings, grid, half), []
    for _, p in df.iterrows():
        imgs = []
        for name, bands, scale, gamma in PANELS:
            for side, d in enumerate((p["pre"].date(), p["post"].date())):
                arr, (cy, cx) = chips.crop(d, int(p["row"]), int(p["col"]), bands, scale, gamma)
                buf = io.BytesIO()  # native 20 m pixels; the page upscales them pixelated and draws the box
                Image.fromarray((arr * 255).astype(np.uint8)).save(buf, "PNG", optimize=True)
                h, w = arr.shape[:2]
                box = [round(100 * (cx - 1) / w, 2), round(100 * (cy - 1) / h, 2), round(300 / w, 2), round(300 / h, 2)]
                imgs.append({"t": f"{name} {'PRE' if side == 0 else 'POST'} {d:%d %b}", "box": box,
                             "src": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()})  # fmt: skip
        pts.append({"id": p["point_id"], "imgs": imgs})
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=False)
    km = round((2 * half + 1) * 0.02, 2)
    html = env.get_template("label.html.j2").render(points=json.dumps(pts), set_name=ref.name, km=km)
    out = out or ref / "label.html"
    out.write_text(html)
    return out


def _estimate(df: pd.DataFrame, col: str, ref: str = "ref", n_boot: int = 2000) -> dict:
    """Stratum-weighted precision / recall / F1 / burnt proportion with bootstrap CIs."""
    strata = df.groupby(["interval", "stratum"])
    N = df.drop_duplicates(["interval", "stratum"])["N_h"].sum()

    def stats(frame):
        p11 = p1_ = p_1 = agree = 0.0
        for (_, _), g in frame.groupby(["interval", "stratum"]):
            W = g["N_h"].iloc[0] / N
            m, r = g[col].to_numpy(bool), g[ref].to_numpy(bool)
            p11 += W * np.mean(m & r)
            agree += W * np.mean(m == r)
            p1_ += W * np.mean(m)
            p_1 += W * np.mean(r)
        prec = p11 / p1_ if p1_ else np.nan
        rec = p11 / p_1 if p_1 else np.nan
        f1 = 2 * p11 / (p1_ + p_1) if (p1_ + p_1) else np.nan
        return np.array([prec, rec, f1, p1_, p_1, agree])

    point = stats(df)
    rng = np.random.default_rng(SEED)
    boots = []
    groups = [g for _, g in strata]
    for _ in range(n_boot):
        boots.append(
            stats(pd.concat([g.sample(len(g), replace=True, random_state=rng.integers(1 << 31)) for g in groups]))
        )
    lo, hi = np.nanpercentile(np.array(boots), [2.5, 97.5], axis=0)
    names = ["precision", "recall", "f1", "mapped_fraction", "reference_fraction", "overall_accuracy"]
    return {n: [round(float(v), 3), round(float(a), 3), round(float(b), 3)] for n, v, a, b in zip(names, point, lo, hi)}


def _read_labels(path: Path) -> pd.Series:
    """point_id,label[,free-text note] — notes may contain commas, so only split twice."""
    rows = [ln.split(",", 2)[:2] for ln in path.read_text().splitlines()[1:] if ln.strip()]
    return pd.Series({pid.strip(): int(lab) for pid, lab in rows})


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = np.mean(a == b)
    pe = sum(np.mean(a == k) * np.mean(b == k) for k in np.union1d(a, b))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def run(settings: Settings, grid: Grid) -> dict:
    """Score every test set. Holdout B is drawn (and sealed) on first run, before it is ever scored."""
    result = score(settings, grid)
    hb = settings["evaluation"].get("holdout_b")
    if hb:
        ref = Path(hb["dir"])
        if not (ref / "sample.csv").exists():
            make_sample(settings, grid, hb, ref, prefix="B", seed=SEED + 100)
            render_chips(settings, grid, ref)
        result["holdout_b"] = score(settings, grid, ref)
        if "pooled" in result["holdout_b"]:
            result["holdout_b"]["scored_once"] = _receipt(settings, ref, result["holdout_b"])
    Path(settings.path("products", "evaluation.json")).write_text(json.dumps(result, indent=2))
    return result


def model_fingerprint(model: dict) -> str:
    from .catalog import params_hash

    return params_hash({k: v for k, v in model.items() if k != "built"})


def _receipt(settings: Settings, ref: Path, result: dict) -> bool:
    """A holdout is "scored once" only for the model that first scored it.

    The first scoring writes ``scoring_receipt.json`` (model fingerprint + headline F1). Any later
    evaluation with a different model still reports numbers, but loses the scored-once claim.
    """
    path = ref / "scoring_receipt.json"
    fp = model_fingerprint(json.loads(settings.path("products", "model.json").read_text()))
    if not path.exists():
        path.write_text(json.dumps({"model_fingerprint": fp, "first_scored": date.today().isoformat(),
                                    "f1": result["pooled"]["fusion"]["f1"]}, indent=2))  # fmt: skip
        return True
    return json.loads(path.read_text())["model_fingerprint"] == fp


def score(settings: Settings, grid: Grid, ref: Path = REF) -> dict:
    sample_path = ref / "sample.csv"
    if not sample_path.exists():
        make_sample(settings, grid)
        render_chips(settings, grid)
    df = pd.read_csv(sample_path, parse_dates=["pre", "post"])
    meta = json.loads((ref / "sample_meta.json").read_text())
    if meta["grid"] != grid.signature:  # row/col and stratum sizes are only valid on their own grid
        raise RuntimeError(f"reference sample in {ref} was drawn on a different grid/AOI; delete it to redraw")
    la, lb = ref / "labels_A.csv", ref / "labels_B.csv"
    if not (la.exists() and lb.exists()):
        return {"status": "awaiting labels", "sheets": str(ref / "chips"), "points": len(df)}

    A, B = _read_labels(la), _read_labels(lb)
    # The retired v1 raster is optional and is not rebuilt on a fresh clone.
    # A missing prediction is unavailable, not a classifier that predicts no burns.
    unavailable = [] if settings.path("products", "burn_season_v1.tif").exists() else ["fusion_v1"]
    methods = [m for m in METHODS if m not in unavailable]
    df["label_A"], df["label_B"] = df["point_id"].map(A), df["point_id"].map(B)
    both = df[(df["label_A"].isin([0, 1])) & (df["label_B"].isin([0, 1]))]
    k = kappa(both["label_A"].to_numpy(), both["label_B"].to_numpy())
    agreed = both[both["label_A"] == both["label_B"]].copy()
    agreed["ref"] = agreed["label_A"].astype(bool)
    agreed["ref_source"] = "A=B"
    # Blind adjudication: a third interpreter (C) sees only the disputed/unclear points.
    n_adjudicated = 0
    if (ref / "labels_C.csv").exists():
        C = _read_labels(ref / "labels_C.csv")
        disputed = df[~df["point_id"].isin(agreed["point_id"])].copy()
        disputed["label_C"] = disputed["point_id"].map(C)
        disputed = disputed[disputed["label_C"].isin([0, 1])]
        disputed["ref"] = disputed["label_C"].astype(bool)
        disputed["ref_source"] = "C"
        n_adjudicated = len(disputed)
        agreed = pd.concat([agreed, disputed], ignore_index=True)
    # Dropping disputed points is not neutral (hard cases may cluster in one stratum),
    # so the headline is bracketed by resolving every dispute as A's and as B's call.
    sens = {}
    for who in ("A", "B"):
        alt = both.copy()
        alt["ref"] = alt[f"label_{who}"].astype(bool)
        sens[who] = alt

    # Every method's decision at every point.
    for _i, g in agreed.groupby("interval"):
        L = _layers(settings, grid, g["pre"].iloc[0].date(), g["post"].iloc[0].date())
        for m in methods:
            agreed.loc[g.index, m] = L[m][g["row"].to_numpy(), g["col"].to_numpy()]
    for m in methods:
        agreed[m] = agreed[m].astype(bool)
    for frame in sens.values():
        for _i, g in frame.groupby("interval"):
            L = _layers(settings, grid, g["pre"].iloc[0].date(), g["post"].iloc[0].date())
            for m in methods:
                frame.loc[g.index, m] = L[m][g["row"].to_numpy(), g["col"].to_numpy()]
        for m in methods:
            frame[m] = frame[m].astype(bool)

    result = {
        "unavailable_methods": unavailable,
        "n_points": int(len(df)),
        "n_both_confident": int(len(both)),
        "n_agreed": int(len(agreed)),
        "n_adjudicated_by_C": int(n_adjudicated),
        "n_unclear_or_disagree": int(len(df) - len(agreed)),
        "cohens_kappa": round(k, 3),
        "intervals": [
            [str(pd.Timestamp(a).date()), str(pd.Timestamp(b).date())]
            for a, b in df.groupby("interval")[["pre", "post"]].first().itertuples(index=False, name=None)
        ],
        "pooled": {m: _estimate(agreed, m) for m in methods},
        "by_interval": {str(i): {m: _estimate(g, m) for m in methods} for i, g in agreed.groupby("interval")},
        "sensitivity_disputes_as_A": {m: _estimate(sens["A"], m, n_boot=300) for m in methods},
        "sensitivity_disputes_as_B": {m: _estimate(sens["B"], m, n_boot=300) for m in methods},
        "per_stratum_agreed_n": {
            f"{k[0]}-{STRATA[k[1]]}": int(v) for k, v in agreed.groupby(["interval", "stratum"]).size().items()
        },
        "note": "values are [estimate, 2.5%, 97.5%]; stratum-weighted (Olofsson et al. 2014)",
    }
    agreed.to_csv(ref / "scored_points.csv", index=False)
    return result
