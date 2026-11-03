"""Analysis-ready data: every source lands on the analysis grid exactly once.

Sentinel-2
  * only NIR/SWIR bands (B8A, B11, B12): they see through the Oct–Nov smoke
    haze far better than visible bands, and they are what separates char from
    stubble;
  * SCL masking with one domain-specific fix — Sen2Cor routinely labels fresh
    burn scars as "dark area" (2) or "cloud shadow" (3). Dark area is kept, and
    a shadow pixel is kept unless a cloud is within ``SHADOW_CASTER_M``: a
    shadow needs a caster;
  * the AOI grid is snapped to the S2 20 m lattice, so loading is a pure crop
    (nearest, no resampling).

MODIS
  * MOD14A1 + MYD14A1 daily fire masks → one (day, y, x) stack on the 1 km
    grid, keeping the max class across Terra/Aqua, plus an "observed" flag so
    "no fire" under cloud is never mistaken for evidence of no burning.
  * MCD64A1 burn date → fine grid, for the baseline comparison only.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import odc.stac
import planetary_computer
import pystac
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import ndimage

from .catalog import LocalCatalog
from .config import Settings
from .grid import Grid
from .ingest import fetch

log = logging.getLogger(__name__)

# Blob storage occasionally drops a range request; retry instead of losing the scene.
for _k, _v in {
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
}.items():
    os.environ.setdefault(_k, _v)

S2_BANDS = ("B8A", "B11", "B12", "B02", "B03", "B04")  # SWIR/NIR for the model; visible for QA chips
SCL_VALID = {2, 4, 5, 6, 7}  # dark area, vegetation, not-vegetated, water, unclassified
SCL_SHADOW = 3
SCL_CLOUD = {8, 9, 10}  # cloud medium/high probability, thin cirrus
SHADOW_CASTER_M = 1500
ARD_PARAMS = {"bands": S2_BANDS, "scl_valid": sorted(SCL_VALID), "shadow_caster_m": SHADOW_CASTER_M, "v": 2}

# MOD14 FireMask classes
FIRE_OBSERVED = {3, 5, 7, 8, 9}  # water, land, fire (low/nominal/high) — i.e. not cloud/missing


def s2_valid_mask(scl: np.ndarray, res_m: float) -> np.ndarray:
    """SCL → validity, keeping burn scars Sen2Cor mislabels as dark/shadow."""
    valid = np.isin(scl, list(SCL_VALID))
    cloud = np.isin(scl, list(SCL_CLOUD))
    if cloud.any():
        near_cloud = ndimage.distance_transform_edt(~cloud) * res_m <= SHADOW_CASTER_M
    else:
        near_cloud = np.zeros_like(cloud)
    valid |= (scl == SCL_SHADOW) & ~near_cloud
    return valid


def _boa_offset(item: pystac.Item) -> int:
    # Processing baseline >= 04.00 (Jan 2022) adds +1000 to L2A digital numbers.
    pb = item.properties.get("s2:processing_baseline", "00.00")
    return 1000 if float(pb) >= 4.0 else 0


def build_s2_ard(item: pystac.Item, grid: Grid, out: Path) -> dict:
    ds = odc.stac.load(
        [item],
        bands=[*S2_BANDS, "SCL"],
        geobox=grid.fine,
        resampling="nearest",
        patch_url=planetary_computer.sign,
        chunks=None,
    ).isel(time=0)
    scl = ds["SCL"].values
    valid = s2_valid_mask(scl, grid.res)
    off = _boa_offset(item)
    bands = []
    for b in S2_BANDS:
        dn = ds[b].values.astype(np.int32)
        valid &= dn > 0
        bands.append(np.clip(dn - off, 1, 65535).astype(np.uint16))
    stack = np.stack(bands)
    stack[:, ~valid] = 0  # 0 == invalid in every band
    _write(out, stack, grid.fine, nodata=0, descriptions=S2_BANDS)
    return {
        "valid_fraction": round(float(valid.mean()), 4),
        "scl_cloud_fraction": round(float(np.isin(scl, list(SCL_CLOUD)).mean()), 4),
    }


HAZE_HOT_MAX = 0.06  # Haze-Optimised Transform (Zhang et al. 2002) on BOA: B02 - 0.5*B04


def read_s2_ard(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (reflectance float32 [bands, H, W] in 0..1, valid mask); band order = S2_BANDS."""
    with rasterio.open(path) as src:
        dn = src.read()
    refl = dn.astype(np.float32) / 10_000
    valid = dn[0] > 0
    if refl.shape[0] >= 6:
        # Post-harvest smoke/smog is not flagged by Sen2Cor's SCL, but it brightens the blue
        # band far more than red. Hazy-then-clear would otherwise read as "darkening" = burn.
        valid &= (refl[3] - 0.5 * refl[5]) <= HAZE_HOT_MAX
    return refl, valid


def _write(path: Path, arr: np.ndarray, gbox, nodata, descriptions=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = arr if arr.ndim == 3 else arr[None]
    profile = dict(
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=arr.dtype,
        crs=str(gbox.crs),
        transform=gbox.transform,
        nodata=nodata,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    tmp = path.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(arr)
        if descriptions:
            for i, d in enumerate(descriptions, 1):
                dst.set_band_description(i, d)
    tmp.rename(path)


def _reproject(src_path: Path, band: int, gbox, resampling=Resampling.nearest, dtype=np.uint8, fill=0):
    out = np.full(gbox.shape, fill, dtype=dtype)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, band),
            destination=out,
            dst_transform=gbox.transform,
            dst_crs=str(gbox.crs),
            resampling=resampling,
            dst_nodata=fill,
        )
    return out


def build_fire_stack(items: list[pystac.Item], grid: Grid, raw_dir: Path, start: date, end: date):
    """Daily max FireMask class across Terra+Aqua on the 1 km grid.

    Returns (days, fire_class uint8 [D, h, w], observed bool [D, h, w], max_frp float32 [D, h, w] in MW).
    """
    n = (end - start).days + 1
    days = [start + timedelta(d) for d in range(n)]
    fire = np.zeros((n, *grid.coarse.shape), np.uint8)
    observed = np.zeros((n, *grid.coarse.shape), bool)
    frp = np.zeros((n, *grid.coarse.shape), np.float32)
    for it in items:
        path = fetch(it, "FireMask", raw_dir)
        frp_path = fetch(it, "MaxFRP", raw_dir)
        with rasterio.open(path) as src:
            dates = [date.fromisoformat(d) for d in src.tags()["Dates"].split()]
        for b, d in enumerate(dates, 1):
            if not (start <= d <= end):
                continue
            cls = _reproject(path, b, grid.coarse)
            i = (d - start).days
            fire[i] = np.maximum(fire[i], cls)
            observed[i] |= np.isin(cls, list(FIRE_OBSERVED))
            raw = _reproject(frp_path, b, grid.coarse, dtype=np.int32, fill=0)
            frp[i] = np.maximum(frp[i], raw * 0.1)  # MaxFRP scale factor 0.1 -> MW
    return days, fire, observed, frp


def build_mcd64(items: list[pystac.Item], grid: Grid, raw_dir: Path) -> np.ndarray:
    """MCD64A1 burn date as an ordinal day on the fine grid (0 = unburned)."""
    out = np.zeros(grid.shape, np.int32)
    for it in items:
        path = fetch(it, "Burn_Date", raw_dir)
        doy = _reproject(path, 1, grid.fine, dtype=np.int16, fill=0)
        year = it.datetime.year if it.datetime else int(it.properties["start_datetime"][:4])
        jan1 = date(year, 1, 1).toordinal() - 1
        burned = doy > 0
        out[burned] = jan1 + doy[burned].astype(np.int32)
    return out


def run(settings: Settings, cat: LocalCatalog, grid: Grid, workers: int = 4) -> dict:
    """Bring ARD up to date with the catalogue. Returns counts of work done."""
    done = {"s2_built": 0, "s2_current": 0, "s2_failed": 0}
    todo = []
    for it in cat.items("source-s2"):
        if cat.is_current("ard-s2", f"ard-{it.id}", _phash({**ARD_PARAMS, "grid": grid.signature})):
            done["s2_current"] += 1
        else:
            todo.append(it)

    def work(it):
        out = settings.path("ard", "s2", f"{it.id}.tif")
        return it, out, build_s2_ard(it, grid, out)

    # Network-bound: overlap scene downloads; catalogue writes stay on this thread.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, it) for it in todo]
        for fut in as_completed(futures):
            try:
                it, out, stats = fut.result()
            except Exception as e:  # retain completed scenes so a retry can reuse them
                log.warning("ARD failed: %s", e)
                done["s2_failed"] += 1
                continue
            log.info("ARD %s valid=%.2f", it.id, stats["valid_fraction"])
            cat.add_derived(
                "ard-s2",
                f"ard-{it.id}",
                it.datetime,
                settings["aoi"]["bbox"],
                {"reflectance": (out, pystac.MediaType.COG)},
                [it],
                {**ARD_PARAMS, "grid": grid.signature},
                {**stats, "s2:acquisition_date": it.datetime.date().isoformat()},
            )
            done["s2_built"] += 1
            cat.save()

    if done["s2_failed"]:
        raise RuntimeError(
            f"{done['s2_failed']} Sentinel-2 scene(s) failed ARD; stopping before fusion. "
            "Completed scenes are retained. Rerun the same command to retry the missing scenes."
        )

    # MODIS layers are tiny: rebuild whenever the set of source items changes.
    start, end = settings.season
    fire_items = cat.items("source-modis-fire")
    ba_items = cat.items("source-mcd64")
    fparams = {
        "sources": sorted(i.id for i in fire_items + ba_items),
        "start": start,
        "end": end,
        "grid": grid.signature,
        "v": 2,
    }
    fire_path = settings.path("ard", "modis_fire.npz")
    if not cat.is_current("ard-modis-fire", "modis-fire-stack", _phash(fparams)):
        days, fire, observed, frp = build_fire_stack(fire_items, grid, settings.path("raw", "modis"), start, end)
        np.savez_compressed(
            fire_path, days=np.array([d.toordinal() for d in days]), fire=fire, observed=observed, frp=frp
        )
        mcd = build_mcd64(ba_items, grid, settings.path("raw", "modis"))
        np.savez_compressed(settings.path("ard", "mcd64_burn_day.npz"), burn_day=mcd)
        cat.add_derived(
            "ard-modis-fire",
            "modis-fire-stack",
            datetime.combine(start, datetime.min.time(), timezone.utc),
            settings["aoi"]["bbox"],
            {
                "fire_stack": (fire_path, "application/x-npz"),
                "mcd64_burn_day": (settings.path("ard", "mcd64_burn_day.npz"), "application/x-npz"),
            },
            fire_items + ba_items,
            fparams,
            {"fire_cell_days": int((fire >= settings["fusion"]["fire_min_class"]).sum())},
        )
        cat.save()
        done["modis_rebuilt"] = 1
    return done


def _phash(params: dict | None = None) -> str:
    from .catalog import params_hash

    return params_hash(params if params is not None else ARD_PARAMS)
