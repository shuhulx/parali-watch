"""H3 x week feature store (GeoParquet, hive-partitioned by week; DuckDB to query).

H3 is the join key that lets a 20 m raster, a 1 km raster and a 500 m raster
meet in one table without another resampling step: every fine pixel is
assigned to a resolution-8 hex once (cached), and everything is a bincount.
Res-8 hexes roll up exactly to res-7 parents for the dispatch list.

Partitions are rewritten per week, so a refresh replaces only the weeks it
touched and readers never see a half-written week.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from datetime import date, timedelta

import duckdb
import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Polygon

from .catalog import LocalCatalog
from .config import Settings
from .grid import Grid

log = logging.getLogger(__name__)


def week_key(d: date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


def hex_index(settings: Settings, grid: Grid, res: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel integer code into ``cells`` (the sorted unique H3 ids). Cached on disk."""
    key = hashlib.sha256(grid.signature.encode()).hexdigest()[:10]
    cache = settings.path("cache", f"h3_r{res}_{key}.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["codes"], z["cells"]
    lon, lat = grid.fine_lonlat
    ids = np.fromiter(
        (h3.str_to_int(h3.latlng_to_cell(a, o, res)) for a, o in zip(lat.ravel(), lon.ravel())),
        dtype=np.uint64,
        count=lat.size,
    )
    cells, codes = np.unique(ids, return_inverse=True)
    np.savez_compressed(cache, codes=codes.astype(np.int32), cells=cells)
    return codes.astype(np.int32), cells


def _hex_polygon(cell: int) -> Polygon:
    return Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(h3.int_to_str(int(cell)))])


def run(settings: Settings, cat: LocalCatalog, grid: Grid) -> dict:
    res = settings["features"]["h3_resolution"]
    parent_res = settings["features"]["rollup_resolution"]
    codes, cells = hex_index(settings, grid, res)
    n_hex = cells.size
    count = lambda mask: np.bincount(codes[mask.ravel()], minlength=n_hex)  # noqa: E731
    n_px = np.bincount(codes, minlength=n_hex)
    ha = grid.pixel_area_ha
    start, end = settings.season
    s0 = start.toordinal()

    with rasterio.open(settings.path("products", "burn_season.tif")) as src:
        burn_day, _, prob, evidence, _ = src.read()
    burn_ord = np.where(burn_day > 0, burn_day.astype(np.int32) + s0 - 1, 0)
    mcd = np.load(settings.path("ard", "mcd64_burn_day.npz"))["burn_day"]
    seen = np.load(settings.path("products", "s2_seen.npz"))
    seen_days = [date.fromordinal(int(d)) for d in seen["days"]]
    fs = np.load(settings.path("ard", "modis_fire.npz"))
    fire = fs["fire"] >= settings["fusion"]["fire_min_class"]
    fire_days = [date.fromordinal(int(d)) for d in fs["days"]]
    px_per_cell = grid.factor**2

    weeks = sorted({week_start(start + timedelta(i)) for i in range((end - start).days + 1)})
    base = pd.DataFrame(
        {
            "h3_r8": [h3.int_to_str(int(c)) for c in cells],
            "n_px": n_px,
        }
    )
    base["h3_r7"] = [h3.cell_to_parent(c, parent_res) for c in base["h3_r8"]]
    geoms = [_hex_polygon(c) for c in cells]

    root = settings.path("features", "hex_week", "x").parent
    written = 0
    keep = {f"week={week_key(ws)}" for ws in weeks}
    if root.exists():  # a shortened season must not leave stale weeks queryable
        for old in root.iterdir():
            if old.name not in keep:
                shutil.rmtree(old, ignore_errors=True)
    for ws in weeks:
        we = min(ws + timedelta(days=6), end)
        ws_eff = max(ws, start)
        in_week = (burn_ord >= ws.toordinal()) & (burn_ord <= we.toordinal())
        ev = evidence % 10
        long_gap = evidence >= 10
        df = base.copy()
        df["week"] = week_key(ws)
        df["week_start"] = pd.Timestamp(ws)
        df["burnt_ha"] = count(in_week) * ha
        df["burnt_ha_confirmed"] = count(in_week & (ev == 2)) * ha
        df["burnt_ha_s2_only"] = count(in_week & (ev == 1)) * ha
        df["burnt_ha_long_gap"] = count(in_week & long_gap) * ha
        df["mcd64_ha"] = count((mcd >= ws_eff.toordinal()) & (mcd <= we.toordinal())) * ha
        wk_seen = [seen["seen"][i] for i, d in enumerate(seen_days) if ws <= d <= we]
        df["s2_clear_scenes"] = len(wk_seen)
        df["s2_seen_frac"] = (count(np.any(wk_seen, axis=0)) / np.maximum(n_px, 1)) if wk_seen else 0.0
        wk_fire = [fire[i] for i, d in enumerate(fire_days) if ws <= d <= we]
        fire_cd = grid.upsample(np.sum(wk_fire, axis=0)) if wk_fire else np.zeros(grid.shape)
        # MODIS fire cell-days, area-weighted onto hexes (additive under roll-up).
        df["modis_fire_cell_days"] = np.bincount(codes, weights=fire_cd.ravel(), minlength=n_hex) / px_per_cell
        df["p_mean"] = (
            np.divide(
                np.bincount(codes[in_week.ravel()], weights=prob.ravel()[in_week.ravel()], minlength=n_hex),
                np.maximum(count(in_week), 1),
            )
            / 100
        )
        # Blind-spot score: MODIS says fire, Sentinel-2 could not look. These are the
        # clusters a pure optical map would silently drop during smog weeks.
        unseen = ~np.any(wk_seen, axis=0) if wk_seen else np.ones(grid.shape, bool)
        df["blind_spot"] = np.bincount(codes, weights=(fire_cd * unseen).ravel(), minlength=n_hex) / px_per_cell
        gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
        part = root / f"week={week_key(ws)}"
        tmp = root / f".tmp-{week_key(ws)}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        gdf.drop(columns="week").to_parquet(tmp / "part-0.parquet", index=False)
        shutil.rmtree(part, ignore_errors=True)
        tmp.rename(part)
        written += 1

    log.info("feature store: %d hexes x %d weeks", n_hex, written)
    return {"hexes": int(n_hex), "weeks": written, "path": str(root)}


def connect(settings: Settings) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    root = settings.path("features", "hex_week", "x").parent
    con.sql(
        f"""CREATE VIEW hex_week AS
            SELECT * EXCLUDE (geometry) FROM read_parquet('{root}/*/*.parquet', hive_partitioning = true)"""
    )
    con.sql(
        """CREATE VIEW hex_season AS
           SELECT h3_r8, any_value(h3_r7) AS h3_r7, sum(burnt_ha) AS burnt_ha,
                  sum(burnt_ha_confirmed) AS burnt_ha_confirmed, sum(mcd64_ha) AS mcd64_ha,
                  sum(modis_fire_cell_days) AS modis_fire_cell_days
           FROM hex_week GROUP BY h3_r8"""
    )
    return con
