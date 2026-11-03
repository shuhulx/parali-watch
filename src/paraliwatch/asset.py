"""Build the static now-cast page (site/index.html) from the feature store.

Static on purpose: the user is a district field coordinator opening one link
on a phone on Monday morning — no server to keep alive, deployable on GitHub
Pages, and every refresh simply rewrites the page. All numbers on the page
come from SQL over the GeoParquet feature store (see ``SQL_*`` below).
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import h3
import numpy as np
import rasterio
from jinja2 import Environment, FileSystemLoader
from PIL import Image
from pyproj import Transformer
from rasterio.warp import Resampling, calculate_default_transform, reproject

from . import __version__
from .catalog import LocalCatalog
from .config import Settings
from .features import connect, week_key
from .grid import Grid
from .places import Gazetteer

log = logging.getLogger(__name__)

SQL_WEEKLY = """
SELECT week, min(week_start) AS week_start,
       sum(burnt_ha) AS burnt_ha, sum(burnt_ha_confirmed) AS confirmed_ha,
       sum(burnt_ha_s2_only) AS s2_only_ha, sum(burnt_ha_long_gap) AS long_gap_ha,
       sum(mcd64_ha) AS mcd64_ha, sum(modis_fire_cell_days) AS fire_cell_days,
       sum(s2_seen_frac * n_px) / sum(n_px) AS s2_seen, max(s2_clear_scenes) AS s2_scenes
FROM hex_week GROUP BY week ORDER BY week
"""

SQL_R7 = """
WITH r7 AS (
  SELECT week, h3_r7, sum(burnt_ha) AS burnt_ha, sum(burnt_ha_confirmed) AS confirmed_ha,
         sum(modis_fire_cell_days) AS fire_cell_days, sum(mcd64_ha) AS mcd64_ha,
         sum(s2_seen_frac * n_px) / sum(n_px) AS s2_seen, sum(n_px) AS n_px,
         sum(blind_spot) AS blind_spot
  FROM hex_week GROUP BY week, h3_r7
)
SELECT *, burnt_ha - coalesce(lag(burnt_ha) OVER (PARTITION BY h3_r7 ORDER BY week), 0) AS delta_ha
FROM r7 ORDER BY week, burnt_ha DESC
"""


def _evidence(row) -> str:
    if row.burnt_ha > 0 and row.confirmed_ha > 0:
        return "confirmed"
    if row.burnt_ha > 0:
        return "s2-only"
    if row.fire_cell_days > 0.05 and row.s2_seen < 0.6:
        return "modis-only"
    return "none"


def _overlay(settings: Settings, grid: Grid, out_dir: Path, weeks: list[date]) -> dict:
    """One transparent PNG per week of the 20 m burn pixels, warped to EPSG:4326."""
    s0 = settings.season[0].toordinal()
    with rasterio.open(settings.path("products", "burn_season.tif")) as src:
        burn_day, evidence = src.read(1).astype(np.int32), src.read(4)
        src_crs, src_tf = src.crs, src.transform
    burn_ord = np.where(burn_day > 0, burn_day + s0 - 1, 0)
    dst_tf, w, h = calculate_default_transform(
        src_crs, "EPSG:4326", grid.shape[1], grid.shape[0], *grid.fine.extent.boundingbox
    )
    bounds = None
    out = {}
    for ws in weeks:
        cls = np.zeros(grid.shape, np.uint8)
        wk = (burn_ord >= ws.toordinal()) & (burn_ord <= ws.toordinal() + 6)
        cls[wk & (evidence % 10 == 1)] = 1
        cls[wk & (evidence % 10 == 2)] = 2
        dst = np.zeros((h, w), np.uint8)
        reproject(
            cls,
            dst,
            src_transform=src_tf,
            src_crs=src_crs,
            dst_transform=dst_tf,
            dst_crs="EPSG:4326",
            resampling=Resampling.nearest,
        )
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[dst == 1] = (255, 159, 28, 235)  # S2-only
        rgba[dst == 2] = (214, 40, 40, 245)  # MODIS-confirmed
        name = f"burn_{week_key(ws)}.png"
        Image.fromarray(rgba).save(out_dir / name, optimize=True)
        out[week_key(ws)] = f"overlays/{name}"
    west, north = dst_tf.c, dst_tf.f
    east, south = west + dst_tf.a * w, north + dst_tf.e * h
    bounds = [[south, west], [north, east]]
    return {"bounds": bounds, "png": out}


def _fire_cells(settings: Settings, grid: Grid, weeks: list[date]) -> dict:
    """MODIS 1 km fire cells per week as lat/lon squares — the scale contrast is the point."""
    fs = np.load(settings.path("ard", "modis_fire.npz"))
    fire = fs["fire"] >= settings["fusion"]["fire_min_class"]
    days = [date.fromordinal(int(d)) for d in fs["days"]]
    gb = grid.coarse
    tf = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    a = gb.transform
    out = {}
    for ws in weeks:
        idx = [i for i, d in enumerate(days) if ws <= d <= ws + timedelta(days=6)]
        n = fire[idx].sum(axis=0) if idx else np.zeros(gb.shape, int)
        cells = []
        for r, c in zip(*np.nonzero(n)):
            xs = [a.c + c * a.a, a.c + (c + 1) * a.a]
            ys = [a.f + r * a.e, a.f + (r + 1) * a.e]
            ring = [tf.transform(x, y) for x, y in ((xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1]))]
            cells.append({"ring": [[round(la, 5), round(lo, 5)] for lo, la in ring], "days": int(n[r, c])})
        out[week_key(ws)] = cells
    return out


def _headline(path: Path) -> dict | None:
    """Prefer holdout B; the legacy receipt only compares summary diagnostics."""
    if not path.exists():
        return None
    ev = json.loads(path.read_text())
    hb = ev.get("holdout_b") or {}
    if "pooled" in hb:
        return {**hb, "set": "holdout B reference set"}
    return {**ev, "set": "test set 1"}


def build(settings: Settings, cat: LocalCatalog, grid: Grid) -> Path:
    con = connect(settings)
    weekly = con.sql(SQL_WEEKLY).df()
    r7 = con.sql(SQL_R7).df()
    gaz = Gazetteer()
    site = settings.site_dir
    (site / "overlays").mkdir(exist_ok=True)

    weeks = [w.date() for w in weekly["week_start"]]
    hexes = {}
    for cell in r7["h3_r7"].unique():
        lat, lon = h3.cell_to_latlng(cell)
        hexes[cell] = {
            "ring": [[round(a, 5), round(b, 5)] for a, b in h3.cell_to_boundary(cell)],
            "centre": [round(lat, 5), round(lon, 5)],
            "places": gaz.nearest(lon, lat, 2),
        }
    r7["evidence"] = [_evidence(r) for r in r7.itertuples()]
    per_week = {
        wk: [
            {
                "h": r.h3_r7,
                "b": round(float(r.burnt_ha), 1),
                "c": round(float(r.confirmed_ha), 1),
                "f": round(float(r.fire_cell_days), 2),
                "s": round(float(r.s2_seen), 2),
                "d": round(float(r.delta_ha), 1),
                "m": round(float(r.mcd64_ha), 1),
                "e": r.evidence,
                "x": round(float(r.blind_spot), 2),
            }
            for r in g.itertuples()
        ]
        for wk, g in r7.groupby("week")
    }
    evaluation_path = settings.path("products", "evaluation.json")
    model = json.loads(settings.path("products", "model.json").read_text())
    data = {
        "weeks": [
            {
                **{
                    k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v)
                    for k, v in row.items()
                    if k != "week_start"
                },
                "start": str(row["week_start"].date()),
                "end": str(row["week_start"].date() + timedelta(days=6)),
            }
            for row in weekly.to_dict("records")
        ],
        "hexes": hexes,
        "per_week": per_week,
        "overlay": _overlay(settings, grid, site / "overlays", weeks),
        "fire": _fire_cells(settings, grid, weeks),
        "aoi": settings["aoi"]["bbox"],
        "evaluation": _headline(evaluation_path),
        "model": model,
        "lineage": {
            "s2_scenes": len(cat.items("ard-s2")),
            "modis_fire_files": len(cat.items("source-modis-fire")),
            "mcd64_files": len(cat.items("source-mcd64")),
            "s2_first": min(i.properties["s2:acquisition_date"] for i in cat.items("ard-s2")),
            "s2_last": max(i.properties["s2:acquisition_date"] for i in cat.items("ard-s2")),
        },
        "as_of": settings.raw.get("asset", {}).get("default_week"),
        "codes": settings.codes.describe(),
        "version": __version__,
        "season": [d.isoformat() for d in settings.season],
    }
    env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=False)
    html = env.get_template("index.html.j2").render(
        data=data, aoi_name=settings["aoi"]["name"]
    )
    out = site / "index.html"
    out.write_text(html)
    log.info("asset written: %s (%.0f kB)", out, out.stat().st_size / 1024)
    return out
