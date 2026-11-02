"""One canonical analysis grid. Every layer is resampled onto it exactly once.

The fine grid is UTM 43N at 20 m. The MODIS grid is the *same* box coarsened
by an integer factor (50 -> 1 km), so a fine pixel's MODIS cell is just
``(row // f, col // f)`` — alignment by construction, no per-pixel lookups.
"""

from __future__ import annotations

import math
from functools import cached_property

import numpy as np
from odc.geo.geobox import GeoBox
from odc.geo.geom import BoundingBox
from pyproj import Transformer

from .config import Settings


class Grid:
    def __init__(self, settings: Settings):
        aoi = settings["aoi"]
        self.crs = aoi["crs"]
        self.res = float(aoi["resolution"])
        self.factor = int(aoi["modis_factor"])
        step = self.res * self.factor
        bb = BoundingBox(*aoi["bbox"], crs="EPSG:4326").to_crs(self.crs)
        snapped = BoundingBox(
            math.floor(bb.left / step) * step,
            math.floor(bb.bottom / step) * step,
            math.ceil(bb.right / step) * step,
            math.ceil(bb.top / step) * step,
            crs=self.crs,
        )
        self.fine: GeoBox = GeoBox.from_bbox(snapped, crs=self.crs, resolution=self.res)
        self.coarse: GeoBox = GeoBox.from_bbox(snapped, crs=self.crs, resolution=step)
        assert self.fine.shape[0] == self.coarse.shape[0] * self.factor
        assert self.fine.shape[1] == self.coarse.shape[1] * self.factor

    @property
    def signature(self) -> str:
        """Identity of the grid; part of every cache key so an AOI change invalidates caches."""
        t = self.fine.transform
        return f"{self.crs}|{t.a},{t.c},{t.e},{t.f}|{self.shape[0]}x{self.shape[1]}|f{self.factor}"

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.fine.shape)

    @property
    def pixel_area_ha(self) -> float:
        return self.res * self.res / 10_000

    def upsample(self, coarse: np.ndarray) -> np.ndarray:
        """Broadcast a coarse (MODIS-grid) array onto the fine grid."""
        f = self.factor
        return np.repeat(np.repeat(coarse, f, axis=-2), f, axis=-1)

    @cached_property
    def fine_lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        return _centres_lonlat(self.fine, self.crs)

    @cached_property
    def coarse_lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        return _centres_lonlat(self.coarse, self.crs)


def _centres_lonlat(gbox: GeoBox, crs: str) -> tuple[np.ndarray, np.ndarray]:
    coords = gbox.coordinates
    ys = coords["y"].values
    xs = coords["x"].values
    xx, yy = np.meshgrid(xs, ys)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(xx, yy)
    return lon, lat
