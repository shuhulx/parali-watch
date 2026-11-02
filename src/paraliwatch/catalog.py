"""Local STAC catalogue: the pipeline's single source of truth.

Every remote scene the pipeline has *seen* and every file it has *made* is a
STAC Item here. Derived items carry ``derived_from`` links to their inputs and
a ``paraliwatch:params`` hash, so

* refresh = "process source items with no current derived item" (idempotent),
* changing a model parameter invalidates exactly the products it touches,
* any pixel in the dashboard can be traced back to the scene IDs behind it.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pystac

from . import __version__

COLLECTIONS = {
    "source-s2": "Sentinel-2 L2A scenes over the AOI tile (remote, Planetary Computer).",
    "source-modis-fire": "MOD14A1/MYD14A1 8-day files of daily 1 km fire masks (remote).",
    "source-mcd64": "MCD64A1 monthly 500 m burned area (remote) — used only as a baseline.",
    "ard-s2": "Cloud/shadow-masked S2 SWIR/NIR reflectance on the analysis grid.",
    "ard-modis-fire": "Daily Terra+Aqua fire mask stacked on the 1 km analysis grid.",
    "product-burn": "Fused per-scene burn detections (Sentinel-2 x MODIS).",
}


def params_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


class LocalCatalog:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.href = self.root / "catalog.json"
        if self.href.exists():
            self.cat = pystac.Catalog.from_file(str(self.href))
        else:
            self.cat = pystac.Catalog(
                id="paraliwatch",
                description="ParaliWatch — stubble-burn monitoring, Ludhiana–Patiala belt.",
            )
        self._cols: dict[str, pystac.Collection] = {}
        for cid, desc in COLLECTIONS.items():
            col = self.cat.get_child(cid)
            if col is None:
                col = pystac.Collection(
                    id=cid,
                    description=desc,
                    extent=pystac.Extent(
                        pystac.SpatialExtent([[-180, -90, 180, 90]]),
                        pystac.TemporalExtent([[None, None]]),
                    ),
                )
                self.cat.add_child(col)
            self._cols[cid] = col
        self._index: dict[str, dict[str, pystac.Item]] = {
            cid: {it.id: it for it in col.get_items()} for cid, col in self._cols.items()
        }

    # -- queries -----------------------------------------------------------
    def items(self, collection: str) -> list[pystac.Item]:
        return sorted(self._index[collection].values(), key=lambda i: (i.datetime or _start(i), i.id))

    def get(self, collection: str, item_id: str) -> pystac.Item | None:
        return self._index[collection].get(item_id)

    def is_current(self, collection: str, item_id: str, phash: str) -> bool:
        it = self.get(collection, item_id)
        return (
            it is not None
            and it.properties.get("paraliwatch:params") == phash
            and all(Path(a.get_absolute_href()).exists() for a in it.assets.values())
        )

    # -- writes ------------------------------------------------------------
    def add(self, collection: str, item: pystac.Item) -> bool:
        """Add or replace an item. Returns True if it was new."""
        col = self._cols[collection]
        new = item.id not in self._index[collection]
        if not new:
            col.remove_item(item.id)
        col.add_item(item)
        self._index[collection][item.id] = item
        return new

    def add_derived(
        self,
        collection: str,
        item_id: str,
        when: datetime,
        bbox: list[float],
        assets: dict[str, tuple[Path, str]],
        derived_from: list[pystac.Item],
        params: dict,
        properties: dict | None = None,
    ) -> pystac.Item:
        geom = _bbox_geom(bbox)
        props = {
            "paraliwatch:params": params_hash(params),
            "processing:software": {"paraliwatch": __version__},
            "processing:datetime": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(properties or {}),
        }
        item = pystac.Item(item_id, geom, bbox, when, props)
        for name, (path, media) in assets.items():
            item.add_asset(name, pystac.Asset(str(Path(path).resolve()), media_type=media, roles=["data"]))
        for src in derived_from:
            item.add_link(pystac.Link("derived_from", src.get_self_href() or src.id, title=src.id))
        self.add(collection, item)
        return item

    def remove(self, collection: str, item_id: str) -> None:
        if self._index[collection].pop(item_id, None) is not None:
            self._cols[collection].remove_item(item_id)

    def save(self) -> None:
        for col in self._cols.values():
            items = list(self._index[col.id].values())
            if items:
                col.extent = pystac.Extent.from_items(items)
        self.cat.normalize_hrefs(str(self.root))
        self.cat.make_all_asset_hrefs_relative()
        self.cat.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)


def _start(item: pystac.Item) -> datetime:
    return datetime.fromisoformat(item.properties["start_datetime"].replace("Z", "+00:00"))


def _bbox_geom(b: list[float]) -> dict:
    x0, y0, x1, y1 = b
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}
