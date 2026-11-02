"""Ingestion: discover remote scenes via STAC and register them in the local catalogue.

Only *metadata* is ingested here. Pixels are fetched lazily by the ARD stage,
and only for items that do not yet have a current ARD product — that is what
makes a mid-season refresh cost O(new scenes).
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

import planetary_computer
import pystac
import pystac_client

from .catalog import LocalCatalog
from .config import Settings

log = logging.getLogger(__name__)


def _client(settings: Settings) -> pystac_client.Client:
    return pystac_client.Client.open(settings["sources"]["stac_url"])


def _clean(item: pystac.Item) -> pystac.Item:
    """Drop remote links so the local catalogue stays self-contained."""
    item = item.clone()
    item.clear_links()
    return item


def discover(settings: Settings, cat: LocalCatalog) -> dict[str, int]:
    src = settings["sources"]
    start, end = settings.season
    bbox = settings["aoi"]["bbox"]
    client = _client(settings)
    added = {"source-s2": 0, "source-modis-fire": 0, "source-mcd64": 0}

    s2 = client.search(
        collections=[src["s2_collection"]],
        bbox=bbox,
        datetime=f"{start}/{end}",
        query={
            "s2:mgrs_tile": {"eq": src["s2_mgrs_tile"]},
            "eo:cloud_cover": {"lt": src["s2_max_scene_cloud"]},
        },
    )
    # The archive sometimes holds two processings of one acquisition: keep the latest.
    latest: dict[str, pystac.Item] = {}
    for it in s2.items():
        key = it.datetime.date().isoformat()
        rank = (it.properties.get("s2:processing_baseline", ""), it.id)
        if key not in latest or rank > (latest[key].properties.get("s2:processing_baseline", ""), latest[key].id):
            latest[key] = it
    existing = {i.datetime.date().isoformat(): i for i in cat.items("source-s2")}
    for key, it in latest.items():
        old = existing.get(key)
        if old is not None and old.id != it.id:  # reprocessed acquisition supersedes the old one
            cat.remove("source-s2", old.id)
            cat.remove("ard-s2", f"ard-{old.id}")
        if cat.get("source-s2", it.id) is None:
            added["source-s2"] += cat.add("source-s2", _clean(it))

    for coll, target in (
        (src["fire_collection"], "source-modis-fire"),
        (src["burned_area_collection"], "source-mcd64"),
    ):
        for it in client.search(collections=[coll], bbox=bbox, datetime=f"{start}/{end}").items():
            if cat.get(target, it.id) is None:
                added[target] += cat.add(target, _clean(it))

    log.info("discovered new items: %s", added)
    return added


def fetch(item: pystac.Item, asset: str, cache_dir: Path) -> Path:
    """Download one whole asset (used for small MODIS tiles) into a content cache."""
    out = cache_dir / f"{item.id}_{asset}.tif"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    href = planetary_computer.sign(item.assets[asset].href)
    tmp = out.with_suffix(".part")
    with urllib.request.urlopen(href, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(out)
    return out
