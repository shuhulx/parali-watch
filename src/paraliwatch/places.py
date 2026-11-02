"""Village / town names from OpenStreetMap, so the dispatch list speaks place names.

A snapshot is committed at reference/places.geojson (© OpenStreetMap
contributors, ODbL); `paraliwatch places` refreshes it.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

OVERPASS = "https://overpass-api.de/api/interpreter"
SNAPSHOT = Path("reference/places.geojson")


def fetch(bbox: list[float], out: Path = SNAPSHOT, pad: float = 0.05) -> int:
    w, s, e, n = bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad
    q = f'[out:json][timeout:60];node["place"~"village|town|hamlet|city"]({s},{w},{n},{e});out body;'
    req = urllib.request.Request(
        OVERPASS, urllib.parse.urlencode({"data": q}).encode(), {"User-Agent": "paraliwatch/0.1"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        elements = json.load(r)["elements"]
    feats = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [el["lon"], el["lat"]]},
            "properties": {"name": el["tags"]["name"], "place": el["tags"].get("place")},
        }
        for el in elements
        if "name" in el.get("tags", {})
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"type": "FeatureCollection", "attribution": "© OpenStreetMap contributors, ODbL", "features": feats}
        )
    )
    return len(feats)


class Gazetteer:
    def __init__(self, path: Path = SNAPSHOT):
        feats = json.loads(Path(path).read_text())["features"] if Path(path).exists() else []
        self.names = [f["properties"]["name"] for f in feats]
        self.xy = np.array([f["geometry"]["coordinates"] for f in feats]) if feats else np.zeros((0, 2))

    def nearest(self, lon: float, lat: float, k: int = 2) -> list[str]:
        if not self.names:
            return []
        dx = (self.xy[:, 0] - lon) * np.cos(np.radians(lat))
        d = dx**2 + (self.xy[:, 1] - lat) ** 2
        return [self.names[i] for i in np.argsort(d)[:k]]
