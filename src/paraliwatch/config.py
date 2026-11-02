"""Run configuration and the assignment's configuration-code rule."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

REGIONS = {
    1: "Punjab — Ludhiana–Patiala belt",
    2: "Sundarbans, West Bengal",
    3: "Marathwada, Maharashtra",
    4: "Kutch, Gujarat",
    5: "Brahmaputra floodplain, Assam",
    6: "Delhi NCR periphery",
    7: "Godavari delta, Andhra Pradesh",
    8: "Western Ghats, Kerala/Karnataka",
}
SENSORS = {
    1: "Sentinel-2",
    2: "Sentinel-1",
    3: "Landsat 8/9",
    4: "MODIS",
    5: "ResourceSat LISS-III / AWiFS",
    6: "Precipitation (IMERG / CHIRPS)",
}
DOMAINS = {
    1: "Vegetation / crop condition",
    2: "Surface water / flood extent",
    3: "Built-up / urban change",
    4: "Drought / soil-moisture proxy",
    5: "Burn scars / fire-affected area",
    6: "Land-surface temperature / snow",
}
TASKS = {
    1: "Change detection",
    2: "Now-casting",
    3: "Segmentation / classification",
    4: "Multi-sensor fusion",
    5: "Gap-filling",
    6: "Embedding + similarity search",
}


@dataclass(frozen=True)
class Codes:
    region: int
    sensor: int
    domain: int
    task: int

    def __str__(self) -> str:
        return f"R{self.region} S{self.sensor} D{self.domain} T{self.task}"

    def describe(self) -> str:
        return (
            f"R{self.region} {REGIONS[self.region]} · S{self.sensor} {SENSORS[self.sensor]} · "
            f"D{self.domain} {DOMAINS[self.domain]} · T{self.task} {TASKS[self.task]}"
        )


def derive_codes(reference_number: int) -> Codes:
    """Section 1 rule: n = last four digits; ÷ is integer division."""
    n = reference_number % 10_000
    return Codes(
        region=n % 8 + 1,
        sensor=(n // 10) % 6 + 1,
        domain=(n // 100) % 6 + 1,
        task=(n // 1000) % 6 + 1,
    )


@dataclass(frozen=True)
class Settings:
    raw: dict
    root: Path

    @classmethod
    def load(cls, path: str | Path) -> Settings:
        path = Path(path).resolve()
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(raw=raw, root=path.parent.parent)

    def __getitem__(self, key: str):
        return self.raw[key]

    @property
    def codes(self) -> Codes:
        return derive_codes(self.raw["reference_number"])

    @property
    def season(self) -> tuple[date, date]:
        s = self.raw["season"]
        return date.fromisoformat(s["start"]), date.fromisoformat(s["end"])

    def path(self, *parts: str) -> Path:
        p = self.root / self.raw["paths"]["data"] / Path(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def site_dir(self) -> Path:
        p = self.root / self.raw["paths"]["site"]
        p.mkdir(parents=True, exist_ok=True)
        return p
