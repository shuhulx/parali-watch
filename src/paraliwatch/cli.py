"""`paraliwatch` command line. Every stage is idempotent; `run` chains them."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from .config import Settings

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False)
DEFAULT_CONFIG = Path("config/punjab_2024.yaml")
Cfg = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Run configuration YAML.")
SampleDir = typer.Argument(Path("reference/testB"), help="Directory holding a sample.csv.")


def _ctx(config: Path):
    from .catalog import LocalCatalog
    from .grid import Grid

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "urllib3", "rasterio", "botocore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    s = Settings.load(config)
    return s, LocalCatalog(s.path("catalog", "catalog.json").parent), Grid(s)


@app.command()
def codes(reference_number: int = typer.Argument(123456)):
    """Derive the assignment configuration codes from a reference number."""
    from .config import derive_codes

    c = derive_codes(reference_number)
    typer.echo(f"{c}\n{c.describe()}")


@app.command()
def ingest(config: Path = Cfg):
    """Discover new scenes (STAC) and register them in the local catalogue."""
    from . import ingest as m

    s, cat, _ = _ctx(config)
    typer.echo(json.dumps(m.discover(s, cat)))
    cat.save()


@app.command()
def ard(config: Path = Cfg, workers: int = 4):
    """Build analysis-ready data for every catalogued scene that lacks it."""
    from . import ard as m

    s, cat, g = _ctx(config)
    typer.echo(json.dumps(m.run(s, cat, g, workers=workers)))


@app.command()
def fuse(config: Path = Cfg):
    """Fit the MODIS-supervised Sentinel-2 burn model and map burns."""
    from . import fusion

    s, cat, g = _ctx(config)
    typer.echo(json.dumps(fusion.run(s, cat, g), indent=2))


@app.command()
def features(config: Path = Cfg):
    """Aggregate products into the H3 x week GeoParquet feature store."""
    from . import features as m

    s, cat, g = _ctx(config)
    typer.echo(json.dumps(m.run(s, cat, g)))


@app.command()
def evaluate(config: Path = Cfg):
    """Score the map and baselines against the photo-interpreted reference set."""
    from . import evaluate as m

    s, _, g = _ctx(config)
    typer.echo(json.dumps(m.run(s, g), indent=2))


@app.command("label-page")
def label_page(sample_dir: Path = SampleDir, config: Path = Cfg):
    """Write a self-contained click-to-label page (keys 0/1/9, CSV export) for a sample set."""
    from . import evaluate as m

    s, _, g = _ctx(config)
    typer.echo(str(m.render_label_page(s, g, sample_dir)))


@app.command()
def asset(config: Path = Cfg):
    """Render the static now-cast dashboard into site/."""
    from . import asset as m

    s, cat, g = _ctx(config)
    typer.echo(str(m.build(s, cat, g)))


@app.command()
def run(config: Path = Cfg, workers: int = 4):
    """Full refresh: ingest -> ARD -> fuse -> features -> evaluate -> asset."""
    ingest(config)
    ard(config, workers)
    fuse(config)
    features(config)
    evaluate(config)
    asset(config)


@app.command()
def places(config: Path = Cfg):
    """Refresh the OpenStreetMap place-name snapshot used by the dispatch list."""
    from . import places as m

    s = Settings.load(config)
    typer.echo(f"{m.fetch(s['aoi']['bbox'])} places")


@app.command()
def query(sql: str, config: Path = Cfg):
    """Run SQL over the feature store. Tables: hex_week, hex_season."""
    from .features import connect

    s = Settings.load(config)
    typer.echo(connect(s).sql(sql).df().to_string(index=False))


if __name__ == "__main__":
    app()
