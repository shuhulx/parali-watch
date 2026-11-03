import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from paraliwatch import evaluate
from paraliwatch.config import Settings
from paraliwatch.evaluate import _estimate, kappa


def test_kappa_perfect_and_chance():
    a = np.array([0, 1, 0, 1])
    assert kappa(a, a) == 1.0
    assert abs(kappa(np.array([0, 1, 0, 1]), np.array([0, 0, 1, 1]))) < 1e-9


def test_stratum_weights_undo_oversampling():
    # Stratum 1 (mapped burn) is 1% of the population but half the sample.
    rows = []
    for i in range(50):
        rows.append(dict(interval=0, stratum=1, N_h=100, m=True, ref=i < 40))
    for i in range(50):
        rows.append(dict(interval=0, stratum=3, N_h=9900, m=False, ref=i < 1))
    est = _estimate(pd.DataFrame(rows), "m", n_boot=50)
    # precision is a within-stratum-1 quantity: 40/50
    assert est["precision"][0] == 0.8
    # recall is dominated by the big stratum's omissions: 0.8*100 / (0.8*100 + 0.02*9900)
    assert abs(est["recall"][0] - 80 / (80 + 198)) < 1e-3


@pytest.mark.parametrize("has_v1", [False, True])
def test_score_does_not_report_missing_v1_as_zero_predictions(tmp_path, monkeypatch, has_v1):
    settings = Settings(raw={"paths": {"data": "data"}}, root=tmp_path)
    if has_v1:
        settings.path("products", "burn_season_v1.tif").touch()
    ref = tmp_path / "reference"
    ref.mkdir()
    pd.DataFrame({
        "point_id": ["P0", "P1"], "pre": ["2024-10-01"] * 2, "post": ["2024-10-11"] * 2,
        "interval": [0, 0], "stratum": [1, 1], "N_h": [100, 100], "row": [0, 0], "col": [0, 1],
    }).to_csv(ref / "sample.csv", index=False)
    (ref / "sample_meta.json").write_text(json.dumps({"grid": "test-grid"}))
    for name in ("A", "B"):
        pd.DataFrame({"point_id": ["P0", "P1"], "label": [1, 0]}).to_csv(ref / f"labels_{name}.csv", index=False)
    monkeypatch.setattr(evaluate, "_layers", lambda *args: {m: np.array([[True, False]]) for m in evaluate.METHODS})
    monkeypatch.setattr(evaluate, "_estimate", lambda *args, **kwargs: {"f1": [1, 1, 1]})

    result = evaluate.score(settings, SimpleNamespace(signature="test-grid"), ref)

    assert result["unavailable_methods"] == ([] if has_v1 else ["fusion_v1"])
    assert ("fusion_v1" in result["pooled"]) == has_v1
    assert ("fusion_v1" in result["by_interval"]["0"]) == has_v1
    assert "fusion" in result["pooled"]
