from datetime import date

import numpy as np
import rasterio
from rasterio.transform import from_origin

from paraliwatch.fusion import FireIndex, Transitions, fold, indices, weak_labels


def _scene(tmp_path, name, b8a, b11, b12, valid=None):
    arr = np.stack([b8a, b11, b12]).astype(np.float32)
    dn = np.round(arr * 10_000).astype(np.uint16)
    if valid is not None:
        dn[:, ~valid] = 0
    p = tmp_path / f"{name}.tif"
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        height=dn.shape[1],
        width=dn.shape[2],
        count=3,
        dtype="uint16",
        crs="EPSG:32643",
        transform=from_origin(0, 0, 20, 20),
        nodata=0,
    ) as dst:
        dst.write(dn)
    return p


def test_nbr_of_char_is_negative_and_of_crop_positive():
    crop = indices(np.array([[0.40], [0.20], [0.10]]))["nbr"][0]
    char = indices(np.array([[0.10], [0.12], [0.14]]))["nbr"][0]
    assert crop > 0.5 and char < 0


def test_fold_emits_one_transition_per_change_and_skips_cloudy_dates(tmp_path):
    shape = (2, 2)
    ones = np.ones(shape)
    crop = _scene(tmp_path, "a", 0.40 * ones, 0.20 * ones, 0.10 * ones)
    cloudy = _scene(tmp_path, "b", 0.40 * ones, 0.20 * ones, 0.10 * ones, valid=np.zeros(shape, bool))
    burnt = 0.40 * ones
    burnt[0, 0] = 0.10  # one pixel burns
    after = _scene(tmp_path, "c", burnt, 0.20 * ones, np.where(burnt < 0.2, 0.14, 0.10))
    f = fold([(date(2024, 10, 1), crop), (date(2024, 10, 3), cloudy), (date(2024, 10, 6), after)], shape, 0.1, 0.06)
    t = f.transitions
    assert t.pix.tolist() == [0]
    # The cloudy date is skipped: the interval runs from the last *clear* observation.
    assert t.day_prev[0] == date(2024, 10, 1).toordinal()
    assert t.day[0] == date(2024, 10, 6).toordinal()
    assert f.n_obs.tolist() == [[2, 2], [2, 2]]


def _fire_index(fire_days, observed_all=True):
    days = np.array([date(2024, 10, 1).toordinal() + i for i in range(10)])
    fire = np.zeros((10, 5, 5), np.uint8)
    for d in fire_days:
        fire[d, 2, 2] = 8
    observed = np.ones((10, 5, 5), bool) if observed_all else np.zeros((10, 5, 5), bool)
    return days, fire, observed


def test_fire_index_counts_inclusive_ranges():
    days, fire, obs = _fire_index([3, 4])
    fi = FireIndex(days, fire, obs, min_class=7, buffer=0)
    cell = np.array([2 * 5 + 2])
    d0 = days[0]
    assert fi.own(cell, np.array([d0 + 3]), np.array([d0 + 3]))[0] == 1
    assert fi.own(cell, np.array([d0]), np.array([d0 + 9]))[0] == 2
    assert fi.own(cell, np.array([d0 + 5]), np.array([d0 + 9]))[0] == 0


def test_weak_labels_positive_negative_and_unlabelled():
    days, fire, obs = _fire_index([4])
    fi = FireIndex(days, fire, obs, min_class=7, buffer=1)
    d0 = int(days[0])
    t = Transitions(
        pix=np.zeros(3, int),
        day_prev=np.array([d0 + 2, d0, d0 + 2]),
        day=np.array([d0 + 5, d0 + 1, d0 + 5]),
        X=np.zeros((3, 1)),
    )
    cells = np.array([12, 0, 13])  # fire cell; far cell; neighbour of fire cell
    assert weak_labels(t, cells, fi, pad=1).tolist() == [1, 0, -1]


def test_no_fire_under_cloud_is_not_a_negative():
    days, fire, obs = _fire_index([], observed_all=False)
    fi = FireIndex(days, fire, obs, min_class=7, buffer=1)
    d0 = int(days[0])
    t = Transitions(pix=np.zeros(1, int), day_prev=np.array([d0]), day=np.array([d0 + 3]), X=np.zeros((1, 1)))
    assert weak_labels(t, np.array([0]), fi, pad=1).tolist() == [-1]


def test_last_near_day_dates_burn_inside_the_interval():
    days, fire, obs = _fire_index([2, 5])
    fi = FireIndex(days, fire, obs, min_class=7, buffer=0)
    d0 = int(days[0])
    cell = np.array([12, 12, 0])
    got = fi.last_near_day(cell, np.array([d0 + 1, d0 + 1, d0]), np.array([d0 + 4, d0 + 8, d0 + 9]))
    assert got.tolist() == [d0 + 2, d0 + 5, -1]


def test_active_sample_takes_uncertain_points_one_per_cell_outside_sealed(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pandas as pd

    from paraliwatch import fusion

    monkeypatch.chdir(tmp_path)
    (tmp_path / "reference").mkdir()
    # 4x4 fine grid, 2x2 coarse cells; the test point at (0, 0) seals coarse cell 0.
    pd.DataFrame({"row": [0], "col": [0]}).to_csv("reference/sample.csv", index=False)
    pd.DataFrame({"point_id": ["T000"], "bin": [0], "row": [3], "col": [3], "pix": [15], "pre": ["2024-10-01"],
                  "post": ["2024-10-05"], "round": [1]}).to_csv("reference/train_sample.csv", index=False)  # fmt: skip
    grid = SimpleNamespace(factor=2, shape=(4, 4), coarse=SimpleNamespace(shape=(2, 2)))
    pix = np.array([0, 2, 3, 8, 9, 15])
    cells = np.array([0, 1, 1, 2, 2, 3])
    p = np.array([0.5, 0.49, 0.45, 0.9, 0.3, 0.5])
    t = Transitions(pix, np.full(6, 739160), np.full(6, 739164), np.zeros((6, 1), np.float32))
    new = fusion.active_sample(t, cells, p, 0.5, grid, n_uncertain=2, n_audit=0)
    # cell 0 sealed, pixel 15 already labelled; best per cell: cell 1 -> pix 2 (0.49), cell 2 -> pix 9 (0.3) vs 8 (0.9)
    assert sorted(new["pix"]) == [2, 9] and set(new["round"]) == {3}
    assert len(pd.read_csv("reference/train_sample.csv")) == 3


def test_student_never_scores_actively_sampled_rows():
    from paraliwatch.fusion import _student

    rng = np.random.default_rng(0)
    n, m = 200, 100
    y = np.r_[np.tile([0, 1], n // 2), rng.integers(0, 2, m)]  # last m rows: noisy "active-learning" labels
    X = np.c_[y[:, None] * 2.0 + rng.normal(0, 0.3, (n + m, 3)), np.zeros(n + m)]
    X[n:, :3] = rng.normal(1, 0.3, (m, 3))  # ambiguous features, random labels
    scored = np.r_[np.ones(n, bool), np.zeros(m, bool)]
    w = np.where(scored, 1.0, 0.0)
    blocks = np.arange(n + m) % 10
    teacher = lambda held, extra: (np.zeros(n + m), None)  # noqa: E731
    _, thr, d = _student(X, y, blocks, w, teacher, scored=scored)
    assert d["n_scored"] == n and d["n_labelled"] == n + m
    assert d["student_oof_auc"] > 0.95  # would be far lower if the random-label rows were scored
