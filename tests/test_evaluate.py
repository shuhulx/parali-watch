import numpy as np
import pandas as pd

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
