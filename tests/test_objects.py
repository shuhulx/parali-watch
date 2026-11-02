from __future__ import annotations

import numpy as np

from paraliwatch.objects import field_majority_day, field_vote, segment_fields


def test_segment_fields_rectangles_with_dark_bund():
    refl = np.full((6, 10, 21), 0.2, dtype=np.float32)
    refl[:, :, 10] = 0.01
    refl[:, :, 11:] = 0.6
    fields = segment_fields(refl, np.ones((10, 21), dtype=bool))
    left, right = fields[5, 5], fields[5, 15]
    assert fields.dtype == np.int32
    assert left > 0 and right > 0 and left != right
    assert np.all(fields[:, :10] == left)
    assert np.all(fields[:, 11:] == right)
    assert np.array_equal(np.unique(fields[fields > 0]), [1, 2])


def test_segment_fields_uses_all_infrared_bands_and_robust_scaling():
    for band in range(3):
        refl = np.ones((6, 10, 21), dtype=np.float32)
        refl[band, :, :10] = 2000
        refl[band, :, 10] = 100
        refl[band, :, 11:] = 6000
        fields = segment_fields(refl, np.ones((10, 21), dtype=bool))
        assert fields[5, 5] > 0
        assert fields[5, 15] > 0
        assert fields[5, 5] != fields[5, 15]


def test_segment_fields_empty_valid_mask():
    fields = segment_fields(np.full((6, 5, 5), np.nan, dtype=np.float32), np.zeros((5, 5), dtype=bool))
    assert fields.dtype == np.int32
    assert not np.any(fields)


def test_segment_fields_splits_large_components_on_grid():
    fields = segment_fields(np.ones((6, 41, 41), dtype=np.float32), np.ones((41, 41), dtype=bool))
    assert np.all(fields > 0)
    assert np.unique(fields).size == 9
    assert np.max(np.bincount(fields.ravel())) <= 400
    assert np.all(fields[:20, :20] == fields[0, 0])
    assert fields[0, 19] != fields[0, 20]
    assert fields[19, 0] != fields[20, 0]


def test_segment_fields_drops_small_components_and_preserves_invalid():
    valid = np.zeros((12, 12), dtype=bool)
    valid[1:4, 1:4] = True
    valid[7:9, 7:9] = True
    refl = np.full((6, 12, 12), np.nan, dtype=np.float32)
    refl[:, valid] = 0.3
    fields = segment_fields(refl, valid)
    assert np.all(fields[1:4, 1:4] > 0)
    assert not np.any(fields[7:9, 7:9])
    assert not np.any(fields[~valid])


def test_segment_fields_uses_four_connectivity():
    valid = np.eye(4, dtype=bool)
    fields = segment_fields(np.ones((6, 4, 4), dtype=np.float32), valid, min_px=1)
    assert np.unique(fields[valid]).size == 4
    assert not np.any(fields[~valid])


def test_field_vote_fills_sixty_percent_and_clears_twenty_percent():
    fields = np.repeat(np.array([[1, 2, 0]], dtype=np.int32), 10, axis=0)
    candidate = np.ones(fields.shape, dtype=bool)
    burn = np.zeros(fields.shape, dtype=bool)
    burn[:6, 0] = True
    burn[:2, 1] = True
    burn[::2, 2] = True
    voted = field_vote(burn, candidate, fields)
    assert voted.dtype == bool
    assert np.all(voted[:, 0])
    assert not np.any(voted[:, 1])
    np.testing.assert_array_equal(voted[:, 2], burn[:, 2])
    assert burn[:, 0].sum() == 6


def test_field_vote_candidate_denominator_and_minimum_count():
    fields = np.repeat(np.array([[1, 2, 3]], dtype=np.int32), 10, axis=0)
    candidate = np.zeros(fields.shape, dtype=bool)
    candidate[:6, 0] = True
    candidate[:2, 1] = True
    burn = np.zeros(fields.shape, dtype=bool)
    burn[:3, 0] = True
    burn[:2, 1] = True
    voted = field_vote(burn, candidate, fields)
    np.testing.assert_array_equal(voted[:, 0], candidate[:, 0])
    assert not np.any(voted[:, 1:])
    assert not np.any(field_vote(burn, candidate, fields, min_frac=0, min_px=0)[:, 2])


def test_field_majority_day_assigns_median_and_preserves_existing_days():
    fields = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [0, 0, 3, 3, 3]], dtype=np.int32)
    days = np.array([[10, 30, 20, 0, 0], [12, 15, 0, 0, 0], [9, 0, 0, 0, 0]], dtype=np.int16)
    voted = np.ones(fields.shape, dtype=bool)
    voted[0, 1] = False
    result = field_majority_day(days, voted, fields)
    expected = [[10, 0, 20, 20, 20], [12, 15, 13, 13, 13], [9, 0, 0, 0, 0]]
    assert result.dtype == np.int16
    np.testing.assert_array_equal(result, expected)
    assert days[0, 1] == 30


def test_field_majority_day_without_original_burns():
    days = np.zeros((2, 3), dtype=np.int16)
    result = field_majority_day(days, np.ones_like(days, dtype=bool), np.ones_like(days, dtype=np.int32))
    np.testing.assert_array_equal(result, days)
