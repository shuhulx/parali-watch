import numpy as np

from paraliwatch.ard import SHADOW_CASTER_M, s2_valid_mask


def test_dark_area_is_kept_because_burn_scars_get_that_label():
    scl = np.full((10, 10), 2)
    assert s2_valid_mask(scl, 20).all()


def test_shadow_without_a_cloud_is_kept():
    scl = np.full((50, 50), 4)
    scl[20:25, 20:25] = 3  # "shadow" with no caster: almost certainly a fresh burn scar
    assert s2_valid_mask(scl, 20)[22, 22]


def test_shadow_next_to_cloud_is_masked_and_cloud_is_masked():
    n = int(SHADOW_CASTER_M / 20) * 3
    scl = np.full((n, n), 4)
    scl[0:5, 0:5] = 9
    scl[6:8, 6:8] = 3
    v = s2_valid_mask(scl, 20)
    assert not v[2, 2] and not v[7, 7]
    assert v[-1, -1]


def test_nodata_and_snow_are_invalid():
    scl = np.array([[0, 1, 11, 4]])
    assert s2_valid_mask(scl, 20).tolist() == [[False, False, False, True]]


def test_smog_haze_is_masked_but_clear_field_is_kept(tmp_path):
    import rasterio
    from rasterio.transform import from_origin

    from paraliwatch.ard import read_s2_ard

    # bands: B8A B11 B12 B02 B03 B04 ; pixel 0 clear stubble, pixel 1 hazy (blue >> red)
    dn = np.array([[[3000, 3000]], [[3500, 3500]], [[2800, 2800]], [[900, 2200]], [[1100, 2000]], [[1300, 1600]]])
    p = tmp_path / "s.tif"
    with rasterio.open(p, "w", driver="GTiff", height=1, width=2, count=6, dtype="uint16", crs="EPSG:32643",
                       transform=from_origin(0, 0, 20, 20), nodata=0) as dst:  # fmt: skip
        dst.write(dn.astype("uint16"))
    _, valid = read_s2_ard(p)
    assert valid.tolist() == [[True, False]]
