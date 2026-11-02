from datetime import datetime, timezone

import pystac

from paraliwatch.catalog import LocalCatalog, params_hash


def _item(i):
    return pystac.Item(
        i,
        {"type": "Point", "coordinates": [76.2, 30.5]},
        [76.2, 30.5, 76.2, 30.5],
        datetime(2024, 10, 1, tzinfo=timezone.utc),
        {},
    )


def test_catalogue_round_trip_is_idempotent(tmp_path):
    cat = LocalCatalog(tmp_path)
    assert cat.add("source-s2", _item("a")) is True
    assert cat.add("source-s2", _item("a")) is False
    cat.save()
    again = LocalCatalog(tmp_path)
    assert [i.id for i in again.items("source-s2")] == ["a"]


def test_derived_item_is_current_only_with_same_params_and_existing_file(tmp_path):
    cat = LocalCatalog(tmp_path / "cat")
    src = _item("scene")
    cat.add("source-s2", src)
    f = tmp_path / "x.tif"
    f.write_bytes(b"x")
    cat.add_derived("ard-s2", "ard-scene", src.datetime, [76, 30, 77, 31], {"r": (f, "image/tiff")}, [src], {"p": 1})
    cat.save()
    cat = LocalCatalog(tmp_path / "cat")
    assert cat.is_current("ard-s2", "ard-scene", params_hash({"p": 1}))
    assert not cat.is_current("ard-s2", "ard-scene", params_hash({"p": 2}))
    f.unlink()
    assert not cat.is_current("ard-s2", "ard-scene", params_hash({"p": 1}))
    assert [link.title for link in cat.get("ard-s2", "ard-scene").get_links("derived_from")] == ["scene"]
