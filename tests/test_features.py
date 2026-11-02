from datetime import date

from paraliwatch.features import week_key, week_start


def test_iso_weeks():
    assert week_key(date(2024, 11, 4)) == "2024-W45"
    assert week_start(date(2024, 11, 7)) == date(2024, 11, 4)
