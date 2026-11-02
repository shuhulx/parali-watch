from paraliwatch.config import derive_codes


def test_reference_123456_gives_r1_s4_d5_t4():
    c = derive_codes(123456)
    assert (c.region, c.sensor, c.domain, c.task) == (1, 4, 5, 4)
    assert str(c) == "R1 S4 D5 T4"


def test_only_last_four_digits_matter():
    assert derive_codes(123456) == derive_codes(3456) == derive_codes(99993456)


def test_codes_stay_in_table_ranges():
    for n in range(10_000):
        c = derive_codes(n)
        assert 1 <= c.region <= 8 and 1 <= c.sensor <= 6 and 1 <= c.domain <= 6 and 1 <= c.task <= 6
