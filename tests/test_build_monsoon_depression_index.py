import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_monsoon_depression_index import (  # noqa: E402
    MIN_DEPRESSION_IMD_CATEGORY,
    build_daily_presence,
    load_india_season_positions,
)


def _synthetic_catalogue() -> pd.DataFrame:
    rows = [
        # In-domain, in-season, category 1 ("low") -- must NOT count. 12:00 UTC
        # on 06-05 is inside the window 06-05T03:00 -> 06-06T03:00, which is
        # IMD day 06-06 (IMD labels a day by its end).
        {"time": pd.Timestamp("2020-06-05 12:00"), "year": 2020, "month": 6, "day": 5,
         "lat": 15.0, "lon": 75.0, "imd_category": 1, "imd_label": "low"},
        # In-domain, in-season, category 2 ("depression") -- must count. Like the
        # 12:00 row below, it falls in IMD day 07-11.
        {"time": pd.Timestamp("2020-07-10 06:00"), "year": 2020, "month": 7, "day": 10,
         "lat": 20.0, "lon": 80.0, "imd_category": 2, "imd_label": "depression"},
        # Same real day, later hour, stronger category -- max should win.
        {"time": pd.Timestamp("2020-07-10 12:00"), "year": 2020, "month": 7, "day": 10,
         "lat": 20.5, "lon": 80.5, "imd_category": 3, "imd_label": "deep_depression"},
        # Out of the India domain (lat too far south) -- must be excluded.
        {"time": pd.Timestamp("2020-08-01 00:00"), "year": 2020, "month": 8, "day": 1,
         "lat": -5.0, "lon": 80.0, "imd_category": 4, "imd_label": "cyclonic_storm"},
        # Out of season (October) -- must be excluded.
        {"time": pd.Timestamp("2020-10-01 00:00"), "year": 2020, "month": 10, "day": 1,
         "lat": 20.0, "lon": 80.0, "imd_category": 3, "imd_label": "deep_depression"},
        # Wrong year -- must be excluded.
        {"time": pd.Timestamp("2019-07-10 00:00"), "year": 2019, "month": 7, "day": 10,
         "lat": 20.0, "lon": 80.0, "imd_category": 3, "imd_label": "deep_depression"},
    ]
    return pd.DataFrame(rows)


class TestLoadIndiaSeasonPositions:
    def test_filters_to_domain_and_season(self, tmp_path):
        frame = _synthetic_catalogue()
        catalogue_path = tmp_path / "catalogue.parquet"
        frame.to_parquet(catalogue_path)

        result = load_india_season_positions(catalogue_path, year=2020)

        # Every in-domain, in-season row survives here regardless of
        # category (category filtering is build_daily_presence's job, not
        # this function's) -- only the out-of-domain, out-of-season, and
        # wrong-year rows are dropped.
        assert len(result) == 3
        assert set(result["imd_label"]) == {"low", "depression", "deep_depression"}


class TestBuildDailyPresence:
    def test_low_category_alone_does_not_count_as_depression(self):
        positions = _synthetic_catalogue().iloc[[0]]  # the category-1 "low" row only

        result = build_daily_presence(positions, year=2020)

        assert not bool(result["depression_present"].sel(time="2020-06-06").item())

    def test_depression_or_stronger_marks_the_real_day_present(self):
        positions = _synthetic_catalogue().iloc[[1, 2]]  # the two 2020-07-10 rows

        result = build_daily_presence(positions, year=2020)

        # 06:00 and 12:00 UTC on 07-10 are both after that day's 03 UTC
        # boundary, so they belong to IMD day 07-11, and 07-10 has nothing.
        assert bool(result["depression_present"].sel(time="2020-07-11").item())
        assert result["max_imd_category"].sel(time="2020-07-11").item() == 3
        assert not bool(result["depression_present"].sel(time="2020-07-10").item())

    def test_every_jjas_day_is_present_even_with_zero_real_rows(self):
        positions = _synthetic_catalogue().iloc[0:0]  # empty

        result = build_daily_presence(positions, year=2020)

        assert result.sizes["time"] == 122  # 30+31+31+30
        assert not bool(result["depression_present"].any())

    def test_min_depression_category_constant_is_depression_not_low(self):
        assert MIN_DEPRESSION_IMD_CATEGORY == 2
