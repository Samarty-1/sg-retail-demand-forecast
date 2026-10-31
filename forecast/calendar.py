"""Chinese New Year: the one big Singapore shopping event that moves between months."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

# First day of Chinese New Year.
CNY = {
    1990: date(1990, 1, 27), 1991: date(1991, 2, 15), 1992: date(1992, 2, 4), 1993: date(1993, 1, 23),
    1994: date(1994, 2, 10), 1995: date(1995, 1, 31), 1996: date(1996, 2, 19), 1997: date(1997, 2, 7),
    1998: date(1998, 1, 28), 1999: date(1999, 2, 16), 2000: date(2000, 2, 5), 2001: date(2001, 1, 24),
    2002: date(2002, 2, 12), 2003: date(2003, 2, 1), 2004: date(2004, 1, 22), 2005: date(2005, 2, 9),
    2006: date(2006, 1, 29), 2007: date(2007, 2, 18), 2008: date(2008, 2, 7), 2009: date(2009, 1, 26),
    2010: date(2010, 2, 14), 2011: date(2011, 2, 3), 2012: date(2012, 1, 23), 2013: date(2013, 2, 10),
    2014: date(2014, 1, 31), 2015: date(2015, 2, 19), 2016: date(2016, 2, 8), 2017: date(2017, 1, 28),
    2018: date(2018, 2, 16), 2019: date(2019, 2, 5), 2020: date(2020, 1, 25), 2021: date(2021, 2, 12),
    2022: date(2022, 2, 1), 2023: date(2023, 1, 22), 2024: date(2024, 2, 10), 2025: date(2025, 1, 29),
    2026: date(2026, 2, 17), 2027: date(2027, 2, 6), 2028: date(2028, 1, 26),
}
RUN_UP_DAYS = 14  # the shopping rush before the holiday


def _share_in_month(start: date, days: int, year: int, month: int) -> float:
    inside = sum(1 for d in range(days)
                 if (start + timedelta(days=d)).year == year and (start + timedelta(days=d)).month == month)
    return inside / days


def cny_run_up_share(month: pd.Timestamp) -> float:
    """Share of the 14 days before Chinese New Year that fall in this month (0 to 1)."""
    year = month.year
    if year not in CNY:
        raise KeyError(f"no Chinese New Year date for {year}")
    start = CNY[year] - timedelta(days=RUN_UP_DAYS)
    return _share_in_month(start, RUN_UP_DAYS, year, month.month)


def cny_holiday_share(month: pd.Timestamp) -> float:
    """Share of the two public-holiday days that fall in this month (0, 0.5 or 1)."""
    year = month.year
    if year not in CNY:
        raise KeyError(f"no Chinese New Year date for {year}")
    return _share_in_month(CNY[year], 2, year, month.month)
