"""US federal holiday dates, including observed weekday dates."""

from datetime import date, timedelta


def _nth_weekday(year: int, month: int, weekday: int, occurrence: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + (occurrence - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def holidays(year: int) -> set[date]:
    actual = {
        date(year, 1, 1),
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _last_weekday(year, 5, 0),
        date(year, 6, 19),
        date(year, 7, 4),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 10, 0, 2),
        _nth_weekday(year, 11, 3, 4),
        date(year, 12, 25),
    }
    observed = set(actual)
    for holiday in actual:
        if holiday.weekday() == 5:
            observed.add(holiday - timedelta(days=1))
        elif holiday.weekday() == 6:
            observed.add(holiday + timedelta(days=1))
    return observed


def is_peak_travel(day: date) -> bool:
    return any(abs((day - holiday).days) <= 2 for holiday in holidays(day.year))
