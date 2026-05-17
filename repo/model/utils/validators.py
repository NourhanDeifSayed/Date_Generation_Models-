import logging
from typing import Tuple
from utils.tokenizer import (
    is_leap_year, days_in_month, weekday_of,
    YEAR_MIN, YEAR_MAX,
    DAY_NAMES, MONTH_NAMES
)

logger = logging.getLogger(__name__)


def validate_date_range(day: int, month: int, year: int) -> Tuple[bool, str]:
    if not (YEAR_MIN <= year <= YEAR_MAX):
        return False, "year out of range"
    if not (1 <= month <= 12):
        return False, "month out of range"
    if not (1 <= day <= days_in_month(month, year)):
        return False, "invalid day"
    return True, "ok"


def validate_leap_condition(year: int, expected_leap: bool) -> Tuple[bool, str]:
    if is_leap_year(year) != expected_leap:
        return False, "leap mismatch"
    return True, "ok"


def validate_weekday_condition(day: int, month: int, year: int,
                               expected_day_str: str) -> Tuple[bool, str]:
    if expected_day_str not in DAY_NAMES:
        return False, "unknown weekday"

    expected_idx = DAY_NAMES.index(expected_day_str)
    actual_idx = weekday_of(day, month, year)

    if actual_idx != expected_idx:
        return False, "weekday mismatch"

    return True, "ok"


def validate_month_condition(month: int, expected_month_str: str) -> Tuple[bool, str]:
    if expected_month_str not in MONTH_NAMES:
        return False, "unknown month"

    if MONTH_NAMES.index(expected_month_str) + 1 != month:
        return False, "month mismatch"

    return True, "ok"


def validate_decade_condition(year: int, expected_decade: int) -> Tuple[bool, str]:
    if year // 10 != expected_decade:
        return False, "decade mismatch"
    return True, "ok"


def validate_all(day: int, month: int, year: int,
                 day_str: str, month_str: str,
                 leap_str: str, decade_str: str) -> Tuple[bool, str]:

    expected_leap = leap_str.strip().lower() == "true"
    expected_decade = int(decade_str.strip())

    checks = [
        validate_date_range(day, month, year),
        validate_month_condition(month, month_str),
        validate_leap_condition(year, expected_leap),
        validate_decade_condition(year, expected_decade),
        validate_weekday_condition(day, month, year, day_str),
    ]

    for ok, msg in checks:
        if not ok:
            return False, msg

    return True, "ok"