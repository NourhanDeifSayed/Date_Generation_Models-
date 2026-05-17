import numpy as np
import logging
from typing import Tuple, List
from datetime import date

logger = logging.getLogger(__name__)

DAY_NAMES = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MONTH_NAMES = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

YEAR_MIN, YEAR_MAX = 1800, 2200
DECADE_MIN, DECADE_MAX = 180, 220


def is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def days_in_month(month: int, year: int) -> int:
    if month == 2:
        return 29 if is_leap_year(year) else 28
    return DAYS_IN_MONTH[month - 1]


def weekday_of(day: int, month: int, year: int) -> int:
    return date(year, month, day).weekday()


class DateTokenizer:
    CONDITION_DIM = 21
    DATE_DIM = 3

    def __init__(self) -> None:
        self.day_to_idx = {d: i for i, d in enumerate(DAY_NAMES)}
        self.month_to_idx = {m: i for i, m in enumerate(MONTH_NAMES)}

    def encode_condition(self, day_str: str, month_str: str,
                         leap_str: str, decade_str: str) -> np.ndarray:

        vec = np.zeros(self.CONDITION_DIM, dtype=np.float32)

        vec[self.day_to_idx[day_str]] = 1.0
        vec[7 + self.month_to_idx[month_str]] = 1.0

        if leap_str.strip().lower() == "true":
            vec[19] = 1.0

        decade = int(decade_str.strip())
        vec[20] = (decade - DECADE_MIN) / (DECADE_MAX - DECADE_MIN)

        return vec

    def decode_condition(self, vec: np.ndarray):
        day_idx = int(np.argmax(vec[:7]))
        month_idx = int(np.argmax(vec[7:19]))
        leap = bool(vec[19] > 0.5)
        decade = int(round(vec[20] * (DECADE_MAX - DECADE_MIN) + DECADE_MIN))
        return DAY_NAMES[day_idx], MONTH_NAMES[month_idx], leap, decade

    def encode_date(self, day: int, month: int, year: int) -> np.ndarray:
        return np.array([
            (day - 1) / 30.0,
            (month - 1) / 11.0,
            (year - YEAR_MIN) / float(YEAR_MAX - YEAR_MIN),
        ], dtype=np.float32)

    def decode_date(self, vec: np.ndarray):
        day = int(round(np.clip(vec[0], 0, 1) * 30 + 1))
        month = int(round(np.clip(vec[1], 0, 1) * 11 + 1))
        year = int(round(np.clip(vec[2], 0, 1) * (YEAR_MAX - YEAR_MIN) + YEAR_MIN))

        day = int(np.clip(day, 1, 31))
        month = int(np.clip(month, 1, 12))
        year = int(np.clip(year, YEAR_MIN, YEAR_MAX))

        max_day = days_in_month(month, year)
        day = min(day, max_day)

        return day, month, year

    def date_to_string(self, day: int, month: int, year: int) -> str:
        return f"{day}-{month:02d}-{year}"

    @staticmethod
    def parse_condition_line(line: str):
        import re
        tokens = re.findall(r'\[([^\]]+)\]', line.strip())
        return tokens[0], tokens[1], tokens[2], tokens[3]

    def encode_date_tokens(self, day: int, month: int, year: int):
        return [day - 1, 30 + (month - 1)]

    def decode_date_tokens(self, tokens, year: int):
        day = tokens[0] + 1
        month = tokens[1] - 30 + 1
        return int(np.clip(day, 1, 31)), int(np.clip(month, 1, 12)), year