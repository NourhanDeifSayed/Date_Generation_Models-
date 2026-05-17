
import numpy as np
import logging
import os
from typing import List, Tuple, Dict
from utils.tokenizer import DateTokenizer
from utils.validators import validate_all

logger = logging.getLogger(__name__)


def validity_rate(generated: List[Tuple[int, int, int]],
                  conditions: List[Tuple[str, str, str, str]]) -> float:
    valid = 0
    for (d, m, y), (ds, ms, ls, decs) in zip(generated, conditions):
        ok, _ = validate_all(d, m, y, ds, ms, ls, decs)
        if ok:
            valid += 1
    rate = valid / max(len(generated), 1)
    logger.debug(f"Validity rate: {rate:.4f}")
    return rate


def condition_match_rates(generated: List[Tuple[int, int, int]],
                          conditions: List[Tuple[str, str, str, str]]
                          ) -> Dict[str, float]:
    from utils.tokenizer import is_leap_year, weekday_of, DAY_NAMES, MONTH_NAMES
    from utils.validators import validate_weekday_condition, validate_month_condition
    counters = {"weekday": 0, "month": 0, "leap": 0, "decade": 0}
    N = max(len(generated), 1)
    for (d, m, y), (ds, ms, ls, decs) in zip(generated, conditions):
        expected_leap = ls.strip().lower() == "true"
        expected_decade = int(decs.strip())
        ok_w, _ = validate_weekday_condition(d, m, y, ds)
        ok_m, _ = validate_month_condition(m, ms)
        ok_l = is_leap_year(y) == expected_leap
        ok_d = (y // 10) == expected_decade
        if ok_w:
            counters["weekday"] += 1
        if ok_m:
            counters["month"] += 1
        if ok_l:
            counters["leap"] += 1
        if ok_d:
            counters["decade"] += 1
    rates = {k: v / N for k, v in counters.items()}
    logger.debug(f"Condition match rates: {rates}")
    return rates


def diversity_score(generated: List[Tuple[int, int, int]]) -> float:
    if not generated:
        return 0.0
    unique = len(set(generated))
    score = unique / len(generated)
    logger.debug(f"Diversity score: {score:.4f}")
    return score


def compute_all_metrics(generated: List[Tuple[int, int, int]],
                        conditions: List[Tuple[str, str, str, str]]
                        ) -> Dict[str, float]:
    vr = validity_rate(generated, conditions)
    cmr = condition_match_rates(generated, conditions)
    divs = diversity_score(generated)
    results = {
        "validity_rate": vr,
        "weekday_match": cmr["weekday"],
        "month_match": cmr["month"],
        "leap_match": cmr["leap"],
        "decade_match": cmr["decade"],
        "diversity_score": divs,
    }
    logger.info("Metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in results.items()))
    return results