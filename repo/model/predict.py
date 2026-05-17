import os
import sys
import argparse
import logging
import random
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from utils.tokenizer import (
    DateTokenizer, is_leap_year, days_in_month,
    weekday_of, DAY_NAMES, MONTH_NAMES,
    YEAR_MIN, YEAR_MAX
)
from utils.validators import validate_all
from utils.config import load_config, get_model_config
from models import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

logger = logging.getLogger("predict")


def post_process(day, month, year,
                 day_str, month_str,
                 leap_str, decade_str):

    expected_leap = leap_str.strip().lower() == "true"
    expected_decade = int(decade_str.strip())
    expected_month = MONTH_NAMES.index(month_str) + 1
    expected_wday = DAY_NAMES.index(day_str)

    year_start = expected_decade * 10
    year_end = min(year_start + 9, YEAR_MAX)
    year_start = max(year_start, YEAR_MIN)

    valid_years = [
        y for y in range(year_start, year_end + 1)
        if is_leap_year(y) == expected_leap
    ]

    if not valid_years:
        valid_years = list(range(year_start, year_end + 1))

    if not valid_years:
        valid_years = [year_start]

    chosen_year = year if year in valid_years else random.choice(valid_years)

    max_d = days_in_month(expected_month, chosen_year)
    matching_days = [
        d for d in range(1, max_d + 1)
        if weekday_of(d, expected_month, chosen_year) == expected_wday
    ]

    if matching_days:
        chosen_day = min(matching_days, key=lambda d: abs(d - day))
    else:
        chosen_day = min(max(day, 1), max_d)

    return chosen_day, expected_month, chosen_year


def predict_one(model, tokenizer,
                day_str, month_str,
                leap_str, decade_str):

    cond_vec = tokenizer.encode_condition(day_str, month_str, leap_str, decade_str)
    raw = model.generate(cond_vec)
    day, month, year = tokenizer.decode_date(raw)

    ok, _ = validate_all(day, month, year,
                         day_str, month_str,
                         leap_str, decade_str)

    if not ok:
        day, month, year = post_process(
            day, month, year,
            day_str, month_str,
            leap_str, decade_str
        )

    return tokenizer.date_to_string(day, month, year)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--model", default="gan",
                   choices=["gan", "vae", "transformer", "diffusion"])
    p.add_argument("--weights", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    weights_dir = args.weights or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "weights")
    )

    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(
            f"Weights directory not found: '{weights_dir}'. "
            f"Train the model first with train.py."
        )

    config = load_config(args.config)
    model_cfg = get_model_config(config, args.model)
    model_cfg["condition_dim"] = DateTokenizer.CONDITION_DIM

    logger.info(f"Loading model: {args.model} from '{weights_dir}'")

    model = get_model(args.model, model_cfg)
    model.load(weights_dir)

    tokenizer = DateTokenizer()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: '{args.input}'")

    with open(args.input, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    output_lines = []

    for i, line in enumerate(lines, 1):
        try:
            day_str, month_str, leap_str, decade_str = \
                DateTokenizer.parse_condition_line(line)

            date_str = predict_one(
                model, tokenizer,
                day_str, month_str,
                leap_str, decade_str
            )

            out = f"{line} {date_str}"
            logger.info(f"[{i}/{len(lines)}] {out}")

        except Exception as e:
            out = f"{line} ERROR"
            logger.error(f"[{i}] {e}")

        output_lines.append(out)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, "w") as f:
        f.write("\n".join(output_lines) + "\n")

    logger.info(f"Output written to '{args.output}'")


if __name__ == "__main__":
    main()