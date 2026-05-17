import numpy as np
import logging
import re
from typing import Tuple, List, Dict
from utils.tokenizer import DateTokenizer, DAY_NAMES

logger = logging.getLogger(__name__)


def parse_data_file(path: str) -> List[Dict]:
    records = []
    tokenizer = DateTokenizer()

    with open(path, "r") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue

            tokens = re.findall(r'\[([^\]]+)\]', line)
            date_part = line.split(']')[-1].strip()

            if len(tokens) < 4 or not date_part:
                continue

            try:
                d_str, m_str, l_str, dec_str = tokens[0], tokens[1], tokens[2], tokens[3]
                parts = date_part.split('-')
                day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
            except (ValueError, IndexError):
                continue

            records.append({
                "day_str": d_str,
                "month_str": m_str,
                "leap_str": l_str,
                "decade_str": dec_str,
                "day": day,
                "month": month,
                "year": year,
            })

    return records


def records_to_arrays(records: List[Dict],
                      tokenizer: DateTokenizer) -> Tuple[np.ndarray, np.ndarray]:
    conditions, dates = [], []

    for r in records:
        try:
            cond = tokenizer.encode_condition(
                r["day_str"], r["month_str"], r["leap_str"], r["decade_str"]
            )
            dt = tokenizer.encode_date(r["day"], r["month"], r["year"])
        except ValueError:
            continue

        conditions.append(cond)
        dates.append(dt)

    return (
        np.array(conditions, dtype=np.float32),
        np.array(dates, dtype=np.float32),
    )


def train_val_test_split(conditions: np.ndarray,
                         dates: np.ndarray,
                         train: float = 0.8,
                         val: float = 0.1,
                         seed: int = 42):

    rng = np.random.default_rng(seed)
    N = len(conditions)
    idx = rng.permutation(N)

    n_train = int(N * train)
    n_val = int(N * val)

    tr = idx[:n_train]
    va = idx[n_train:n_train + n_val]
    te = idx[n_train + n_val:]

    return (
        (conditions[tr], dates[tr]),
        (conditions[va], dates[va]),
        (conditions[te], dates[te]),
    )


def compute_sample_weights(records: List[Dict]) -> np.ndarray:
    counts = {d: 0 for d in DAY_NAMES}

    for r in records:
        if r["day_str"] in counts:
            counts[r["day_str"]] += 1

    N = len(records)
    weights = []

    for r in records:
        c = counts.get(r["day_str"], 1)
        weights.append(N / (7.0 * max(c, 1)))

    return np.array(weights, dtype=np.float32)


def make_batches(conditions: np.ndarray,
                 dates: np.ndarray,
                 batch_size: int = 64,
                 shuffle: bool = True,
                 seed: int = 42):

    N = len(conditions)
    idx = np.arange(N)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    batches = []

    for start in range(0, N, batch_size):
        b = idx[start:start + batch_size]
        batches.append((conditions[b], dates[b]))

    return batches


def load_dataset(data_path: str,
                 train_split: float = 0.8,
                 val_split: float = 0.1,
                 batch_size: int = 64,
                 seed: int = 42):

    tokenizer = DateTokenizer()
    records = parse_data_file(data_path)

    if len(records) == 0:
        raise RuntimeError("No valid records found")

    conditions, dates = records_to_arrays(records, tokenizer)

    (c_tr, d_tr), (c_va, d_va), (c_te, d_te) = train_val_test_split(
        conditions, dates,
        train=train_split,
        val=val_split,
        seed=seed
    )

    train_batches = make_batches(c_tr, d_tr, batch_size, True, seed)
    val_batches = make_batches(c_va, d_va, batch_size, False, seed)
    test_batches = make_batches(c_te, d_te, batch_size, False, seed)

    return train_batches, val_batches, test_batches, tokenizer