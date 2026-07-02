"""Small statistical helpers for paper-grade benchmark reporting."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean
from typing import Callable, Iterable


def bootstrap_ci(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    samples: int = 1000,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Return mean and percentile bootstrap CI for a list of scalar values."""
    xs = [float(v) for v in values]
    if not xs:
        return 0.0, 0.0, 0.0
    center = mean(xs)
    if len(xs) == 1:
        return round(center, 4), round(center, 4), round(center, 4)
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        draw = [xs[rng.randrange(len(xs))] for _ in xs]
        estimates.append(mean(draw))
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    lo = estimates[max(0, int(alpha * samples))]
    hi = estimates[min(samples - 1, int((1.0 - alpha) * samples))]
    return round(center, 4), round(lo, 4), round(hi, 4)


def paired_mcnemar(rows: list[dict], baseline: str, treatment: str, key: str) -> dict:
    """Approximate paired McNemar comparison for boolean correctness rows."""
    by_case: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        case_id = str(row.get("case_id") or row.get("scenario") or row.get("id"))
        system = str(row.get("system"))
        if system in {baseline, treatment}:
            by_case[case_id][system] = bool(row.get(key))

    b_wins = 0
    t_wins = 0
    paired = 0
    for systems in by_case.values():
        if baseline not in systems or treatment not in systems:
            continue
        paired += 1
        b_ok = systems[baseline]
        t_ok = systems[treatment]
        if b_ok and not t_ok:
            b_wins += 1
        elif t_ok and not b_ok:
            t_wins += 1

    discordant = b_wins + t_wins
    if discordant == 0:
        p_value = 1.0
        statistic = 0.0
    else:
        statistic = (abs(b_wins - t_wins) - 1) ** 2 / discordant
        # Survival function of chi-square(df=1): erfc(sqrt(x / 2)).
        p_value = math.erfc(math.sqrt(statistic / 2.0))
    return {
        "baseline": baseline,
        "treatment": treatment,
        "metric": key,
        "paired_cases": paired,
        "baseline_only_correct": b_wins,
        "treatment_only_correct": t_wins,
        "mcnemar_chi2": round(statistic, 4),
        "p_value_approx": round(p_value, 6),
    }


def aggregate_ci(rows: list[dict], metric_keys: list[str], group_key: str = "system") -> dict:
    """Aggregate boolean or numeric row metrics with bootstrap intervals."""
    out: dict[str, dict] = {}
    groups = sorted({str(row.get(group_key)) for row in rows})
    for group in groups:
        group_rows = [row for row in rows if str(row.get(group_key)) == group]
        entry = {"n": len(group_rows)}
        for key in metric_keys:
            values = []
            for row in group_rows:
                value = row.get(key)
                if isinstance(value, bool):
                    values.append(1.0 if value else 0.0)
                elif isinstance(value, (int, float)):
                    values.append(float(value))
            center, lo, hi = bootstrap_ci(values)
            entry[key] = center
            entry[f"{key}_ci95"] = [lo, hi]
        out[group] = entry
    return out


def accuracy_at_k(ranked: list[str], truth: str, k: int) -> bool:
    truth_n = normalize_id(truth)
    return any(normalize_id(item) == truth_n for item in ranked[:k])


def reciprocal_rank(ranked: list[str], truth: str) -> float:
    truth_n = normalize_id(truth)
    for idx, item in enumerate(ranked, start=1):
        if normalize_id(item) == truth_n:
            return 1.0 / idx
    return 0.0


def normalize_id(value: str | None) -> str:
    return (value or "").strip().lower().replace("_", "-")
