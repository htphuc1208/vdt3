"""Small statistical helpers for paper-grade benchmark reporting."""
from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean
from typing import Iterable


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


def wilson_ci(successes: int, total: int, *, confidence: float = 0.95) -> tuple[float, float, float]:
    """Wilson score interval for binomial rates."""
    if total <= 0:
        return 0.0, 0.0, 0.0
    z = 1.959963984540054 if confidence == 0.95 else 1.959963984540054
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) / denom
    return round(phat, 4), round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def paired_mcnemar(rows: list[dict], baseline: str, treatment: str, key: str) -> dict:
    """Exact paired McNemar comparison for boolean correctness rows.

    Repeated LLM runs are first aggregated per scenario/case and system, so
    stochastic repeats do not masquerade as additional independent scenarios.
    """
    by_case: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        case_id = str(row.get("case_id") or row.get("scenario") or row.get("id"))
        system = str(row.get("system"))
        if system in {baseline, treatment}:
            by_case[case_id][system].append(bool(row.get(key)))

    b_wins = 0
    t_wins = 0
    paired = 0
    for systems in by_case.values():
        if baseline not in systems or treatment not in systems:
            continue
        paired += 1
        b_ok = _majority(systems[baseline])
        t_ok = _majority(systems[treatment])
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
        tail = sum(math.comb(discordant, i) for i in range(0, min(b_wins, t_wins) + 1))
        p_value = min(1.0, 2.0 * tail * (0.5 ** discordant))
    return {
        "baseline": baseline,
        "treatment": treatment,
        "metric": key,
        "paired_cases": paired,
        "baseline_only_correct": b_wins,
        "treatment_only_correct": t_wins,
        "mcnemar_chi2": round(statistic, 4),
        "p_value_exact": round(p_value, 6),
    }


def paired_bootstrap_effect(
    rows: list[dict],
    baseline: str,
    treatment: str,
    key: str,
    *,
    samples: int = 1000,
    seed: int = 7,
) -> dict:
    """Scenario-level paired bootstrap for treatment-baseline mean difference."""
    paired_values = []
    by_case: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        case_id = str(row.get("case_id") or row.get("scenario") or row.get("id"))
        system = str(row.get("system"))
        if system not in {baseline, treatment}:
            continue
        value = row.get(key)
        if isinstance(value, bool):
            by_case[case_id][system].append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            by_case[case_id][system].append(float(value))

    for systems in by_case.values():
        if baseline in systems and treatment in systems:
            paired_values.append(mean(systems[treatment]) - mean(systems[baseline]))

    center, lo, hi = bootstrap_ci(paired_values, samples=samples, seed=seed)
    return {
        "baseline": baseline,
        "treatment": treatment,
        "metric": key,
        "paired_cases": len(paired_values),
        "mean_difference": center,
        "mean_difference_ci95": [lo, hi],
    }


def aggregate_ci(rows: list[dict], metric_keys: list[str], group_key: str = "system") -> dict:
    """Aggregate boolean or numeric row metrics with appropriate intervals."""
    out: dict[str, dict] = {}
    groups = sorted({str(row.get(group_key)) for row in rows})
    for group in groups:
        group_rows = [row for row in rows if str(row.get(group_key)) == group]
        entry = {"n": len(group_rows)}
        for key in metric_keys:
            values = []
            bool_values = []
            for row in group_rows:
                value = row.get(key)
                if isinstance(value, bool):
                    values.append(1.0 if value else 0.0)
                    bool_values.append(value)
                elif isinstance(value, (int, float)):
                    values.append(float(value))
            if len(bool_values) == len(values) and values:
                center, lo, hi = wilson_ci(sum(1 for value in bool_values if value), len(bool_values))
            else:
                center, lo, hi = bootstrap_ci(values)
            entry[key] = center
            entry[f"{key}_ci95"] = [lo, hi]
        out[group] = entry
    return out


def _majority(values: list[bool]) -> bool:
    return sum(1 for value in values if value) >= (len(values) / 2.0)


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
