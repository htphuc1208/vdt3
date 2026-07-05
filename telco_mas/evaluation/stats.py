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


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation).

    Accurate to ~1e-9 over (0, 1); used for a-priori power/sample-size planning,
    not for any inferential test (those stay exact).
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def mcnemar_required_pairs(
    delta: float,
    discordant_rate: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict:
    """Paired-binary (McNemar) sample size via Connor's (1987) normal approximation.

    ``delta`` is the marginal accuracy difference to detect (|p01 - p10|); the
    discordant proportion ``discordant_rate`` (= p01 + p10) is the nuisance
    parameter that dominates required n. Returns the number of *paired cases*.
    """
    delta = abs(float(delta))
    pd = float(discordant_rate)
    if delta <= 0.0 or pd <= 0.0 or pd < delta:
        return {"required_pairs": None, "reason": "require 0 < delta <= discordant_rate"}
    z_a = _norm_ppf(1 - alpha / 2.0)
    z_b = _norm_ppf(power)
    n = (z_a * math.sqrt(pd) + z_b * math.sqrt(pd - delta * delta)) ** 2 / (delta * delta)
    return {
        "required_pairs": int(math.ceil(n)),
        "delta": round(delta, 4),
        "discordant_rate": round(pd, 4),
        "alpha": alpha,
        "power": power,
        "method": "connor_1987_normal_approx",
    }


def _exact_mcnemar_reject(discordant: int, treatment_wins: int, alpha: float) -> bool:
    """Whether the two-sided exact McNemar test rejects at ``alpha`` for this split."""
    if discordant == 0:
        return False
    lo = min(treatment_wins, discordant - treatment_wins)
    tail = sum(math.comb(discordant, i) for i in range(0, lo + 1))
    p_value = min(1.0, 2.0 * tail * (0.5 ** discordant))
    return p_value <= alpha


def mcnemar_exact_power(
    n_pairs: int,
    p01: float,
    p10: float,
    *,
    alpha: float = 0.05,
) -> dict:
    """Exact power of the two-sided exact McNemar test.

    ``p01`` = P(treatment correct, baseline wrong), ``p10`` = P(baseline correct,
    treatment wrong). Enumerates the discordant count d ~ Binom(n, p01+p10) and,
    conditional on d, the treatment-win count ~ Binom(d, p01/(p01+p10)); sums the
    exact-test rejection mass. Matches the exact test actually used to report results.
    """
    pd = float(p01) + float(p10)
    if not 0.0 <= pd <= 1.0 or p01 < 0 or p10 < 0:
        return {"power": None, "reason": "invalid probabilities"}
    if pd == 0.0:
        return {"power": 0.0, "n_pairs": n_pairs, "p01": p01, "p10": p10, "alpha": alpha}
    p_t = float(p01) / pd
    power = 0.0
    for d in range(0, n_pairs + 1):
        p_d = math.comb(n_pairs, d) * (pd ** d) * ((1.0 - pd) ** (n_pairs - d))
        if p_d == 0.0:
            continue
        reject_mass = 0.0
        for b in range(0, d + 1):
            if _exact_mcnemar_reject(d, b, alpha):
                reject_mass += math.comb(d, b) * (p_t ** b) * ((1.0 - p_t) ** (d - b))
        power += p_d * reject_mass
    return {
        "power": round(power, 4),
        "n_pairs": n_pairs,
        "p01": round(float(p01), 4),
        "p10": round(float(p10), 4),
        "discordant_rate": round(pd, 4),
        "alpha": alpha,
        "method": "exact_two_sided_mcnemar",
    }


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
