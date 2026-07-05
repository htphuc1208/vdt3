"""A-priori power / sample-size analysis for the paired McNemar RCA claims.

These tests pin the planning tools that gate any confirmatory run (PLAN v6 §4):
a clear-win claim must be preceded by a sample-size justification, not run at a
convenient n and interpreted post hoc.
"""
from __future__ import annotations

import math

from telco_mas.evaluation.stats import (
    _norm_ppf,
    mcnemar_exact_power,
    mcnemar_required_pairs,
)


def test_norm_ppf_known_quantiles():
    assert math.isclose(_norm_ppf(0.975), 1.959964, abs_tol=1e-4)
    assert math.isclose(_norm_ppf(0.80), 0.841621, abs_tol=1e-4)
    assert math.isclose(_norm_ppf(0.5), 0.0, abs_tol=1e-6)


def test_required_pairs_grows_as_effect_shrinks():
    big = mcnemar_required_pairs(0.15, 0.35)["required_pairs"]
    small = mcnemar_required_pairs(0.10, 0.30)["required_pairs"]
    # A smaller marginal effect needs strictly more paired cases.
    assert small > big > 0


def test_required_pairs_rejects_infeasible_inputs():
    # delta cannot exceed the discordant proportion.
    assert mcnemar_required_pairs(0.40, 0.30)["required_pairs"] is None
    assert mcnemar_required_pairs(0.0, 0.30)["required_pairs"] is None


def test_openrca_n51_is_underpowered():
    """The frozen OpenRCA Telecom run (n=51) cannot confirm a 10pp effect."""
    power = mcnemar_exact_power(51, p01=0.20, p10=0.10)["power"]
    assert power < 0.30  # ~0.18 in practice — the null result is uninformative


def test_power_is_monotonic_in_n():
    low = mcnemar_exact_power(40, 0.25, 0.10)["power"]
    high = mcnemar_exact_power(200, 0.25, 0.10)["power"]
    assert high > low


def test_no_effect_gives_near_alpha_power():
    # Under the null (p01 == p10) the exact test rejects at most ~alpha.
    power = mcnemar_exact_power(120, 0.15, 0.15)["power"]
    assert power <= 0.05 + 1e-9
