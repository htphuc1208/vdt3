"""The four RCA decision mechanisms, evaluated on synthetic telecom cases.

Each maps a Case -> a ranked list of candidate root components. Hit@1 = the true
root is ranked first. The mechanisms deliberately isolate one signal each so the
regime sweep can attribute wins/losses:

  de_collapse       : loudest anomaly (peak magnitude).      -- score only
  topology_causal   : explanatory coverage over the graph.   -- structure only
  het_temporal      : earliest onset (temporal precedence).  -- time only
  alarm_correlation : upstream-of-most-alarms, then earliest. -- structure + time
"""
from __future__ import annotations

from .telco_rca_sim import Case, descendants


def _coverage(case: Case, node: str) -> int:
    """How many symptomatic nodes a fault at ``node`` explains (itself + served)."""
    served = descendants(case.topo, node)
    return len(case.symptomatic & served) + (1 if node in case.symptomatic else 0)


def de_collapse(case: Case) -> list[str]:
    return sorted(case.candidates, key=lambda n: case.peak[n], reverse=True)


def topology_causal(case: Case) -> list[str]:
    return sorted(case.candidates, key=lambda n: (_coverage(case, n), case.peak[n]), reverse=True)


def het_temporal(case: Case) -> list[str]:
    # earliest onset first; break ties by more coverage then louder.
    return sorted(case.candidates, key=lambda n: (case.onset[n], -_coverage(case, n), -case.peak[n]))


def alarm_correlation(case: Case) -> list[str]:
    # classic root-cause-alarm: maximise downstream coverage, then earliest onset.
    return sorted(case.candidates, key=lambda n: (-_coverage(case, n), case.onset[n], -case.peak[n]))


METHODS = {
    "de_collapse": de_collapse,
    "topology_causal": topology_causal,
    "het_temporal": het_temporal,
    "alarm_correlation": alarm_correlation,
}
