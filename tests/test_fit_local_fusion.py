"""Guards the offline local-fusion fit's disjointness logic.

The single most important property is that dev/val selection reproduces the hard
split's runtime-id hashing exactly, so locked holdout / confirmatory cases can be
excluded and never contaminate a fit.
"""
from __future__ import annotations

import hashlib

from telco_mas.shardrca.fit_local_fusion import runtime_case_id


def test_runtime_case_id_matches_split_hashing():
    raw = "RCAEval-RE2-TT-ts-travel_cpu-1"
    expected = "RE2-TT-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    assert runtime_case_id(raw) == expected


def test_runtime_case_id_source_prefix():
    assert runtime_case_id("RCAEval-RE3-OB-adservice_f1-2").startswith("RE3-OB-")
    assert runtime_case_id("RCAEval-RE1-SS-catalogue_mem-1").startswith("RE1-SS-")
