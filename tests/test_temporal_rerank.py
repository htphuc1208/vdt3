"""Temporal-precedence re-ranking: an earlier-onset cause should outrank a late symptom."""
from __future__ import annotations

from telco_mas.shardrca.board import Blackboard, CandidateRootCause, Finding
from telco_mas.shardrca.synthesizer import SynthesizerResult
from telco_mas.shardrca.temporal import (
    component_onsets,
    precedence_scores,
    temporal_rerank,
)


def _board():
    return Blackboard(case_id="t", findings=[
        Finding(shard_id="s", modality="metrics", component="root", signal="cpu", window_start=100.0, score=1.0),
        Finding(shard_id="s", modality="metrics", component="symptom", signal="lat", window_start=140.0, score=1.0),
    ])


def test_component_onsets_take_earliest():
    board = Blackboard(case_id="t", findings=[
        Finding(shard_id="s", modality="metrics", component="a", signal="x", window_start=200.0, score=1.0),
        Finding(shard_id="s", modality="metrics", component="a", signal="y", window_start=100.0, score=1.0),
    ])
    assert component_onsets(board) == {"a": 100.0}


def test_precedence_earliest_is_one_latest_is_zero():
    p = precedence_scores({"root": 100.0, "symptom": 140.0})
    assert p["root"] == 1.0
    assert p["symptom"] == 0.0


def test_ties_all_get_full_precedence():
    p = precedence_scores({"a": 50.0, "b": 50.0})
    assert p == {"a": 1.0, "b": 1.0}


def _result():
    # Fusion wrongly ranks the late symptom first by a small margin.
    return SynthesizerResult(
        winner=CandidateRootCause(component="symptom", score=1.0),
        candidates=[
            CandidateRootCause(component="symptom", score=1.0),
            CandidateRootCause(component="root", score=0.9),
        ],
    )


def test_beta_zero_is_noop():
    out = temporal_rerank(_result(), _board(), beta=0.0)
    assert out.winner.component == "symptom"


def test_temporal_promotes_earlier_root():
    out = temporal_rerank(_result(), _board(), beta=1.0)
    # root: 0.9*(1+1*1.0)=1.8 ; symptom: 1.0*(1+1*0.0)=1.0
    assert out.winner.component == "root"


def test_no_onset_data_is_noop():
    empty = Blackboard(case_id="t", findings=[])
    out = temporal_rerank(_result(), empty, beta=2.0)
    assert out.winner.component == "symptom"
