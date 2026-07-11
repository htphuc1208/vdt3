"""OpenRCA-compatible evaluator for formatted predictions."""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable


@dataclass(frozen=True)
class OpenRCAEvalResult:
    row_id: int
    task_index: str
    instruction: str
    prediction: str
    scoring_points: str
    passed: list[str]
    failed: list[str]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "task_index": self.task_index,
            "instruction": self.instruction,
            "prediction": self.prediction,
            "scoring_points": self.scoring_points,
            "passed": "\n".join(self.passed),
            "failed": "\n".join(self.failed),
            "score": self.score,
            "strict_correct": self.score == 1.0,
        }


def evaluate_prediction(prediction: str, scoring_points: str) -> tuple[list[str], list[str], float]:
    predicted = _parse_prediction(prediction)
    components = re.findall(r"The (?:\d+-th|only) predicted root cause component is ([^\n]+)", scoring_points)
    reasons = re.findall(r"The (?:\d+-th|only) predicted root cause reason is ([^\n]+)", scoring_points)
    times = re.findall(
        r"The (?:\d+-th|only) root cause occurrence time is within 1 minutes \(i.e., <=1min\) of ([^\n]+)",
        scoring_points,
    )
    scoring_length = max(len(components), len(reasons), len(times))
    score_count = len(components) + len(reasons) + len(times)
    if score_count == 0:
        return [], [], 0.0

    best_score = 0
    best_passing: list[str] = []
    if scoring_length == len(predicted):
        for permutation in itertools.permutations(predicted):
            current_score = 0
            current_passing = []
            for index in range(scoring_length):
                if len(components) == scoring_length and permutation[index]["root cause component"] == components[index]:
                    current_score += 1
                    current_passing.append(components[index])
                if len(reasons) == scoring_length and permutation[index]["root cause reason"] == reasons[index]:
                    current_score += 1
                    current_passing.append(reasons[index])
                if len(times) == scoring_length and _within_one_minute(
                    times[index], permutation[index]["root cause occurrence datetime"]
                ):
                    current_score += 1
                    current_passing.append(times[index])
            if current_score > best_score:
                best_score = current_score
                best_passing = current_passing

    failed = list(set(components + reasons + times) - set(best_passing))
    # Match microsoft/OpenRCA main/evaluate.py exactly. This matters for
    # aggregate partial accuracy even though strict accuracy is unchanged.
    return best_passing, failed, round(best_score / score_count, 2)


def build_eval_result(
    *,
    row_id: int,
    task_index: str,
    instruction: str,
    prediction: str,
    scoring_points: str,
) -> OpenRCAEvalResult:
    passed, failed, score = evaluate_prediction(prediction, scoring_points)
    return OpenRCAEvalResult(row_id, task_index, instruction, prediction, scoring_points, passed, failed, score)


def summarize_results(results: Iterable[OpenRCAEvalResult]) -> list[dict[str, Any]]:
    all_items = list(results)
    buckets = {"easy": [], "middle": [], "hard": []}
    for item in all_items:
        buckets[_difficulty(item.task_index)].append(item)
    rows = []
    for name, items in [*buckets.items(), ("total", all_items)]:
        count = len(items)
        strict = sum(1 for item in items if item.score == 1.0)
        partial = sum(item.score for item in items)
        rows.append({
            "difficulty": name,
            "total": count,
            "strict_correct": strict,
            "strict_accuracy": round(strict / count, 4) if count else 0.0,
            "partial_score": round(partial, 4),
            "partial_accuracy": round(partial / count, 4) if count else 0.0,
        })
    return rows


def _parse_prediction(prediction: str) -> list[dict[str, str]]:
    pattern = (
        r'{\s*'
        r'(?:"root cause occurrence datetime":\s*"(.*?)")?,?\s*'
        r'(?:"root cause component":\s*"(.*?)")?,?\s*'
        r'(?:"root cause reason":\s*"(.*?)")?\s*}'
    )
    matches = re.findall(pattern, prediction)
    return [
        {
            "root cause occurrence datetime": datetime_text,
            "root cause component": component,
            "root cause reason": reason,
        }
        for datetime_text, component, reason in matches
        if datetime_text or component or reason
    ]


def _within_one_minute(expected: str, actual: str) -> bool:
    try:
        expected_dt = datetime.strptime(expected, "%Y-%m-%d %H:%M:%S")
        actual_dt = datetime.strptime(actual, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return False
    return abs(expected_dt - actual_dt).total_seconds() <= 60


def _difficulty(task_index: str) -> str:
    try:
        task_id = int(str(task_index).split("_")[1])
    except (IndexError, ValueError):
        return "hard"
    if task_id <= 3:
        return "easy"
    if task_id <= 6:
        return "middle"
    return "hard"
