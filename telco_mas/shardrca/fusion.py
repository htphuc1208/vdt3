"""Evidence-isolated candidate fusion for ShardRCA."""
from __future__ import annotations

from collections import Counter, defaultdict
from math import exp, log

from .board import (
    Blackboard,
    CandidateEvidence,
    CandidateRootCause,
    Finding,
    Modality,
    WorkerDistribution,
)
from .synthesizer import SynthesizerResult, infer_reason
from .weights import FusionWeights

# Retained as a module-level default for backward compatibility; the live path
# reads modality weights from a (freezable) FusionWeights artifact instead.
MODALITY_WEIGHTS: dict[str, float] = dict(FusionWeights.default().modality_weights)


def _worker_effective_weight(
    distributions: list[WorkerDistribution],
    component_set: set[str],
    correlation_rho: float,
) -> float:
    """Single equal log-pool weight = (effective independent experts) / N.

    Product-of-experts is only valid for *conditionally independent* sources.
    Evidence-isolated workers are not independent: they observe the same
    propagated symptoms of one root, so their errors correlate. We estimate that
    redundancy with the mean positive Pearson correlation between each worker's
    component-level posterior vector. High mean posterior correlation => the
    ensemble behaves like fewer independent experts:

        N_eff = 1 + (N - 1) * (1 - rho * mean_correlation)
        weight = N_eff / N          (applied equally to every worker)

    ``rho = 0`` gives weight 1 (unchanged PoE); a fully-agreeing ensemble at
    ``rho = 1`` gives weight 1/N (a weighted geometric mean = one effective
    expert). Returns 1.0 for degenerate ensembles (< 2 workers).
    """
    component_order = sorted(component_set)
    if not component_order:
        return 1.0
    vectors: list[list[float]] = []
    for dist in distributions:
        scope = {c for c in dist.candidate_scope if c in component_set}
        if not scope:
            continue
        vectors.append(_worker_component_vector(dist, component_order, scope))
    n = len(vectors)
    if n < 2 or correlation_rho <= 0.0:
        return 1.0
    correlations: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            # Negative correlation is disagreement, not redundant confirmation.
            correlations.append(max(0.0, _pearson_correlation(vectors[i], vectors[j])))
    mean_correlation = sum(correlations) / len(correlations) if correlations else 0.0
    n_eff = 1.0 + (n - 1) * (1.0 - correlation_rho * mean_correlation)
    return max(1.0 / n, min(1.0, n_eff / n))


def _worker_component_vector(
    distribution: WorkerDistribution,
    component_order: list[str],
    scope: set[str],
) -> list[float]:
    """Project a worker posterior onto component mass for correlation checks."""

    values = {component: 0.0 for component in component_order}
    scoped_items = [item for item in distribution.candidates if item.component in scope]
    explicit_mass = sum(max(0.0, float(item.probability)) for item in scoped_items)
    scale = 1.0 / explicit_mass if explicit_mass > 1.0 else 1.0
    scaled_explicit_mass = min(1.0, explicit_mass)
    explicit_components: set[str] = set()
    for item in scoped_items:
        probability = max(0.0, float(item.probability)) * scale
        values[item.component] += probability
        explicit_components.add(item.component)

    other_mass = min(float(distribution.other_mass), max(0.0, 1.0 - scaled_explicit_mass))
    remaining = [component for component in scope if component not in explicit_components]
    if other_mass > 0.0:
        if remaining:
            fallback = other_mass / len(remaining)
            for component in remaining:
                values[component] += fallback
        else:
            fallback = other_mass / max(1, len(scope))
            for component in scope:
                values[component] += fallback

    total = sum(values.values())
    if total <= 0.0:
        return [1.0 / len(component_order) for _ in component_order]
    return [values[component] / total for component in component_order]


def _pearson_correlation(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    centered_a = [value - mean_a for value in a]
    centered_b = [value - mean_b for value in b]
    denom_a = sum(value * value for value in centered_a)
    denom_b = sum(value * value for value in centered_b)
    denominator = (denom_a * denom_b) ** 0.5
    if denominator <= 0.0:
        return 0.0
    return sum(x * y for x, y in zip(centered_a, centered_b)) / denominator


def candidate_evidence_from_findings(
    shard_id: str,
    findings: list[Finding],
    *,
    max_candidates: int = 6,
) -> list[CandidateEvidence]:
    """Summarize one worker shard as local candidate evidence records."""

    if not findings:
        return []
    local_board = Blackboard(case_id=shard_id)
    local_board.extend(findings)
    candidates: list[CandidateEvidence] = []
    for rank, (component, score) in enumerate(local_board.top_components(max_candidates), start=1):
        evidence = local_board.evidence_for(component, limit=6)
        modality = _dominant_modality(evidence)
        reason = infer_reason(evidence)
        candidates.append(
            CandidateEvidence(
                component=component,
                reason_family=reason,
                support_score=float(score),
                refute_score=0.0,
                modality=modality,
                shard_id=shard_id,
                evidence_ptrs=[item.evidence_ptr for item in evidence if item.evidence_ptr][:6],
                local_rank=rank,
                rationale=_local_rationale(component, reason, evidence, score),
            )
        )
    return candidates


def fuse_candidate_evidence(
    evidence: list[CandidateEvidence],
    board: Blackboard,
    *,
    max_candidates: int = 5,
    weights: FusionWeights | None = None,
) -> SynthesizerResult:
    """Fuse local candidate distributions without asking one model to read the global board."""

    weights = weights or FusionWeights.load()
    if not evidence:
        fallback = CandidateRootCause(
            component="UNKNOWN",
            reason="unknown",
            confidence=0.0,
            rationale="No local candidate evidence was emitted.",
        )
        return SynthesizerResult(winner=fallback, candidates=[fallback], vote_breakdown={})

    by_component: dict[str, list[CandidateEvidence]] = defaultdict(list)
    display_name: dict[str, str] = {}
    for item in evidence:
        key = item.component.strip().lower()
        if not key:
            continue
        by_component[key].append(item)
        display_name.setdefault(key, item.component)

    scored: list[tuple[str, float]] = []
    reason_scores: dict[str, Counter[str]] = {}
    for key, items in by_component.items():
        weighted_sum = sum(_weighted_support(item, weights) for item in items)
        scored.append((key, weighted_sum))
        reasons: Counter[str] = Counter()
        for item in items:
            reasons[item.reason_family or "unknown"] += _weighted_support(item, weights)
        reason_scores[key] = reasons

    scored.sort(key=lambda item: item[1], reverse=True)
    total_positive_score = sum(max(0.0, score) for _, score in scored) or 1.0
    candidates: list[CandidateRootCause] = []
    for key, score in scored[:max_candidates]:
        items = by_component[key]
        reason = reason_scores[key].most_common(1)[0][0] if reason_scores[key] else "unknown"
        ptrs = _unique_ptrs(items)
        component = display_name[key]
        candidates.append(
            CandidateRootCause(
                component=component,
                reason=reason,
                occurrence_time=_first_occurrence(board.evidence_for(component, limit=6)),
                confidence=max(0.0, min(0.95, max(0.0, score) / total_positive_score)),
                rationale=_fusion_rationale(component, reason, items, score),
                evidence=ptrs,
                score=score,
            )
        )

    winner = candidates[0] if candidates else CandidateRootCause(component="UNKNOWN", reason="unknown")
    return SynthesizerResult(
        winner=winner,
        candidates=candidates or [winner],
        vote_breakdown={candidate.component: round(candidate.score, 4) for candidate in candidates},
    )


def fuse_worker_distributions(
    distributions: list[WorkerDistribution],
    board: Blackboard,
    *,
    components: list[str],
    reasons: list[str],
    max_candidates: int = 8,
    epsilon: float = 1e-6,
    weights: FusionWeights | None = None,
) -> SynthesizerResult:
    """Fuse local posteriors as a redundancy-discounted log-opinion pool.

    With the default weights (``correlation_rho=0``, ``temperature=1``) this is
    exactly the previous equal-weight product of experts. A preregistered
    validation artifact can down-weight correlated workers and temper the fused
    posterior, but current claim protocols use the default no-fit artifact.
    """

    weights = weights or FusionWeights.load()
    component_set = set(components)
    reason_set = set(reasons)
    pairs = [(component, reason) for component in components for reason in reasons]
    if not pairs:
        fallback = CandidateRootCause(component="UNKNOWN", reason="unknown")
        return SynthesizerResult(winner=fallback, candidates=[fallback], vote_breakdown={})

    pool_weight = _worker_effective_weight(distributions, component_set, weights.correlation_rho)
    logs: dict[tuple[str, str], float] = {pair: -log(len(pairs)) for pair in pairs}
    explicit_by_worker: dict[str, dict[tuple[str, str], CandidateEvidence]] = {}
    valid_distributions = []
    for distribution in distributions:
        scope = [component for component in distribution.candidate_scope if component in component_set]
        if not scope:
            continue
        w = pool_weight
        if weights.modality_reliability_enabled:
            # Unequal per-modality weight; this is what lets a more-reliable
            # modality overturn a numeric majority (i.e. change the argmax).
            w *= weights.modality_weight(str(distribution.modality))
        explicit: dict[tuple[str, str], CandidateEvidence] = {}
        explicit_probability: dict[tuple[str, str], float] = {}
        explicit_mass = 0.0
        for item in distribution.candidates:
            pair = (item.component, item.reason_family)
            if item.component not in scope or item.reason_family not in reason_set or pair in explicit:
                continue
            explicit[pair] = item
            probability = max(0.0, float(item.probability))
            explicit_probability[pair] = probability
            explicit_mass += probability
        if explicit_mass > 1.0 + 1e-6:
            scale = 1.0 / explicit_mass
            explicit_probability = {pair: probability * scale for pair, probability in explicit_probability.items()}
            explicit_mass = 1.0
        other_mass = min(float(distribution.other_mass), max(0.0, 1.0 - explicit_mass))
        remaining = max(1, len(scope) * len(reasons) - len(explicit))
        fallback_probability = max(epsilon, other_mass / remaining)
        # A modality-scoped worker only has an opinion over components in its
        # ``scope``. For pairs outside its scope it must *abstain* (contribute a
        # uniform probability over the full pair space), NOT contribute nothing:
        # contributing nothing is equivalent to asserting probability 1
        # (certainty), which asymmetrically favours under-covered components and
        # let artifact pairs win over the components a worker explicitly flagged.
        scope_set = set(scope)
        abstain_probability = 1.0 / len(pairs)
        for (component, reason) in pairs:
            if component in scope_set:
                item = explicit.get((component, reason))
                probability = max(
                    epsilon,
                    explicit_probability.get((component, reason), 0.0) if item is not None else fallback_probability,
                )
            else:
                probability = abstain_probability
            logs[(component, reason)] += w * log(probability)
        explicit_by_worker[distribution.worker_id] = explicit
        valid_distributions.append(distribution)

    if not valid_distributions:
        fallback = CandidateRootCause(component="UNKNOWN", reason="unknown")
        return SynthesizerResult(winner=fallback, candidates=[fallback], vote_breakdown={})

    temperature = max(1e-6, float(weights.temperature))
    maximum = max(logs.values())
    unnormalized = {pair: exp((value - maximum) / temperature) for pair, value in logs.items()}
    denominator = sum(unnormalized.values()) or 1.0
    posterior = {pair: value / denominator for pair, value in unnormalized.items()}
    ranked = sorted(posterior.items(), key=lambda item: _worker_pair_rank_key(item, explicit_by_worker))[:max_candidates]
    candidates: list[CandidateRootCause] = []
    for (component, reason), probability in ranked:
        evidence_items = [
            explicit[(component, reason)]
            for explicit in explicit_by_worker.values()
            if (component, reason) in explicit
        ]
        evidence_ptrs = _unique_ptrs(evidence_items)
        supporters = sorted(
            {
                item.worker_id or item.shard_id
                for item in evidence_items
                if item.worker_id or item.shard_id
            }
        )
        candidates.append(
            CandidateRootCause(
                component=component,
                reason=reason,
                occurrence_time=_first_occurrence(board.evidence_for(component, limit=20)),
                confidence=max(0.0, min(1.0, probability)),
                rationale=(
                    f"Equal-weight product-of-experts posterior={probability:.6f}; "
                    f"explicit_supporters={','.join(supporters) or 'none'}."
                ),
                evidence=evidence_ptrs,
                score=probability,
            )
        )
    winner = candidates[0]
    return SynthesizerResult(
        winner=winner,
        candidates=candidates,
        vote_breakdown={
            f"{candidate.component}|{candidate.reason}": round(candidate.score, 8)
            for candidate in candidates
        },
    )


def _worker_pair_rank_key(
    item: tuple[tuple[str, str], float],
    explicit_by_worker: dict[str, dict[tuple[str, str], CandidateEvidence]],
) -> tuple[float, float, int, int, tuple[str, str]]:
    pair, probability = item
    evidence_items = [
        explicit[pair]
        for explicit in explicit_by_worker.values()
        if pair in explicit
    ]
    support = sum(_weighted_support(evidence) for evidence in evidence_items)
    ptr_count = sum(1 for evidence in evidence_items for ptr in evidence.evidence_ptrs if ptr)
    supporter_count = len({
        evidence.worker_id or evidence.shard_id
        for evidence in evidence_items
        if evidence.worker_id or evidence.shard_id
    })
    return (-probability, -support, -supporter_count, -ptr_count, pair)


def _weighted_support(item: CandidateEvidence, weights: FusionWeights | None = None) -> float:
    weights = weights or FusionWeights.default()
    raw = float(item.support_score) - float(item.refute_score)
    rank_discount = 1.0 / max(1.0, float(item.local_rank)) ** 0.5
    return raw * weights.modality_weight(str(item.modality)) * rank_discount


def _dominant_modality(evidence: list[Finding]) -> Modality:
    if not evidence:
        return "auxiliary"
    counts = Counter(item.modality for item in evidence)
    return counts.most_common(1)[0][0]


def _unique_ptrs(items: list[CandidateEvidence]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in sorted(items, key=_weighted_support, reverse=True):
        for ptr in item.evidence_ptrs:
            if ptr and ptr not in seen:
                out.append(ptr)
                seen.add(ptr)
            if len(out) >= 8:
                return out
    return out


def _first_occurrence(findings: list[Finding]) -> str | None:
    for finding in findings:
        if finding.window_start is not None:
            return str(finding.window_start)
    return None


def _local_rationale(component: str, reason: str, evidence: list[Finding], score: float) -> str:
    modalities = sorted({item.modality for item in evidence})
    return (
        f"{component} is local rank evidence for {reason}; "
        f"score={score:.3f}; modalities={','.join(modalities) or 'none'}."
    )


def _fusion_rationale(component: str, reason: str, evidence: list[CandidateEvidence], score: float) -> str:
    modalities = sorted({str(item.modality) for item in evidence})
    shards = sorted({item.shard_id for item in evidence if item.shard_id})
    return (
        f"{component} won local-candidate fusion for {reason}; score={score:.3f}; "
        f"modalities={','.join(modalities) or 'none'}; shards={len(shards)}."
    )
