"""Deterministic telemetry mining tools for ShardRCA."""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .board import Finding
from .catalog import ShardSpec, _align_bounds_to_series, _choose_timestamp_column, _parse_time_value, _pd


@dataclass
class _Stats:
    count: int = 0
    total: float = 0.0
    sumsq: float = 0.0
    min_value: float | None = None
    max_value: float | None = None
    first_time: float | None = None
    last_time: float | None = None

    def update(self, values, times=None) -> None:
        pd = _pd()
        nums = pd.to_numeric(values, errors="coerce").dropna()
        if nums.empty:
            return
        self.count += int(len(nums))
        self.total += float(nums.sum())
        self.sumsq += float((nums * nums).sum())
        current_min = float(nums.min())
        current_max = float(nums.max())
        self.min_value = current_min if self.min_value is None else min(self.min_value, current_min)
        self.max_value = current_max if self.max_value is None else max(self.max_value, current_max)
        if times is not None:
            t = pd.to_numeric(times, errors="coerce").dropna()
            if not t.empty:
                t_min = float(t.min())
                t_max = float(t.max())
                self.first_time = t_min if self.first_time is None else min(self.first_time, t_min)
                self.last_time = t_max if self.last_time is None else max(self.last_time, t_max)

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.sample_variance)

    @property
    def sample_variance(self) -> float:
        if self.count <= 1:
            return 0.0
        numerator = self.sumsq - (self.total * self.total / self.count)
        return max(0.0, numerator / (self.count - 1))


@dataclass(frozen=True)
class _WelchShift:
    delta: float
    t_stat: float
    df: float
    p_value: float
    standard_error: float


_MIN_P_VALUE = 1e-300


def mine_shard(spec: ShardSpec, *, limit: int | None = 20, chunksize: int = 50_000) -> list[Finding]:
    if spec.modality == "metrics":
        return mine_metric_shifts(
            spec.paths,
            shard_id=spec.shard_id,
            pivot_time=spec.query_time,
            start_time=spec.start_time,
            end_time=spec.end_time,
            components=spec.components,
            limit=limit,
            chunksize=chunksize,
        )
    if spec.modality == "logs":
        return mine_log_patterns(
            spec.paths,
            shard_id=spec.shard_id,
            pivot_time=spec.query_time,
            start_time=spec.start_time,
            end_time=spec.end_time,
            components=spec.components,
            limit=limit,
            chunksize=chunksize,
        )
    if spec.modality == "traces":
        return mine_trace_latency(
            spec.paths,
            shard_id=spec.shard_id,
            pivot_time=spec.query_time,
            start_time=spec.start_time,
            end_time=spec.end_time,
            components=spec.components,
            limit=limit,
            chunksize=chunksize,
        )
    return []


def mine_metric_shifts(
    paths: Iterable[str | Path],
    *,
    shard_id: str = "metrics_0",
    pivot_time: float | int | str | None = None,
    start_time: float | int | str | None = None,
    end_time: float | int | str | None = None,
    components: list[str] | None = None,
    limit: int | None = 20,
    chunksize: int = 50_000,
) -> list[Finding]:
    return _mine_numeric_timeseries(
        paths,
        modality="metrics",
        shard_id=shard_id,
        pivot_time=pivot_time,
        start_time=start_time,
        end_time=end_time,
        components=components,
        limit=limit,
        chunksize=chunksize,
        preferred_value_cols=("value",),
        preferred_component_cols=("cmdb_id", "component", "service", "container_name", "serviceName"),
        preferred_metric_cols=("name", "metric", "item", "dsName"),
    )


def mine_trace_latency(
    paths: Iterable[str | Path],
    *,
    shard_id: str = "traces_0",
    pivot_time: float | int | str | None = None,
    start_time: float | int | str | None = None,
    end_time: float | int | str | None = None,
    components: list[str] | None = None,
    limit: int | None = 20,
    chunksize: int = 50_000,
) -> list[Finding]:
    return _mine_numeric_timeseries(
        paths,
        modality="traces",
        shard_id=shard_id,
        pivot_time=pivot_time,
        start_time=start_time,
        end_time=end_time,
        components=components,
        limit=limit,
        chunksize=chunksize,
        preferred_value_cols=("duration", "elapsedTime", "latency", "value"),
        preferred_component_cols=("cmdb_id", "serviceName", "component", "service", "container_name"),
        preferred_metric_cols=("operationName", "methodName", "dsName", "name", "metric"),
    )


def mine_log_patterns(
    paths: Iterable[str | Path],
    *,
    shard_id: str = "logs_0",
    pivot_time: float | int | str | None = None,
    start_time: float | int | str | None = None,
    end_time: float | int | str | None = None,
    components: list[str] | None = None,
    limit: int | None = 20,
    chunksize: int = 50_000,
) -> list[Finding]:
    pd = _pd()
    component_filter = {c.lower() for c in components or []}
    counters: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: {"pre": 0, "post": 0})
    evidence: dict[tuple[str, str, str], str] = {}
    numeric_paths: list[str | Path] = []

    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() != ".csv":
            continue
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        columns = [str(col) for col in header.columns]
        message_col = _first_existing(columns, ("message", "msg", "log", "content", "body"))
        if not message_col:
            numeric_paths.append(path)
            continue
        time_col = _choose_timestamp_column(columns)
        component_col = _first_existing(columns, ("container_name", "component", "service", "serviceName", "cmdb_id"))
        for chunk in pd.read_csv(path, chunksize=chunksize):
            if message_col not in chunk:
                continue
            if time_col and time_col in chunk:
                times = _coerce_times(chunk[time_col])
                pre_mask, post_mask = _pre_post_masks(times, pivot_time, start_time, end_time)
            else:
                split = max(1, len(chunk) // 2)
                positions = pd.Series(range(len(chunk)), index=chunk.index)
                pre_mask = positions < split
                post_mask = ~pre_mask
            comps = chunk[component_col].astype(str) if component_col and component_col in chunk else pd.Series(path.stem, index=chunk.index)
            for phase, mask in (("pre", pre_mask), ("post", post_mask)):
                indices = chunk.index[mask]
                if indices.empty:
                    continue
                component_values = comps.loc[indices].astype(str)
                if component_filter:
                    keep = component_values.str.lower().isin(component_filter)
                    if not bool(keep.any()):
                        continue
                    component_values = component_values.loc[keep]
                    indices = component_values.index
                local = pd.DataFrame(
                    {
                        "component": component_values,
                        "template": chunk.loc[indices, message_col].astype(str).map(normalise_log_template),
                    },
                    index=indices,
                )
                for (component, template), group in local.groupby(["component", "template"], dropna=True):
                    key = (component, template, "log_template")
                    counters[key][phase] += int(len(group))
                    evidence.setdefault(key, f"{path.name}:{group.index[0]}")

    findings: list[Finding] = []
    for (component, template, signal), counts in counters.items():
        pre = counts["pre"]
        post = counts["post"]
        delta = post - pre
        if pre == 0 and post == 0:
            continue
        score = _log_count_shift_score(pre, post)
        findings.append(
            Finding(
                shard_id=shard_id,
                modality="logs",
                component=component,
                signal=signal,
                direction="more" if delta >= 0 else "less",
                magnitude=float(abs(delta)),
                score=round(float(score), 6),
                evidence_ptr=evidence.get((component, template, signal), ""),
                summary=f"log template count changed pre={pre} post={post}: {template[:120]}",
                metadata={
                    "template": template,
                    "pre_count": pre,
                    "post_count": post,
                    "score_method": "laplace_log_ratio_prior_support",
                    "laplace_prior_per_window": 1.0,
                    "support_prior_mass": 2.0,
                },
            )
        )

    if numeric_paths:
        findings.extend(
            _mine_numeric_timeseries(
                numeric_paths,
                modality="logs",
                shard_id=shard_id,
                pivot_time=pivot_time,
                start_time=start_time,
                end_time=end_time,
                components=components,
                limit=limit,
                chunksize=chunksize,
                preferred_value_cols=("value",),
                preferred_component_cols=("component", "service", "container_name", "serviceName"),
                preferred_metric_cols=("name", "metric"),
            )
        )
    ranked = sorted(findings, key=lambda item: item.score, reverse=True)
    return ranked if limit is None else ranked[: max(1, limit)]


def normalise_log_template(message: Any) -> str:
    text = str(message).strip().lower()
    text = re.sub(r"\b[0-9a-f]{8,}\b", "<hex>", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}[ t]\d{2}:\d{2}:\d{2}(?:\.\d+)?\b", "<datetime>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", text)
    text = re.sub(r"\s+", " ", text)
    return text[:300]


def _log_count_shift_score(pre: int, post: int, *, prior: float = 1.0, support_tau: float | None = None) -> float:
    """Calibrate log-count shifts with Laplace smoothing and prior support.

    ``prior`` is the symmetric Laplace pseudo-count per window. When
    ``support_tau`` is not supplied, the support shrinkage uses the effective
    prior sample mass implied by that smoothing (two windows times ``prior``).
    That keeps one-off template changes from outranking supported shifts without
    introducing an unrelated free constant.
    """

    total = max(0, pre) + max(0, post)
    if total == 0:
        return 0.0
    log_ratio = abs(math.log((max(0, post) + prior) / (max(0, pre) + prior)))
    prior_support = 2.0 * max(0.0, prior) if support_tau is None else max(0.0, support_tau)
    support = math.sqrt(total / (total + prior_support)) if prior_support > 0 else 1.0
    return log_ratio * support


def _mine_numeric_timeseries(
    paths: Iterable[str | Path],
    *,
    modality: str,
    shard_id: str,
    pivot_time: float | int | str | None,
    start_time: float | int | str | None,
    end_time: float | int | str | None,
    components: list[str] | None,
    limit: int | None,
    chunksize: int,
    preferred_value_cols: tuple[str, ...],
    preferred_component_cols: tuple[str, ...],
    preferred_metric_cols: tuple[str, ...],
) -> list[Finding]:
    pd = _pd()
    component_filter = {c.lower() for c in components or []}
    pre: dict[tuple[str, str, str], _Stats] = defaultdict(_Stats)
    post: dict[tuple[str, str, str], _Stats] = defaultdict(_Stats)
    evidence: dict[tuple[str, str, str], str] = {}

    for raw_path in paths:
        path = Path(raw_path)
        for chunk in _read_table_chunks(path, chunksize):
            if chunk.empty:
                continue
            columns = [str(col) for col in chunk.columns]
            time_col = _choose_timestamp_column(columns)
            if not time_col or time_col not in chunk:
                continue
            times = _coerce_times(chunk[time_col])
            pre_mask, post_mask = _pre_post_masks(times, pivot_time, start_time, end_time)
            long_cols = _long_columns(
                columns,
                preferred_value_cols=preferred_value_cols,
                preferred_component_cols=preferred_component_cols,
                preferred_metric_cols=preferred_metric_cols,
            )
            if long_cols:
                value_col, component_col, metric_col = long_cols
                local = chunk[[value_col, component_col, metric_col]].copy()
                local["_time"] = times
                local["_phase"] = "other"
                local.loc[pre_mask, "_phase"] = "pre"
                local.loc[post_mask, "_phase"] = "post"
                local = local[local["_phase"].isin({"pre", "post"})]
                for (component, metric, phase), group in local.groupby([component_col, metric_col, "_phase"], dropna=True):
                    component = str(component)
                    if component_filter and component.lower() not in component_filter:
                        continue
                    metric = str(metric)
                    key = (component, metric, path.name)
                    target = pre if phase == "pre" else post
                    target[key].update(group[value_col], group["_time"])
                    evidence.setdefault(key, f"{path.name}:{metric}")
                continue

            value_cols = [
                col for col in columns
                if col != time_col and col.lower() not in {"timestamp", "datetime", "date"}
            ]
            for col in value_cols:
                component, metric = split_metric_column(col)
                if component_filter and component.lower() not in component_filter:
                    continue
                values = pd.to_numeric(chunk[col], errors="coerce")
                if values.notna().sum() == 0:
                    continue
                key = (component, metric, path.name)
                pre[key].update(values.loc[pre_mask], times.loc[pre_mask])
                post[key].update(values.loc[post_mask], times.loc[post_mask])
                evidence.setdefault(key, f"{path.name}:{col}")

    findings = _stats_to_findings(
        pre,
        post,
        evidence,
        shard_id=shard_id,
        modality=modality,
        limit=limit,
    )
    return findings


def _stats_to_findings(
    pre: dict[tuple[str, str, str], _Stats],
    post: dict[tuple[str, str, str], _Stats],
    evidence: dict[tuple[str, str, str], str],
    *,
    shard_id: str,
    modality: str,
    limit: int | None,
) -> list[Finding]:
    tests: list[tuple[tuple[str, str, str], _Stats, _Stats, _WelchShift]] = []
    for key, post_stats in post.items():
        pre_stats = pre.get(key, _Stats())
        if pre_stats.count == 0 or post_stats.count == 0:
            continue
        tests.append((key, pre_stats, post_stats, _welch_mean_shift(pre_stats, post_stats)))

    q_values = _benjamini_hochberg([shift.p_value for _, _, _, shift in tests])
    findings: list[Finding] = []
    for (key, pre_stats, post_stats, shift), q_value in zip(tests, q_values):
        component, signal, source = key
        delta = shift.delta
        score = _fdr_significance_score(q_value)
        magnitude = abs(delta) / (abs(pre_stats.mean) + 1.0)
        direction = "high" if delta >= 0 else "low"
        findings.append(
            Finding(
                shard_id=shard_id,
                modality=modality,  # type: ignore[arg-type]
                component=component,
                signal=signal,
                direction=direction,
                magnitude=round(float(magnitude), 6),
                score=round(float(score), 6),
                window_start=post_stats.first_time,
                window_end=post_stats.last_time,
                evidence_ptr=evidence.get(key, source),
                summary=(
                    f"{signal} shifted {direction}: pre_mean={pre_stats.mean:.4g}, "
                    f"post_mean={post_stats.mean:.4g}, delta={delta:.4g}, "
                    f"welch_t={shift.t_stat:.4g}, q={q_value:.4g}, "
                    f"pre_n={pre_stats.count}, post_n={post_stats.count}"
                ),
                metadata={
                    "source_file": source,
                    "score_method": "welch_t_bh_fdr_neg_log10_q",
                    "multiple_testing": "benjamini_hochberg",
                    "pre_mean": pre_stats.mean,
                    "post_mean": post_stats.mean,
                    "pre_std": pre_stats.std,
                    "post_std": post_stats.std,
                    "pre_n": pre_stats.count,
                    "post_n": post_stats.count,
                    "delta": delta,
                    "standard_error": shift.standard_error,
                    "welch_t": _finite_float_or_none(shift.t_stat),
                    "welch_t_infinite": not math.isfinite(shift.t_stat),
                    "welch_df": _finite_float_or_none(shift.df),
                    "p_value": shift.p_value,
                    "q_value": q_value,
                },
            )
        )
    ranked = sorted(findings, key=lambda item: (item.score, item.magnitude), reverse=True)
    return ranked if limit is None else ranked[: max(1, limit)]


def _welch_mean_shift(pre_stats: _Stats, post_stats: _Stats) -> _WelchShift:
    """Two-sample mean-shift test using Welch's unequal-variance statistic."""

    delta = post_stats.mean - pre_stats.mean
    if pre_stats.count < 2 or post_stats.count < 2:
        return _WelchShift(delta=delta, t_stat=0.0, df=0.0, p_value=1.0, standard_error=math.inf)

    pre_var = pre_stats.sample_variance
    post_var = post_stats.sample_variance
    pre_term = pre_var / pre_stats.count
    post_term = post_var / post_stats.count
    se2 = pre_term + post_term
    if se2 <= 0.0:
        if abs(delta) <= 0.0:
            return _WelchShift(delta=delta, t_stat=0.0, df=math.inf, p_value=1.0, standard_error=0.0)
        signed_inf = math.copysign(math.inf, delta)
        return _WelchShift(delta=delta, t_stat=signed_inf, df=math.inf, p_value=_MIN_P_VALUE, standard_error=0.0)

    se = math.sqrt(se2)
    t_stat = delta / se
    denom = 0.0
    if pre_stats.count > 1:
        denom += (pre_term * pre_term) / (pre_stats.count - 1)
    if post_stats.count > 1:
        denom += (post_term * post_term) / (post_stats.count - 1)
    df = (se2 * se2 / denom) if denom > 0.0 else math.inf
    p_value = _student_t_two_sided_pvalue(abs(t_stat), df)
    return _WelchShift(delta=delta, t_stat=t_stat, df=df, p_value=p_value, standard_error=se)


def _student_t_two_sided_pvalue(t_abs: float, df: float) -> float:
    if not math.isfinite(t_abs):
        return _MIN_P_VALUE
    if t_abs <= 0.0:
        return 1.0
    if not math.isfinite(df) or df > 1e7:
        return max(_MIN_P_VALUE, min(1.0, math.erfc(t_abs / math.sqrt(2.0))))
    if df <= 0.0:
        return 1.0
    x = df / (df + t_abs * t_abs)
    return max(_MIN_P_VALUE, min(1.0, _regularized_incomplete_beta(0.5 * df, 0.5, x)))


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_continued_fraction(a, b, x) / a
    return 1.0 - bt * _beta_continued_fraction(b, a, 1.0 - x) / b


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3e-14
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    m = len(p_values)
    if m == 0:
        return []
    clipped = [max(_MIN_P_VALUE, min(1.0, float(p))) for p in p_values]
    order = sorted(range(m), key=lambda index: clipped[index])
    adjusted = [1.0] * m
    running = 1.0
    for rank, index in enumerate(reversed(order), start=1):
        original_rank = m - rank + 1
        q_value = clipped[index] * m / original_rank
        running = min(running, q_value)
        adjusted[index] = min(1.0, running)
    return adjusted


def _fdr_significance_score(q_value: float) -> float:
    q = max(_MIN_P_VALUE, min(1.0, float(q_value)))
    return max(0.0, -math.log10(q))


def _finite_float_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def split_metric_column(column: str) -> tuple[str, str]:
    if "_" not in column:
        return column, ""
    component, metric = column.rsplit("_", 1)
    if metric.startswith("container-") and "_" in component:
        component, metric_prefix = component.split("_", 1)
        metric = f"{metric_prefix}_{metric}"
    return component, metric


def _read_table_chunks(path: Path, chunksize: int):
    pd = _pd()
    if path.suffix.lower() == ".parquet":
        try:
            frame = pd.read_parquet(path)
            for start in range(0, len(frame), chunksize):
                yield frame.iloc[start : start + chunksize]
        except Exception:
            return
        return
    if path.suffix.lower() == ".json":
        try:
            frame = pd.read_json(path)
            yield frame
        except Exception:
            return
        return
    if path.suffix.lower() != ".csv":
        return
    try:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            yield chunk
    except Exception:
        return


def _coerce_times(series):
    pd = _pd()
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.astype("float64")
    parsed = pd.to_datetime(series, errors="coerce")
    values = parsed.astype("int64") / 1_000_000_000
    values = values.where(parsed.notna())
    return values.astype("float64")


def _pre_post_masks(times, pivot_time, start_time=None, end_time=None, *, lookback_s: int = 30 * 60, lookahead_s: int = 30 * 60):
    pd = _pd()
    pivot = _parse_time_value(pivot_time)
    start = _parse_time_value(start_time)
    end = _parse_time_value(end_time)
    valid = times.notna()
    if pivot is None and start is None and end is None:
        observed = times[valid]
        if observed.empty:
            false = pd.Series(False, index=times.index)
            return false, false
        pivot_aligned = float(observed.median())
        pre_start = float(observed.min())
        post_end = float(observed.max())
    elif pivot is not None:
        pivot_aligned, _ = _align_bounds_to_series(pivot, pivot, times)
        # lookback_s/lookahead_s are expressed in seconds; scale them into the
        # timestamp series' native unit (s/ms/us/ns) so the pre/post windows are
        # wide enough to contain samples. Basing this on the series magnitude
        # (not on whether the pivot was rescaled) fixes the ms-timestamp case
        # where an already-ms pivot yielded a 1800 ms pre-window (0 samples).
        scale = _series_time_scale(times)
        pre_start = pivot_aligned - lookback_s * scale
        post_end = pivot_aligned + lookahead_s * scale
    else:
        aligned_start, aligned_end = _align_bounds_to_series(start or end or 0.0, end or start or 0.0, times)
        pivot_aligned = (aligned_start + aligned_end) / 2.0
        pre_start = aligned_start
        post_end = aligned_end
    pre_mask = valid & (times >= pre_start) & (times < pivot_aligned)
    post_mask = valid & (times >= pivot_aligned) & (times <= post_end)
    return pre_mask, post_mask


def _series_time_scale(times) -> float:
    """Return the multiplier that converts seconds into the timestamp series'
    native unit, inferred from the magnitude of the (epoch) timestamps.

    Epoch magnitudes: seconds ~1e9, milliseconds ~1e12, microseconds ~1e15,
    nanoseconds ~1e18. This mirrors the unit inference in
    ``_align_bounds_to_series`` so pre/post windows expressed in seconds are
    sized correctly regardless of timestamp resolution.
    """
    pd = _pd()
    values = times.dropna() if hasattr(times, "dropna") else pd.Series(times).dropna()
    if len(values) == 0:
        return 1.0
    median = float(pd.Series(values).abs().median())
    if median > 1e17:
        return 1_000_000_000.0  # nanoseconds
    if median > 1e14:
        return 1_000_000.0  # microseconds
    if median > 1e11:
        return 1_000.0  # milliseconds
    return 1.0  # seconds


def _long_columns(
    columns: list[str],
    *,
    preferred_value_cols: tuple[str, ...],
    preferred_component_cols: tuple[str, ...],
    preferred_metric_cols: tuple[str, ...],
) -> tuple[str, str, str] | None:
    value_col = _first_existing(columns, preferred_value_cols)
    component_col = _first_existing(columns, preferred_component_cols)
    metric_col = _first_existing(columns, preferred_metric_cols)
    if value_col and component_col and metric_col:
        return value_col, component_col, metric_col
    return None


def _first_existing(columns: list[str], candidates: Iterable[str]) -> str | None:
    lower = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def summarise_findings(findings: list[Finding], *, limit: int = 20) -> list[dict[str, Any]]:
    return [finding.compact() for finding in sorted(findings, key=lambda item: item.score, reverse=True)[:limit]]


def component_counter(findings: list[Finding]) -> Counter:
    counter: Counter = Counter()
    for finding in findings:
        counter[finding.component] += finding.score
    return counter
