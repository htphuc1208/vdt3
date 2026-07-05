"""Miner worker wrappers for ShardRCA."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

from ..schemas import UsageStats
from .board import Finding
from .catalog import ShardSpec
from .mining import mine_shard


@dataclass
class MinerResult:
    shard_id: str
    findings: list[Finding]
    usage: UsageStats
    notes: str = ""


class MinerWorker:
    """A deterministic worker over one shard.

    The v4 architecture lets an LLM read tool outputs, but the mining itself is
    intentionally deterministic and compact. This keeps labels out of prompts
    and makes offline smoke tests meaningful.
    """

    def __init__(self, *, limit: int = 20, chunksize: int = 50_000) -> None:
        self.limit = limit
        self.chunksize = chunksize

    def run(self, spec: ShardSpec) -> MinerResult:
        findings = mine_shard(spec, limit=self.limit, chunksize=self.chunksize)
        return MinerResult(
            shard_id=spec.shard_id,
            findings=findings,
            usage=UsageStats(tool_calls=1),
            notes=f"mined {len(findings)} findings from {len(spec.paths)} files",
        )

    def run_many(self, specs: list[ShardSpec]) -> list[MinerResult]:
        """Mine compatible component shards in one physical telemetry scan.

        Findings are partitioned back into their original shards before they
        reach an investigator. This preserves evidence isolation while avoiding
        one full CSV scan per component group.
        """

        indexed = list(enumerate(specs))
        grouped = sorted(indexed, key=lambda item: _scan_key(item[1]))
        results: dict[int, MinerResult] = {}
        for _, members_iter in groupby(grouped, key=lambda item: _scan_key(item[1])):
            members = list(members_iter)
            component_members = [(index, spec) for index, spec in members if spec.components]
            unfiltered_members = [(index, spec) for index, spec in members if not spec.components]

            if component_members:
                union = sorted({component.lower() for _, spec in component_members for component in spec.components})
                template = component_members[0][1]
                shared_spec = ShardSpec(
                    shard_id=f"shared_{template.modality}",
                    modality=template.modality,
                    paths=template.paths,
                    query_time=template.query_time,
                    start_time=template.start_time,
                    end_time=template.end_time,
                    components=union,
                    metadata={"shared_scan": True},
                )
                shared = mine_shard(shared_spec, limit=None, chunksize=self.chunksize)
                for index, spec in component_members:
                    allowed = {component.lower() for component in spec.components}
                    local = [
                        finding.model_copy(update={"shard_id": spec.shard_id})
                        for finding in shared
                        if finding.component.lower() in allowed
                    ][: self.limit]
                    results[index] = MinerResult(
                        shard_id=spec.shard_id,
                        findings=local,
                        usage=UsageStats(tool_calls=1),
                        notes=f"partitioned {len(local)} findings from one shared {spec.modality} scan",
                    )

            for index, spec in unfiltered_members:
                results[index] = self.run(spec)

        return [results[index] for index in range(len(specs))]


def _scan_key(spec: ShardSpec) -> tuple:
    return (
        spec.modality,
        tuple(spec.paths),
        spec.query_time,
        spec.start_time,
        spec.end_time,
    )
