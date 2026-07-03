"""Command-line runner for TelcoMAS.

Examples:
    python -m apps.cli --list
    python -m apps.cli --scenario fiber_cut
    python -m apps.cli --scenario dns_failure --mode single --trace
"""
from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from telco_mas.environment.scenarios import get_scenario, list_scenario_ids
from telco_mas.llm import LLMClient, LLMError
from telco_mas.pipeline import run
from telco_mas.schemas import PipelineResult

console = Console()


def _print_trace(result: PipelineResult) -> None:
    console.rule("[bold]Agent trace")
    for step in result.trace:
        if step.tool_calls:
            for tc in step.tool_calls:
                args = ", ".join(f"{k}={v}" for k, v in tc.arguments.items())
                console.print(f"[cyan]{step.agent}[/cyan] → [yellow]{tc.name}[/yellow]({args})")
                console.print(f"    [dim]{tc.result_preview}[/dim]")
        elif step.content.strip():
            preview = step.content.strip().replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:199] + "…"
            console.print(f"[cyan]{step.agent}[/cyan]: [dim]{preview}[/dim]")


def _print_result(result: PipelineResult, scenario_id: str | None) -> None:
    console.rule("[bold green]Incident resolution")

    if result.triage:
        console.print(
            f"[bold]Triage[/bold]: severity=[red]{result.triage.severity.value}[/red], "
            f"domain={result.triage.suspected_domain.value} — {result.triage.summary}"
        )

    if result.hypotheses:
        table = Table(title="Domain-expert hypotheses")
        table.add_column("Expert")
        table.add_column("Faulty element")
        table.add_column("Fault type")
        table.add_column("Conf", justify="right")
        for h in result.hypotheses:
            table.add_row(h.proposed_by, h.faulty_element_id or "?", h.fault_type or "?", f"{h.confidence:.2f}")
        console.print(table)

    c = result.consensus
    if c:
        votes = ", ".join(f"{k}:{v}" for k, v in sorted(c.vote_breakdown.items(), key=lambda x: -x[1]))
        body = (
            f"[bold]Root cause:[/bold] {c.root_cause}\n"
            f"[bold]Faulty element:[/bold] {c.faulty_element_id}  "
            f"[bold]Type:[/bold] {c.fault_type}  [bold]Confidence:[/bold] {c.confidence:.2f}\n"
            f"[bold]Votes:[/bold] {votes or 'n/a'}\n"
            f"[dim]{c.explanation}[/dim]"
        )
        console.print(Panel(body, title="Consensus", border_style="green"))

    if result.remediation:
        steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(result.remediation.steps))
        console.print(Panel(
            f"[bold]SOP:[/bold] {result.remediation.sop_id}\n{result.remediation.summary}\n{steps}",
            title="Remediation plan", border_style="blue",
        ))

    if result.validation:
        color = "green" if result.validation.resolved else "red"
        status = "RESOLVED" if result.validation.resolved else "NOT RESOLVED"
        console.print(f"[bold {color}]Validation: {status}[/bold {color}] — {result.validation.notes}")

    u = result.usage
    console.print(
        f"\n[dim]Usage: {u.total_tokens} tokens, {u.llm_calls} LLM calls, "
        f"{u.tool_calls} tool calls, {result.latency_s}s[/dim]"
    )

    if scenario_id:
        sc = get_scenario(scenario_id)
        ok = c and c.faulty_element_id == sc.element_id
        mark = "[green]correct[/green]" if ok else f"[red]wrong[/red] (truth: {sc.element_id})"
        console.print(f"[bold]Ground truth check:[/bold] localization {mark}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a telecom incident through TelcoMAS")
    parser.add_argument("--scenario", help="scenario id (see --list)")
    parser.add_argument(
        "--mode",
        choices=["multi", "single", "no_rag", "no_consensus", "no_arbiter", "no_partition", "no_debate"],
        default="multi",
    )
    parser.add_argument("--list", action="store_true", help="list available scenarios")
    parser.add_argument("--trace", action="store_true", help="print the full agent trace")
    args = parser.parse_args(argv)

    if args.list or not args.scenario:
        console.print("[bold]Available scenarios:[/bold]")
        for sid in list_scenario_ids():
            console.print(f"  - {sid}: {get_scenario(sid).title}")
        if not args.scenario:
            return 0

    llm = LLMClient()
    if not llm.settings.has_api_key:
        console.print("[red]No OPENAI_API_KEY set.[/red] Set it in your environment or .env "
                      "(OpenAI), or point OPENAI_BASE_URL at DeepSeek. See .env.example.")
        return 2

    console.print(f"[bold]Scenario:[/bold] {args.scenario}  [bold]Mode:[/bold] {args.mode}  "
                  f"[bold]Model:[/bold] {llm.settings.model} ({llm.settings.provider_label})")
    try:
        result = run(args.scenario, mode=args.mode, llm=llm,
                     progress=lambda m: console.print(f"[dim]· {m}[/dim]"))
    except LLMError as exc:
        console.print(f"[red]LLM error:[/red] {exc}")
        return 1

    if args.trace:
        _print_trace(result)
    _print_result(result, args.scenario)
    return 0


if __name__ == "__main__":
    sys.exit(main())
