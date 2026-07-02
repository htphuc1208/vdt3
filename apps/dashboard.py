"""Streamlit dashboard — watch the multi-agent team resolve a telecom incident.

Run with:  streamlit run apps/dashboard.py
"""
from __future__ import annotations

import os
import sys

# make the repo root importable no matter where streamlit is launched from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402  (bundled with streamlit)
import streamlit as st  # noqa: E402

from telco_mas.environment.scenarios import get_scenario, list_scenario_ids  # noqa: E402
from telco_mas.llm import LLMClient, LLMError  # noqa: E402
from telco_mas.pipeline import prepare  # noqa: E402
from telco_mas.agents.orchestrator import MultiAgentOrchestrator  # noqa: E402
from telco_mas.baseline import run_single_agent  # noqa: E402

st.set_page_config(page_title="TelcoMAS — Multi-Agent Incident Handling", layout="wide")
st.title("📡 TelcoMAS — Multi-Agent System for Telecom Incident Handling")
st.caption("Detection → correlation → domain-expert diagnosis → consensus → remediation → validation")

settings_client = LLMClient()
settings = settings_client.settings

with st.sidebar:
    st.header("Configuration")
    st.write(f"**Model:** `{settings.model}`")
    st.write(f"**Provider:** {settings.provider_label}")
    if not settings.has_api_key:
        st.error("No `OPENAI_API_KEY` set. Add it to `.env` (OpenAI) or point "
                 "`OPENAI_BASE_URL` at DeepSeek. See `.env.example`.")
    scenario_id = st.selectbox("Incident scenario", list_scenario_ids(),
                               format_func=lambda s: f"{s} — {get_scenario(s).title}")
    mode = st.radio("System", ["Multi-Agent (TelcoMAS)", "Single-Agent baseline"], index=0)
    run_clicked = st.button("▶ Run analysis", type="primary", disabled=not settings.has_api_key)


scenario = get_scenario(scenario_id)
ctx, incident, _ = prepare(scenario)

left, right = st.columns([3, 2])
with left:
    st.subheader(f"Incident: {incident.title}")
    st.write(incident.description)
    st.markdown("**Active alarms**")
    st.dataframe(
        pd.DataFrame([
            {"element": a.element_id, "severity": a.severity.value, "alarm": a.name, "cause": a.probable_cause}
            for a in incident.alarms
        ]),
        use_container_width=True, hide_index=True,
    )
with right:
    st.markdown("**Network topology**")
    st.dataframe(
        pd.DataFrame([
            {"id": e.id, "type": e.type.value, "domain": e.domain.value, "parent": e.parent_id or "-"}
            for e in ctx.sim.topology.all()
        ]),
        use_container_width=True, hide_index=True, height=300,
    )
    with st.expander("Ground truth (for grading)"):
        st.write(f"**Root-cause element:** `{scenario.element_id}` · **fault:** `{scenario.fault_type}` "
                 f"· **domain:** {scenario.domain.value}")


if run_clicked:
    llm = LLMClient()
    steps: list[str] = []
    with st.status("Running the agent team…", expanded=True) as status:
        def progress(msg: str) -> None:
            steps.append(msg)
            status.update(label=msg)
            st.write(f"• {msg}")

        try:
            if mode.startswith("Multi"):
                result = MultiAgentOrchestrator(llm, ctx).run(incident, progress=progress)
            else:
                result = run_single_agent(incident, ctx, llm, progress=progress)
            status.update(label="Analysis complete", state="complete")
        except LLMError as exc:
            status.update(label="LLM error", state="error")
            st.error(str(exc))
            st.stop()

    st.session_state["result"] = result

result = st.session_state.get("result")
if result is not None and result.incident.id == incident.id:
    st.divider()
    c = result.consensus
    correct = bool(c and c.faulty_element_id == scenario.element_id)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Localization", "✅ correct" if correct else "❌ wrong")
    m2.metric("Confidence", f"{(c.confidence if c else 0):.0%}")
    m3.metric("Resolved", "✅ yes" if (result.validation and result.validation.resolved) else "❌ no")
    m4.metric("Tokens", f"{result.usage.total_tokens}")

    if result.triage:
        st.info(f"**Triage:** {result.triage.severity.value} · suspected {result.triage.suspected_domain.value} — {result.triage.summary}")

    if result.hypotheses:
        st.subheader("Domain-expert hypotheses")
        st.dataframe(
            pd.DataFrame([
                {"expert": h.proposed_by, "faulty_element": h.faulty_element_id, "fault_type": h.fault_type,
                 "confidence": round(h.confidence, 2), "root_cause": h.root_cause}
                for h in result.hypotheses
            ]),
            use_container_width=True, hide_index=True,
        )

    if c:
        st.subheader("Consensus (weighted vote + arbiter)")
        cc1, cc2 = st.columns([2, 3])
        with cc1:
            if c.vote_breakdown:
                st.bar_chart(pd.DataFrame({"vote score": c.vote_breakdown}))
        with cc2:
            st.success(f"**Root cause:** {c.root_cause}\n\n**Element:** `{c.faulty_element_id}` "
                       f"· **type:** `{c.fault_type}` · **confidence:** {c.confidence:.0%}")
            st.caption(c.explanation)

    if result.remediation:
        st.subheader("Remediation plan")
        st.write(f"**SOP:** `{result.remediation.sop_id}` — {result.remediation.summary}")
        for i, s in enumerate(result.remediation.steps, 1):
            st.write(f"{i}. {s}")

    if result.validation:
        if result.validation.resolved:
            st.success(f"**Validation:** incident resolved. {result.validation.notes}")
        else:
            st.error(f"**Validation:** not resolved. {result.validation.notes}")

    with st.expander("Full agent trace"):
        for step in result.trace:
            if step.tool_calls:
                for tc in step.tool_calls:
                    args = ", ".join(f"{k}={v}" for k, v in tc.arguments.items())
                    st.markdown(f"**{step.agent}** → `{tc.name}({args})`")
                    st.caption(tc.result_preview)
            elif step.content.strip():
                st.markdown(f"**{step.agent}**: {step.content.strip()[:400]}")
