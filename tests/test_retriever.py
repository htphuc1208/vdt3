"""The TF-IDF retriever surfaces the right SOP / historical incident."""
from telco_mas.knowledge.retriever import build_default_retriever


def test_retriever_ranks_correct_sop():
    r = build_default_retriever()
    hits = r.search("optical loss of signal fiber link down unreachable", kind="sop", top_k=3)
    assert "SOP-TRANSPORT-FIBER" in [d.id for d, _ in hits]


def test_retriever_finds_similar_incident():
    r = build_default_retriever()
    hits = r.search("dns servfail pdu session setup failing radio healthy", kind="incident", top_k=3)
    assert any(d.meta["fault_type"] == "DNS_FAILURE" for d, _ in hits)
