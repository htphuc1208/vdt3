"""Knowledge-base tools: SOP retrieval and historical-incident lookup (RAG)."""
from __future__ import annotations

import json
from typing import Any


def search_knowledge_base(ctx: Any, query: str, top_k: int = 3) -> str:
    hits = ctx.retriever.search(query, kind="sop", top_k=top_k)
    return json.dumps(
        [
            {
                "sop_id": doc.id,
                "title": doc.meta["title"],
                "domain": doc.meta["domain"],
                "symptoms": doc.meta["symptoms"],
                "steps": doc.meta["steps"],
                "score": round(score, 3),
            }
            for doc, score in hits
        ]
    )


def get_historical_incidents(ctx: Any, query: str, top_k: int = 3) -> str:
    hits = ctx.retriever.search(query, kind="incident", top_k=top_k)
    return json.dumps(
        [
            {
                "incident_id": doc.id,
                "symptoms": doc.meta["symptoms"],
                "root_cause": doc.meta["root_cause"],
                "resolution": doc.meta["resolution"],
                "score": round(score, 3),
            }
            for doc, score in hits
        ]
    )
