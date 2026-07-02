"""A tiny dependency-free TF-IDF retriever over SOPs and historical incidents.

Deliberately embedding-free: no external API and no heavy ML dependency, so the
knowledge base works fully offline and installs in seconds.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from .kb_data import HISTORICAL_INCIDENTS, SOP_LIBRARY

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class Document:
    id: str
    kind: str  # "sop" | "incident"
    text: str
    meta: dict = field(default_factory=dict)


class TfidfRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self._tf: list[Counter] = []
        self._idf: dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        df: Counter = Counter()
        for doc in self.documents:
            counts = Counter(_tokenize(doc.text))
            self._tf.append(counts)
            for term in counts:
                df[term] += 1
        n = max(len(self.documents), 1)
        self._idf = {term: math.log((1 + n) / (1 + freq)) + 1.0 for term, freq in df.items()}

    def _vector(self, counts: Counter) -> dict[str, float]:
        return {t: c * self._idf.get(t, 0.0) for t, c in counts.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return num / (na * nb) if na and nb else 0.0

    def search(self, query: str, kind: str | None = None, top_k: int = 3) -> list[tuple[Document, float]]:
        q_vec = self._vector(Counter(_tokenize(query)))
        scored: list[tuple[Document, float]] = []
        for doc, counts in zip(self.documents, self._tf):
            if kind and doc.kind != kind:
                continue
            scored.append((doc, self._cosine(q_vec, self._vector(counts))))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def _sop_document(sop: dict) -> Document:
    text = f"{sop['title']}. Domain {sop['domain']}. Symptoms: {sop['symptoms']} Steps: " + " ".join(sop["steps"])
    return Document(id=sop["id"], kind="sop", text=text, meta=sop)


def _incident_document(inc: dict) -> Document:
    text = f"{inc['symptoms']} Root cause: {inc['root_cause']} Resolution: {inc['resolution']}"
    return Document(id=inc["id"], kind="incident", text=text, meta=inc)


def build_default_retriever() -> TfidfRetriever:
    docs = [_sop_document(s) for s in SOP_LIBRARY] + [_incident_document(i) for i in HISTORICAL_INCIDENTS]
    return TfidfRetriever(docs)
