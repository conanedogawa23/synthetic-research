"""Citation lock for a respondent reply.

The model may only name documents that were sent with the question. Graph
rows are retrieval hints, not citable evidence. This module does not call a
model; the engine asks DeepInfra whether the transcripts were enough and
whether the reply follows from the passages.
"""

from __future__ import annotations

import re

_DOC_ID = re.compile(r"\b(?:UP-\d+-c\d+|WEB-\d+|GRAPH-\d+|VEC-\d+|[TRS]-\d+|SL-[A-Z0-9]+|ST-\d+)\b")
_EMPTY_BRACKETS = re.compile(r"\[\s*\]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def lock_citations(reply: str, evidence_ids: list[str], allowed: set[str]) -> tuple[str, list[str], list[str]]:
    """Drop ids that were not supplied. Return the reply, the kept ids, and the removed ids."""
    citable = {document_id for document_id in allowed if not str(document_id).startswith("GRAPH-")}
    removed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        document_id = match.group(0)
        if document_id in citable:
            return document_id
        if document_id not in removed:
            removed.append(document_id)
        return ""

    cleaned = _DOC_ID.sub(replace, reply or "")
    cleaned = _EMPTY_BRACKETS.sub("", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
    cleaned = _MULTI_SPACE.sub(" ", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned).strip()
    kept: list[str] = []
    for document_id in list(evidence_ids) + _DOC_ID.findall(cleaned):
        if document_id in citable and document_id not in kept:
            kept.append(document_id)
    return cleaned, kept, removed


def grounding_confidence(*, supported: bool, removed: list[str], used_web: bool) -> dict:
    """Confidence is 1 only when the reply survived the citation check unchanged."""
    if supported and not removed:
        basis = "supported by the cited transcripts"
        if used_web:
            basis = "supported by the cited documents"
        return {"score": 1.0, "label": "high", "basis": basis}
    reason = "citations were corrected to the supplied documents"
    if not supported:
        reason = "the reply was rewritten to what the documents support"
    return {"score": 0.45, "label": "medium", "basis": reason}
