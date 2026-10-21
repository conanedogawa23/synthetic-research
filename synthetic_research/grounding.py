"""Citation lock for a respondent reply.

The model may only name documents that were sent with the question. Graph
rows are retrieval hints, not citable evidence. This module does not call a
model. Invented document ids are removed from the reply text.
"""

from __future__ import annotations

import re

_DOC_ID = re.compile(r"\b(?:UP-\d+-c\d+|WEB-\d+|GRAPH-\d+|VEC-\d+|[TRS]-\d+|SL-[A-Z0-9]+|ST-\d+)\b")
_EMPTY_BRACKETS = re.compile(r"\[\s*\]|\(\s*\)")
_LEADING_LIST_MARK = re.compile(r"(\[|\()\s*,+\s*")
_TRAILING_LIST_MARK = re.compile(r"\s*,+\s*(\]|\))")
_REPEATED_COMMA = re.compile(r"(?:,\s*){2,}")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def passage_status(text: str) -> str:
    """Status the passage gives itself. No file or campaign is named here."""
    lowered = (text or "").lower()
    notes = []
    if "classroom" in lowered or "fictional" in lowered:
        notes.append("this passage says the material is fictional or for classroom use")
    if "industry" in lowered and any(word in lowered for word in ("typical", "generic", "leaders usually", "in this industry")):
        notes.append("this passage presents industry practice rather than this organisation's own figures")
    return " ".join(notes)


def visible_evidence_ids(reply: str, evidence_ids: list[str]) -> list[str]:
    """Chips follow the reply. Graph rows never appear. Web appears when the reply names it, or when no transcript was cited."""
    mentioned = set(_DOC_ID.findall(reply or ""))
    has_transcript = any(
        document_id and not str(document_id).startswith(("GRAPH-", "WEB-", "VEC-"))
        for document_id in evidence_ids or []
    )
    kept: list[str] = []
    for document_id in evidence_ids or []:
        if not document_id or str(document_id).startswith("GRAPH-") or document_id in kept:
            continue
        outside = str(document_id).startswith(("WEB-", "VEC-"))
        if outside and has_transcript and document_id not in mentioned:
            continue
        kept.append(document_id)
    return kept


_CITATION_GROUP = re.compile(
    r"[ \t]*[\[(][ \t]*(?:(?:"
    + _DOC_ID.pattern
    + r")[ \t]*,[ \t]*)*(?:"
    + _DOC_ID.pattern
    + r")[ \t]*[\])]"
)


def _lift_inline_citations(reply: str) -> tuple[str, list[str]]:
    """The ids belong on the chips, not in the middle of a sentence."""
    found: list[str] = []

    def replace(match: re.Match[str]) -> str:
        for document_id in _DOC_ID.findall(match.group(0)):
            if document_id not in found:
                found.append(document_id)
        return ""

    return _CITATION_GROUP.sub(replace, reply or ""), found


def lock_citations(reply: str, evidence_ids: list[str], allowed: set[str]) -> tuple[str, list[str], list[str]]:
    """Drop ids that were not supplied. Return the reply, the kept ids, and the removed ids."""
    citable = {document_id for document_id in allowed if not str(document_id).startswith("GRAPH-")}
    stripped, lifted = _lift_inline_citations(reply or "")
    removed = [document_id for document_id in lifted if document_id not in citable]

    def replace(match: re.Match[str]) -> str:
        document_id = match.group(0)
        if document_id in citable:
            return document_id
        if document_id not in removed:
            removed.append(document_id)
        return ""

    cleaned = _tidy_citation_text(_DOC_ID.sub(replace, stripped))
    seed = [item for item in [*lifted, *evidence_ids] if item in citable]
    kept = visible_evidence_ids(cleaned, seed)
    for document_id in _DOC_ID.findall(cleaned):
        if document_id in citable and document_id not in kept:
            kept.append(document_id)
    return cleaned, kept, removed


def _tidy_citation_text(text: str) -> str:
    cleaned = _LEADING_LIST_MARK.sub(r"\1", text or "")
    cleaned = _TRAILING_LIST_MARK.sub(r"\1", cleaned)
    cleaned = _REPEATED_COMMA.sub(", ", cleaned)
    cleaned = _EMPTY_BRACKETS.sub("", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned)
    cleaned = _MULTI_SPACE.sub(" ", cleaned)
    return cleaned.strip()


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
