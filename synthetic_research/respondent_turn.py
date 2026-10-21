"""Turn policy for one respondent answer.

The planner can require a sentence. This module decides which sentences those
are, keeps the recent conversation, and adds a missing sentence after the
model speaks. It does not call a model and it does not replace the reply
with a document excerpt.
"""

from __future__ import annotations

PRIOR_TURN_LIMIT = 4


def prior_turns(turns: list[dict]) -> list[dict]:
    """The last cited replies, question and reply text only."""
    chosen: list[dict] = []
    for turn in turns or []:
        if turn.get("abstain"):
            continue
        question = str(turn.get("question") or "").strip()
        reply = str(turn.get("reply") or "").strip()
        if not question or not reply:
            continue
        chosen.append({"question": question, "reply": reply})
    return chosen[-PRIOR_TURN_LIMIT:]


def required_statements(plan: dict) -> list[dict]:
    """Sentences a direct answer or a conflict must contain, with their documents."""
    mode = plan.get("mode")
    if mode == "direct":
        text = str(plan.get("reply") or "").strip()
        document_id = str(plan.get("directDocumentId") or "").strip()
        if text and document_id:
            return [{"text": text, "documentIds": [document_id]}]
        return []
    if mode != "conflict":
        return []
    statements: list[dict] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for conflict in plan.get("conflicts") or []:
        for side in conflict.get("sides") or []:
            text = str(side.get("excerpt") or "").strip()
            document_ids: list[str] = []
            for document_id in [side.get("excerptDocumentId"), *(side.get("documentIds") or [])]:
                if document_id and document_id not in document_ids:
                    document_ids.append(str(document_id))
            key = (text, tuple(document_ids))
            if key in seen or (not text and not document_ids):
                continue
            seen.add(key)
            statements.append({"text": text, "documentIds": document_ids})
    if statements:
        return statements
    text = str(plan.get("reply") or "").strip()
    document_ids = [str(item) for item in plan.get("evidenceIds") or [] if item]
    if text:
        return [{"text": text, "documentIds": document_ids}]
    return []


def enforce_required_statements(
    reply: str,
    evidence_ids: list[str],
    statements: list[dict],
    allowed: set[str],
) -> tuple[str, list[str]]:
    """Keep the model's sentences. Add any required sentence or document it omitted."""
    text = " ".join((reply or "").split())
    kept = [item for item in evidence_ids if item in allowed]
    for statement in statements:
        sentence = " ".join(str(statement.get("text") or "").split())
        if sentence and sentence not in text:
            text = f"{text} {sentence}".strip()
        for document_id in statement.get("documentIds") or []:
            if document_id in allowed and document_id not in kept:
                kept.append(document_id)
    return text, kept
