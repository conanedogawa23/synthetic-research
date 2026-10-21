"""Turn policy for one respondent answer.

The planner can require a sentence. This module decides which sentences those
are, keeps the recent conversation, and adds a missing sentence after the
model speaks. It does not call a model and it does not replace the reply
with a document excerpt.
"""

from __future__ import annotations

from synthetic_research.retrieve import tokenize

CHAT_TURN_LIMIT = 2
LEDGER_LIMIT = 12
MODEL_EVIDENCE_EXTRA = 8


def _cited_commitments(turns: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for turn in turns or []:
        if turn.get("abstain"):
            continue
        question = str(turn.get("question") or "").strip()
        reply = str(turn.get("reply") or "").strip()
        if not question or not reply:
            continue
        evidence_ids = [str(item) for item in turn.get("evidenceIds") or [] if item]
        rows.append({"question": question, "reply": reply, "evidenceIds": evidence_ids})
    return rows


def prior_turns(turns: list[dict]) -> list[dict]:
    """The last two cited replies for chat messages, question and reply text only."""
    return [
        {"question": row["question"], "reply": row["reply"]}
        for row in _cited_commitments(turns)[-CHAT_TURN_LIMIT:]
    ]


def prior_commitments(question: str, turns: list[dict]) -> list[dict]:
    """Earlier cited turns the persona must stay consistent with, up to twelve lines."""
    cited = _cited_commitments(turns)
    if not cited:
        return []
    recent = cited[-CHAT_TURN_LIMIT:]
    older = cited[: -CHAT_TURN_LIMIT] if len(cited) > CHAT_TURN_LIMIT else []
    chosen: list[dict] = [dict(row) for row in recent]
    seen = {row["question"] for row in chosen}
    question_tokens = set(tokenize(question))
    if question_tokens and older:
        scored: list[tuple[int, dict]] = []
        for row in older:
            hay = set(tokenize(f"{row['question']} {row['reply']}"))
            overlap = len(question_tokens & hay)
            if overlap:
                scored.append((overlap, row))
        scored.sort(key=lambda item: (-item[0], item[1]["question"]))
        for _overlap, row in scored:
            if len(chosen) >= LEDGER_LIMIT:
                break
            if row["question"] in seen:
                continue
            chosen.append(dict(row))
            seen.add(row["question"])
    return chosen[:LEDGER_LIMIT]


def planner_document_ids(plan: dict) -> set[str]:
    """Documents the planner treats as required for this answer."""
    ids: set[str] = set()
    direct_id = plan.get("directDocumentId")
    if direct_id:
        ids.add(str(direct_id))
    for document_id in plan.get("evidenceIds") or []:
        if document_id:
            ids.add(str(document_id))
    for document_id in plan.get("fallbackEvidenceIds") or []:
        if document_id:
            ids.add(str(document_id))
    for conflict in plan.get("conflicts") or []:
        for side in conflict.get("sides") or []:
            excerpt_id = side.get("excerptDocumentId")
            if excerpt_id:
                ids.add(str(excerpt_id))
            for document_id in side.get("documentIds") or []:
                if document_id:
                    ids.add(str(document_id))
    return ids


def bounded_model_evidence(evidence: list[dict], plan: dict, *, extra_limit: int = MODEL_EVIDENCE_EXTRA) -> list[dict]:
    """Required planner documents plus the top remaining spans for the model prompt."""
    required = planner_document_ids(plan)
    by_id: dict[str, dict] = {}
    for item in evidence:
        document_id = item.get("id")
        if document_id and document_id not in by_id:
            by_id[str(document_id)] = item
    chosen: list[dict] = []
    seen: set[str] = set()
    for document_id in required:
        row = by_id.get(document_id)
        if row and document_id not in seen:
            chosen.append(row)
            seen.add(document_id)
    remaining = [item for item in evidence if str(item.get("id") or "") not in seen]
    remaining.sort(key=lambda row: (-float(row.get("score") or 0), str(row.get("id") or "")))
    added = 0
    for item in remaining:
        if added >= extra_limit:
            break
        document_id = str(item.get("id") or "")
        if not document_id or document_id in seen:
            continue
        chosen.append(item)
        seen.add(document_id)
        added += 1
    return chosen


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
