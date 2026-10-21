"""Turn policy for one respondent answer.

The planner can require a sentence. This module decides which sentences those
are, keeps the recent conversation, and adds a missing sentence after the
model speaks. It does not call a model and it does not replace the reply
with a document excerpt.
"""

from __future__ import annotations

import re

from synthetic_research.retrieve import tokenize

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DIGIT = re.compile(r"\d[\d,]*(?:\.\d+)?")
_NUMBER_WORDS = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|eighteen|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
    r"(?:[-\s](?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand))*\b",
    re.IGNORECASE,
)
_UNIT = re.compile(r"\b(?:rupees?|cups?|months?|days?|weeks?|percent|pack)\b", re.IGNORECASE)
_MONEY = re.compile(r"\b(?:€|euros?|billion|million|rupees?)\b", re.IGNORECASE)
_PERCENT = re.compile(r"\d[\d,]*(?:\.\d+)?%")
_CASE_MATERIAL = re.compile(r"classroom use only|synthetic case", re.IGNORECASE)
_YEAR_KEY = re.compile(r"^\d{4}$")
_SCREENER = re.compile(
    r"out of scope|hard out|member count|gold slice|tea-only|do not interview|"
    r"metro office drinkers|ready-to-drink coffee in the last|incidence is higher",
    re.IGNORECASE,
)

THIN_WORD_LIMIT = 45
SHORT_SPAN_WORDS = 60
LONG_SPAN_CHARS = 350
FIGURE_RESTORE_LIMIT = 3
_INTERPERSONAL = re.compile(
    r"\bwho\b|\bwhom\b|who else|with you|in that moment|anyone else|anybody else",
    re.IGNORECASE,
)
_PRESENCE = re.compile(
    r"\bflatmate\b|\basleep\b|\balone\b|\bmother\b|\bfather\b|\bpartner\b|\bhousehold\b",
    re.IGNORECASE,
)

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


def _clauses(text: str) -> list[str]:
    """Sentences and standalone lines from a document span."""
    clauses: list[str] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = _SENTENCE_SPLIT.split(line) if re.search(r"[.!?]", line) else [line]
        for part in parts:
            piece = part.strip()
            if piece:
                clauses.append(piece)
    if not clauses and (text or "").strip():
        return [(text or "").strip()]
    return clauses


def _sentences(text: str) -> list[str]:
    return _clauses(text)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _figure_keys(text: str) -> set[str]:
    """Prices, counts, and spelled quantities. A bare single digit is not a figure."""
    keys: set[str] = set()
    for match in _DIGIT.finditer(text or ""):
        raw = match.group(0).replace(",", "")
        if len(raw.replace(".", "")) >= 2:
            keys.add(raw)
            continue
        window = text[max(0, match.start() - 16) : match.end() + 16]
        if _UNIT.search(window):
            keys.add(raw)
    for match in _NUMBER_WORDS.finditer(text or ""):
        keys.add(_normalize(match.group(0)))
    return keys


def _restorable_figure_keys(text: str) -> set[str]:
    """Figures worth restoring. A four-digit year alone is not enough."""
    keys: set[str] = set()
    for key in _figure_keys(text):
        if _YEAR_KEY.match(key.replace(",", "")):
            continue
        keys.add(key)
    return keys


def _qualifying_figure_gap(sentence: str, present: set[str]) -> set[str]:
    """Missing count, money, or percent figures from a sentence."""
    missing = _restorable_figure_keys(sentence) - present
    if not missing:
        return set()
    if _MONEY.search(sentence) or _PERCENT.search(sentence) or _UNIT.search(sentence):
        return missing
    for key in missing:
        digits = key.replace(".", "").replace(",", "")
        if digits.isdigit() and len(digits) >= 3:
            return missing
    return set()


def _question_overlap(question: str, sentence: str) -> int:
    """Shared content words between the question and a candidate sentence."""
    question_tokens = {token for token in tokenize(question) if len(token) > 2 and not token.isdigit()}
    if not question_tokens:
        return 0
    sentence_tokens = {token for token in tokenize(sentence) if len(token) > 2 and not token.isdigit()}
    return len(question_tokens & sentence_tokens)


def _is_case_material(sentence: str) -> bool:
    return bool(_CASE_MATERIAL.search(sentence))


def _span_word_count(text: str) -> int:
    return len(str(text or "").split())


def _is_long_span(text: str) -> bool:
    body = str(text or "")
    return len(body) > LONG_SPAN_CHARS or _span_word_count(body) > SHORT_SPAN_WORDS


def interpersonal_question(question: str) -> bool:
    return bool(_INTERPERSONAL.search(question or ""))


def _mentions_presence(text: str) -> bool:
    return bool(_PRESENCE.search(text or ""))


def _merge_evidence_lookup(
    model_evidence: list[dict],
    corpus_evidence: list[dict] | None,
) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for item in (corpus_evidence or []) + list(model_evidence):
        document_id = str(item.get("id") or "")
        if not document_id:
            continue
        prev = by_id.get(document_id)
        prev_len = len(str(prev.get("text") or "")) if prev else 0
        new_len = len(str(item.get("text") or ""))
        if prev is None or new_len > prev_len:
            by_id[document_id] = item
    return by_id


def _covered(reply_norm: str, sentence: str) -> bool:
    norm = _normalize(sentence)
    if norm and norm in reply_norm:
        return True
    tokens = [token for token in tokenize(sentence) if len(token) > 2]
    if not tokens:
        return False
    reply_tokens = set(tokenize(reply_norm))
    hits = sum(1 for token in tokens if token in reply_tokens)
    return hits / len(tokens) >= 0.75


def restore_cited_specifics(
    reply: str,
    evidence_ids: list[str],
    model_evidence: list[dict],
    *,
    question: str = "",
    corpus_evidence: list[dict] | None = None,
    allowed: set[str] | None = None,
    retrieval_ids: set[str] | None = None,
) -> tuple[str, list[str]]:
    """Keep the model's answer. Restore missing figures the question calls for, not whole files."""
    by_id = _merge_evidence_lookup(model_evidence, corpus_evidence)
    kept_ids = [item for item in evidence_ids if item in allowed] if allowed else list(evidence_ids)
    cited_text = " ".join(str(by_id.get(document_id, {}).get("text") or "") for document_id in evidence_ids)
    cited_norm = _normalize(cited_text)
    kept: list[str] = []
    for sentence in _clauses(reply):
        if _SCREENER.search(sentence) and _normalize(sentence) not in cited_norm:
            continue
        if _is_case_material(sentence) and _normalize(sentence) not in cited_norm:
            continue
        kept.append(sentence)
    text = " ".join(kept).strip()
    reply_norm = _normalize(text)
    present = _figure_keys(text)

    figure_candidates: list[tuple[int, str]] = []
    for document_id in evidence_ids:
        item = by_id.get(document_id) or {}
        if str(item.get("kind") or "") == "screener":
            continue
        body = str(item.get("text") or "")
        for sentence in _clauses(body):
            if _SCREENER.search(sentence) or _is_case_material(sentence):
                continue
            if _normalize(sentence) in reply_norm:
                continue
            if not _qualifying_figure_gap(sentence, present):
                continue
            overlap = _question_overlap(question, sentence)
            if overlap < 1:
                continue
            figure_candidates.append((overlap, sentence))
    figure_candidates.sort(key=lambda row: (-row[0], row[1]))
    for _overlap, sentence in figure_candidates[:FIGURE_RESTORE_LIMIT]:
        norm = _normalize(sentence)
        if not norm or norm in reply_norm:
            continue
        text = f"{text} {sentence}".strip()
        reply_norm = f"{reply_norm} {norm}".strip()
        present |= _figure_keys(sentence)

    if _span_word_count(text) < THIN_WORD_LIMIT and interpersonal_question(question):
        for document_id in evidence_ids:
            item = by_id.get(document_id) or {}
            body = str(item.get("text") or "")
            if _is_long_span(body):
                continue
            for sentence in _clauses(body):
                if not _mentions_presence(sentence) or _covered(reply_norm, sentence):
                    continue
                norm = _normalize(sentence)
                if not norm or norm in reply_norm:
                    continue
                text = f"{text} {sentence}".strip()
                reply_norm = f"{reply_norm} {norm}".strip()
                if document_id not in kept_ids:
                    kept_ids.append(document_id)
                break
            if _mentions_presence(text):
                break

    if _span_word_count(text) < THIN_WORD_LIMIT:
        for document_id in evidence_ids:
            item = by_id.get(document_id) or {}
            if str(item.get("kind") or "") == "screener":
                continue
            body = str(item.get("text") or "")
            if _is_long_span(body):
                continue
            for sentence in _clauses(body):
                if _SCREENER.search(sentence) or _is_case_material(sentence) or _covered(reply_norm, sentence):
                    continue
                norm = _normalize(sentence)
                if not norm or norm in reply_norm:
                    continue
                text = f"{text} {sentence}".strip()
                reply_norm = f"{reply_norm} {norm}".strip()
                present |= _figure_keys(sentence)
                if _span_word_count(text) >= THIN_WORD_LIMIT:
                    break
            if _span_word_count(text) >= THIN_WORD_LIMIT:
                break
    text = " ".join(text.split())

    if interpersonal_question(question) and not _mentions_presence(text):
        allowed_ids = set(retrieval_ids or set()) | set(allowed or set()) | set(by_id)
        candidates: list[tuple[int, str, str]] = []
        corpus = sorted(
            (corpus_evidence or list(by_id.values())),
            key=lambda row: (-float(row.get("score") or 0), str(row.get("id") or "")),
        )
        for item in corpus:
            document_id = str(item.get("id") or "")
            if not document_id or document_id not in allowed_ids:
                continue
            if str(item.get("kind") or "") not in {"transcript", "review"}:
                continue
            for sentence in _clauses(str(item.get("text") or "")):
                if _SCREENER.search(sentence) or _is_case_material(sentence):
                    continue
                if not _mentions_presence(sentence):
                    continue
                if _normalize(sentence) in reply_norm:
                    continue
                overlap = _question_overlap(question, sentence)
                if overlap < 1 and not _mentions_presence(sentence):
                    continue
                candidates.append((overlap + (2 if item.get("kind") == "transcript" else 0), sentence, document_id))
        candidates.sort(key=lambda row: (-row[0], row[1]))
        if candidates:
            _score, sentence, document_id = candidates[0]
            norm = _normalize(sentence)
            if norm and norm not in reply_norm:
                text = f"{text} {sentence}".strip()
                reply_norm = f"{reply_norm} {norm}".strip()
                if document_id not in kept_ids:
                    kept_ids.append(document_id)

    return " ".join(text.split()), kept_ids
