"""Synthesis for one compiled respondent.

Process for a question over a cohort group:

1. Score each document by how directly one of its sentences answers the
   question. The highest score is the document that states the answer.
2. Leave indirect references for the model. A retrieved document that never
   states the answer is a reference, not an answer.
3. When transcripts and reviews disagree on spend, who brews, or whether a
   routine is allowed, keep every side. Do not collapse the group into one
   preference. Those sides are already stated, so they are quoted. Only the
   indirect case is inferred by the model.
"""

from __future__ import annotations

import re

from synthetic_research.retrieve import tokenize

VOICE_KINDS = {"transcript", "review"}
DIRECT_PHRASE = 3
DIRECT_OVERLAP = 2
DIRECT_RATIO = 0.5
RANK_LIMIT = 8

_KIND_RANK = {"transcript": 3, "review": 2, "sales": 1, "web": 1, "screener": 0}

_NEGATION = re.compile(
    r"\b(?:not|never|no|won't|wont|don't|dont|cannot|can't|cant)\b",
    re.IGNORECASE,
)


def _compiled(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern, re.IGNORECASE) for pattern in patterns)


# Each point is a disagreement that appears once many transcripts are compiled
# into one respondent. Stances are explicit. A single transcript that states
# both stances is tension inside one voice, not a group conflict.
POINTS = (
    {
        "id": "spend",
        "label": "do not want to spend",
        "triggers": ("spend", "pay", "price", "money", "rupee", "rupees"),
        "stances": (
            {
                "id": "refuse",
                "label": "do not want to spend",
                "patterns": _compiled(
                    (
                        r"\b(?:do not|don't|dont|will not|won't|wont|cannot|can't|cant)\s+pay\b",
                        r"\b(?:do not|don't|dont)\s+want to spend\b",
                        r"\bcannot justify\b",
                        r"\bcan(?:not|'t)\s+justify\b",
                        r"\bthe can loses\b",
                        r"\bor i skip\b",
                        r"\boffer price\b",
                        r"\btwice the money\b",
                    )
                ),
            },
            {
                "id": "willing",
                "label": "would spend",
                "negated": True,
                "patterns": _compiled(
                    (
                        r"\bwould pay\b",
                        r"\bwill pay\b",
                        r"\bpay more\b",
                        r"\bpay cafe money\b",
                    )
                ),
            },
        ),
    },
    {
        "id": "brewer",
        "label": "choice on the person who brews",
        "triggers": ("brew", "brews", "brewed", "brewing"),
        "stances": (
            {
                "id": "self",
                "label": "I brew it",
                "patterns": _compiled(
                    (
                        r"\bmorning drink is mine\b",
                        r"\bi brew\b",
                        r"\bi make (?:the |my )?coffee\b",
                    )
                ),
            },
            {
                "id": "household",
                "label": "someone in the house brews it",
                "patterns": _compiled(
                    (
                        r"\bmy mother\b",
                        r"\bwho finished\b",
                        r"\brelative(?:'s|s)? house\b",
                    )
                ),
            },
            {
                "id": "counter",
                "label": "the counter brews it",
                "patterns": _compiled(
                    (
                        r"\bcanteen\b",
                        r"\broadside\b",
                        r"\bkirana\b",
                        r"\bdarshini\b",
                        r"\bfilter counter\b",
                    )
                ),
            },
        ),
    },
    {
        "id": "routine",
        "label": "routine not allowed",
        "triggers": ("routine", "habit", "weekday", "daily", "sunday", "every day"),
        "stances": (
            {
                "id": "allowed",
                "label": "the routine is allowed",
                "patterns": _compiled(
                    (
                        r"\bweekday morning\b",
                        r"\boffice bag\b",
                        r"\bevery morning\b",
                        r"\bevery day\b",
                        r"\bi keep this\b",
                    )
                ),
            },
            {
                "id": "refused",
                "label": "the routine is not allowed",
                "patterns": _compiled(
                    (
                        r"\btreat is allowed on sunday\b",
                        r"\bonly on sunday\b",
                        r"\broutine (?:is )?not allowed\b",
                        r"\bnot allowed\b",
                        r"\bwill not switch\b",
                        r"\bfeels selfish\b",
                    )
                ),
            },
        ),
    },
)


def blank_synthesis(mode: str) -> dict:
    return {
        "mode": mode,
        "directDocumentId": None,
        "ranked": [],
        "conflicts": [],
    }


def public_synthesis(plan: dict) -> dict:
    return {
        "mode": plan.get("mode") or "none",
        "directDocumentId": plan.get("directDocumentId"),
        "ranked": [dict(row) for row in plan.get("ranked") or []],
        "conflicts": [_public_conflict(conflict) for conflict in plan.get("conflicts") or []],
    }


def format_synthesis(synthesis: dict | None) -> list[str]:
    if not synthesis or synthesis.get("mode") in {None, "", "none", "system"}:
        return []
    lines = []
    mode = synthesis.get("mode")
    direct_id = synthesis.get("directDocumentId")
    if mode == "direct" and direct_id:
        lines.append(f"Direct answer: {direct_id}.")
    elif mode == "conflict" and direct_id:
        lines.append(f"Direct answer: {direct_id}. The group conflicts.")
    elif mode == "conflict":
        lines.append("The group conflicts.")
    elif mode == "indirect":
        lines.append("Inference: indirect.")
    for conflict in synthesis.get("conflicts") or []:
        rendered = []
        for side in conflict.get("sides") or []:
            document_ids = ", ".join(side.get("documentIds") or [])
            rendered.append(f"{side.get('label')} [{document_ids}]")
        lines.append(f"Conflict ({conflict.get('label')}): " + "; ".join(rendered) + ".")
    return lines


def plan_synthesis(
    question: str,
    documents: list[dict],
    *,
    retrieved_ids: set[str] | None = None,
) -> dict:
    """Plan the reply. Indirect plans have an empty reply for the model to write."""
    question_tokens = tokenize(question)
    retrieved = retrieved_ids or set()
    usable = [document for document in documents if document.get("id") and isinstance(document.get("text") or "", str)]
    ranked_direct, ranked_indirect = _rank_documents(question_tokens, usable, retrieved)
    conflicts = _active_conflicts(question, question_tokens, usable)
    ranked = _merge_ranked(ranked_direct, ranked_indirect, conflicts)
    quoted_reply, quoted_ids = _compose_conflict(ranked_direct, conflicts) if conflicts else ("", [])
    stated = bool(conflicts) and _conflict_is_stated(question_tokens, conflicts, ranked_direct)
    if stated and quoted_reply and quoted_ids:
        if ranked_direct and ranked_direct[0].get("sentence"):
            _align_covered_sides(conflicts, ranked_direct[0]["id"], ranked_direct[0]["sentence"])
        draft = _plan(
            mode="conflict",
            direct_id=ranked_direct[0]["id"] if ranked_direct else None,
            reply=quoted_reply,
            evidence_ids=quoted_ids,
            ranked=ranked,
            conflicts=conflicts,
        )
        draft["evidenceIds"] = cited_ids(draft)
        return draft
    if ranked_direct:
        reply = _group_reply(ranked_direct)
        if reply:
            draft = _plan(
                mode="direct",
                direct_id=ranked_direct[0]["id"],
                reply=reply,
                evidence_ids=[],
                ranked=ranked,
                conflicts=[],
            )
            draft["evidenceIds"] = cited_ids(draft)
            return draft
    return _plan(
        mode="indirect",
        direct_id=None,
        reply="",
        evidence_ids=[],
        ranked=ranked,
        conflicts=conflicts,
        fallback_reply=quoted_reply,
        fallback_ids=quoted_ids,
    )


def cited_ids(plan: dict, extra: list[str] | None = None) -> list[str]:
    """The documents that state or conflict on the answer, most direct first.

    A near miss and a screener count are not part of that group. Several
    documents that clear the same bar are.
    """
    ordered: list[str] = []

    def add(document_id: str | None) -> None:
        if document_id and document_id not in ordered:
            ordered.append(document_id)

    directs = [
        row
        for row in plan.get("ranked") or []
        if row.get("relation") == "direct" and row.get("kind") != "screener"
    ]
    best = max((float(row.get("score") or 0) for row in directs), default=0.0)
    add(plan.get("directDocumentId"))
    for row in directs:
        score = float(row.get("score") or 0)
        if best and score < best * 0.85:
            continue
        add(row.get("id"))
    for conflict in plan.get("conflicts") or []:
        for side in conflict.get("sides") or []:
            for document_id in side.get("documentIds") or []:
                add(document_id)
    if plan.get("mode") == "indirect":
        added = 0
        for row in plan.get("ranked") or []:
            if row.get("relation") != "indirect" or row.get("kind") == "screener":
                continue
            add(row.get("id"))
            added += 1
            if added >= 4:
                break
    close = {
        row.get("id")
        for row in directs
        if not best or float(row.get("score") or 0) >= best * 0.85
    }
    for document_id in list(extra or []) + list(plan.get("evidenceIds") or []):
        if document_id in close or document_id in ordered:
            add(document_id)
    if not ordered:
        for document_id in extra or []:
            add(document_id)
    return ordered


def model_respects_plan(plan: dict, reply: str, evidence_ids: list[str]) -> bool:
    """A rewrite may tighten wording. It may not drop the direct document or a conflict side."""
    cited = set(evidence_ids or [])
    text = (reply or "").strip()
    if not text:
        return False
    if plan.get("mode") == "indirect" and plan.get("conflicts"):
        return _cites_every_side(plan, cited)
    if plan.get("mode") == "direct":
        return bool(plan.get("directDocumentId")) and plan["directDocumentId"] in cited
    if plan.get("mode") == "conflict":
        for conflict in plan.get("conflicts") or []:
            for side in conflict.get("sides") or []:
                excerpt = side.get("excerpt") or ""
                document_ids = set(side.get("documentIds") or [])
                if excerpt and excerpt in text:
                    continue
                if not excerpt and document_ids & cited:
                    continue
                return False
    return True


def fallback_plan(plan: dict) -> dict:
    """Use the quoted sides when the model drops a direct document or a conflict side."""
    if plan.get("mode") in {"direct", "conflict"} and plan.get("reply"):
        return plan
    reply = str(plan.get("fallbackReply") or "").strip()
    evidence_ids = [item for item in plan.get("fallbackEvidenceIds") or [] if item]
    if not reply or not evidence_ids:
        return plan
    restored = dict(plan)
    restored["mode"] = "conflict" if plan.get("conflicts") else "direct"
    restored["reply"] = reply
    restored["evidenceIds"] = evidence_ids
    return restored


def _plan(
    *,
    mode: str,
    direct_id: str | None,
    reply: str,
    evidence_ids: list[str],
    ranked: list[dict],
    conflicts: list[dict],
    fallback_reply: str = "",
    fallback_ids: list[str] | None = None,
) -> dict:
    return {
        "mode": mode,
        "directDocumentId": direct_id,
        "reply": reply,
        "evidenceIds": evidence_ids,
        "ranked": ranked,
        "conflicts": conflicts,
        "fallbackReply": fallback_reply,
        "fallbackEvidenceIds": list(fallback_ids or []),
    }


def _cites_every_side(plan: dict, cited: set[str]) -> bool:
    for conflict in plan.get("conflicts") or []:
        for side in conflict.get("sides") or []:
            if not set(side.get("documentIds") or []) & cited:
                return False
    return True


def _conflict_is_stated(question_tokens: list[str], conflicts: list[dict], direct_rows: list[dict]) -> bool:
    if direct_rows:
        return True
    for conflict in conflicts:
        point = next(item for item in POINTS if item["id"] == conflict["point"])
        for side in conflict.get("sides") or []:
            excerpt = side.get("excerpt") or ""
            if _sentence_score(question_tokens, excerpt) > 0:
                return True
            if _question_touches(excerpt, point["triggers"]):
                return True
    return False


def _public_conflict(conflict: dict) -> dict:
    return {
        "point": conflict.get("point"),
        "label": conflict.get("label"),
        "sides": [
            {
                "stance": side.get("stance"),
                "label": side.get("label"),
                "documentIds": list(side.get("documentIds") or []),
                "excerptDocumentId": side.get("excerptDocumentId"),
                "excerpt": side.get("excerpt") or "",
            }
            for side in conflict.get("sides") or []
        ],
    }


def _sentences(text: str) -> list[str]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]


def _phrase_length(question_tokens: list[str], sentence_tokens: list[str]) -> int:
    best = 0
    for start in range(len(question_tokens)):
        previous = -1
        length = 0
        for token in question_tokens[start:]:
            found = None
            for index in range(previous + 1, len(sentence_tokens)):
                if sentence_tokens[index] != token:
                    continue
                if previous >= 0 and index - previous > 2:
                    break
                found = index
                break
            if found is None:
                break
            previous = found
            length += 1
            if length > best:
                best = length
    return best


def _sentence_score(question_tokens: list[str], sentence: str) -> float:
    if not question_tokens:
        return 0.0
    sentence_tokens = tokenize(sentence)
    if not sentence_tokens:
        return 0.0
    present = set(sentence_tokens)
    unique = []
    seen = set()
    for token in question_tokens:
        if token in present and token not in seen:
            unique.append(token)
            seen.add(token)
    if not unique:
        return 0.0
    # A long question word such as "challenges" has to be in the sentence.
    # Overlap on "priorities" alone does not state that answer.
    required = [token for token in question_tokens if len(token) >= 8]
    if any(token not in present for token in required):
        return 0.0
    ratio = len(unique) / len(question_tokens)
    phrase = _phrase_length(question_tokens, sentence_tokens)
    extra = any(token not in set(question_tokens) for token in sentence_tokens)
    direct = phrase >= DIRECT_PHRASE or (len(unique) >= DIRECT_OVERLAP and ratio >= DIRECT_RATIO and extra)
    if not direct:
        return 0.0
    focus = len(unique) / len(sentence_tokens)
    return round(ratio + (0.2 * phrase) + focus, 4)


def _rank_documents(
    question_tokens: list[str],
    documents: list[dict],
    retrieved_ids: set[str],
) -> tuple[list[dict], list[dict]]:
    direct = []
    indirect = []
    question_set = set(question_tokens)
    for document in documents:
        best_score = 0.0
        best_sentence = ""
        shared = 0
        for sentence in _sentences(document.get("text") or ""):
            score = _sentence_score(question_tokens, sentence)
            if score > best_score:
                best_score = score
                best_sentence = sentence
            shared = max(shared, len(question_set.intersection(tokenize(sentence))))
        row = {
            "id": document["id"],
            "kind": document.get("kind") or "",
            "score": best_score,
            "sentence": best_sentence,
            "document": document,
        }
        if best_score > 0 and (document.get("kind") or "") != "web":
            row["relation"] = "direct"
            direct.append(row)
            continue
        if document["id"] in retrieved_ids or shared:
            row["relation"] = "indirect"
            row["score"] = round(shared / max(len(question_tokens), 1), 4)
            row["sentence"] = ""
            indirect.append(row)
    direct.sort(key=lambda row: (-row["score"], -_KIND_RANK.get(row["kind"], 0), row["id"]))
    indirect.sort(key=lambda row: (-row["score"], -_KIND_RANK.get(row["kind"], 0), row["id"]))
    return direct, indirect


def _merge_ranked(direct: list[dict], indirect: list[dict], conflicts: list[dict]) -> list[dict]:
    rows = [_public_rank(row) for row in direct]
    seen = {row["id"] for row in rows}
    for conflict in conflicts:
        for side in conflict.get("sides") or []:
            document_id = side.get("excerptDocumentId")
            if not document_id or document_id in seen:
                continue
            rows.append(
                {
                    "id": document_id,
                    "kind": "",
                    "relation": "conflict",
                    "score": 0.0,
                    "sentence": side.get("excerpt") or "",
                }
            )
            seen.add(document_id)
    for row in indirect:
        if row["id"] in seen or len(rows) >= RANK_LIMIT:
            continue
        rows.append(_public_rank(row))
        seen.add(row["id"])
    return rows


def _public_rank(row: dict) -> dict:
    return {
        "id": row["id"],
        "kind": row.get("kind") or "",
        "relation": row.get("relation") or "indirect",
        "score": row.get("score") or 0.0,
        "sentence": row.get("sentence") or "",
    }


def _active_conflicts(question: str, question_tokens: list[str], documents: list[dict]) -> list[dict]:
    conflicts = []
    for point in POINTS:
        if not _question_touches(question, point["triggers"]):
            continue
        conflict = _point_conflict(point, documents, question_tokens)
        if conflict:
            conflicts.append(conflict)
    return conflicts


def _question_touches(question: str, triggers: tuple[str, ...]) -> bool:
    text = question or ""
    return any(re.search(rf"\b{re.escape(trigger)}\b", text, flags=re.IGNORECASE) for trigger in triggers)


def _point_conflict(point: dict, documents: list[dict], question_tokens: list[str]) -> dict | None:
    excerpts: dict[str, dict[str, str]] = {stance["id"]: {} for stance in point["stances"]}
    voices = [document for document in documents if (document.get("kind") or "") in VOICE_KINDS]
    for document in voices:
        for stance in point["stances"]:
            sentence = _matching_sentence(document.get("text") or "", stance)
            if sentence:
                excerpts[stance["id"]][document["id"]] = sentence
    populated = {stance_id: docs for stance_id, docs in excerpts.items() if docs}
    if len(populated) < 2:
        return None
    signature: dict[str, set[str]] = {}
    for stance_id, docs in populated.items():
        for document_id in docs:
            signature.setdefault(document_id, set()).add(stance_id)
    if len({frozenset(values) for values in signature.values()}) < 2:
        return None
    by_id = {document["id"]: document for document in voices}
    sides = []
    for stance in point["stances"]:
        docs = excerpts[stance["id"]]
        if not docs:
            continue
        chosen = _choose_excerpt(docs, question_tokens, signature, stance["id"], by_id)
        if not chosen:
            continue
        document_id, sentence = chosen
        sides.append(
            {
                "stance": stance["id"],
                "label": stance["label"],
                "documentIds": sorted(docs),
                "excerptDocumentId": document_id,
                "excerpt": sentence,
            }
        )
    if len(sides) < 2:
        return None
    return {"point": point["id"], "label": point["label"], "sides": sides}


def _matching_sentence(text: str, stance: dict) -> str | None:
    for sentence in _sentences(text):
        if _sentence_matches(sentence, stance):
            return sentence
    return None


def _sentence_matches(sentence: str, stance: dict) -> bool:
    for pattern in stance["patterns"]:
        match = pattern.search(sentence)
        if not match:
            continue
        if stance.get("negated") and _is_negated(sentence, match.start()):
            continue
        return True
    return False


def _align_covered_sides(conflicts: list[dict], document_id: str, sentence: str) -> None:
    matched = _matched_stances(sentence)
    for conflict in conflicts:
        for side in conflict.get("sides") or []:
            if (conflict.get("point"), side.get("stance")) not in matched:
                continue
            document_ids = list(side.get("documentIds") or [])
            if document_id not in document_ids:
                document_ids.append(document_id)
            side["documentIds"] = sorted(document_ids)
            side["excerpt"] = sentence
            side["excerptDocumentId"] = document_id


def _matched_stances(sentence: str) -> set[tuple[str, str]]:
    found = set()
    if not sentence:
        return found
    for point in POINTS:
        for stance in point["stances"]:
            if _sentence_matches(sentence, stance):
                found.add((point["id"], stance["id"]))
    return found


def _is_negated(sentence: str, start: int) -> bool:
    prefix = sentence[max(0, start - 32) : start]
    return _NEGATION.search(prefix) is not None


def _choose_excerpt(
    excerpts: dict[str, str],
    question_tokens: list[str],
    signature: dict[str, set[str]],
    stance_id: str,
    documents_by_id: dict[str, dict],
) -> tuple[str, str] | None:
    clean = [document_id for document_id in excerpts if signature.get(document_id) == {stance_id}]
    pool = clean or list(excerpts)

    def sort_key(document_id: str) -> tuple:
        kind = (documents_by_id.get(document_id) or {}).get("kind") or ""
        return (
            -_sentence_score(question_tokens, excerpts[document_id]),
            -_KIND_RANK.get(kind, 0),
            document_id,
        )

    chosen_id = sorted(pool, key=sort_key)[0]
    return chosen_id, excerpts[chosen_id]


def _compose_conflict(direct_rows: list[dict], conflicts: list[dict]) -> tuple[str, list[str]]:
    chosen: list[tuple[str, str]] = []
    seen_text = set()

    def add(document_id: str, sentence: str) -> None:
        text = (sentence or "").strip()
        if not document_id or not text or text in seen_text:
            return
        seen_text.add(text)
        chosen.append((document_id, text))

    lead = direct_rows[0]["sentence"] if direct_rows and direct_rows[0].get("sentence") else ""
    if lead:
        add(direct_rows[0]["id"], lead)
    covered = _matched_stances(lead)
    for conflict in conflicts:
        for side in conflict.get("sides") or []:
            if (conflict.get("point"), side.get("stance")) in covered:
                continue
            add(side.get("excerptDocumentId") or "", side.get("excerpt") or "")
            if len(chosen) >= 6:
                break
        if len(chosen) >= 6:
            break
    cited = []
    for document_id, _sentence in chosen:
        if document_id not in cited:
            cited.append(document_id)
    reply = " ".join(sentence for _document_id, sentence in chosen)
    return reply, cited


def _group_reply(direct_rows: list[dict]) -> str:
    chosen = []
    seen = set()
    for row in direct_rows:
        sentence = (row.get("sentence") or "").strip()
        if not sentence or sentence in seen:
            continue
        seen.add(sentence)
        chosen.append(sentence)
        if len(chosen) == 3:
            break
    return " ".join(chosen)


def _direct_reply(document: dict, question_tokens: list[str]) -> str:
    scored = []
    for sentence in _sentences(document.get("text") or ""):
        score = _sentence_score(question_tokens, sentence)
        if score > 0:
            scored.append((score, sentence))
    if not scored:
        return ""
    keep = {sentence for _score, sentence in sorted(scored, key=lambda item: item[0], reverse=True)[:2]}
    ordered = [sentence for sentence in _sentences(document.get("text") or "") if sentence in keep]
    return " ".join(ordered)
