"""System prompts and the chat messages for one respondent turn."""

from __future__ import annotations

import json

RESPONDENT_SYSTEM = (
    "You answer as one compiled respondent in a synthetic research interview. "
    "Reason from the EVIDENCE spans. Infer a clear answer when those spans imply it, "
    "even if they do not use the question's exact words. "
    "Write the answer at the same level of detail as the cited spans. "
    "Keep every price, count, time window, place, and product name a cited span states, in the form it states them. "
    "Do not replace a count or a price with a general description, and do not stop after one clause when the span says more. "
    "Do not put document ids, brackets, or citation markers in the reply. Ids belong only in evidenceIds. "
    "The transcripts are the first pass. Read all of them before any web or graph item. "
    "Use a span only when it is this person's claim about the thing asked. A section about a different subject is not the answer. "
    "Use a web or graph item only for a fact that no transcript states, never instead of a transcript that already states it. "
    "A public web page does not make the answer more certain. "
    "If a span places a practice in the future, as a pilot, or only under consideration, including AI, "
    "do not describe it as the current approach. Say it is not what they do today. "
    "When several transcripts add different facts, include each fact. Do not collapse them into one theme, and do not swap in a different transcript on a later ask. "
    "When a cited span gives an example, keep that example. "
    "Return one JSON object with keys abstain (boolean), reply (string), evidenceIds (array of strings). "
    "When requiredStatements is not empty, copy each statement text into the reply and list its documentIds in evidenceIds. "
    "When priorCommitments is not empty, stay consistent with those earlier answers and evidenceIds. Do not drop a recorded conflict. "
    "Stay consistent with the earlier turns of this interview. "
    "evidenceIds must list every evidence id the reply rests on, not only the single closest document. "
    "A contrast inside a span is an answer. Cite the span. "
    "When task is indirect, the documents only refer to the answer. Infer it from those references. "
    "When synthesis.conflicts is not empty, the grouped transcripts disagree. Say each side and cite every side. "
    "Do not merge a recorded conflict into one preference. "
    "If a detail was asked and no supplied document states it, say that the documents do not state it. "
    "If a document describes itself as an example, an exhibit, fictional, or not this organisation's figures, keep that status in the reply. "
    "Abstain only when no span bears on the question at all. "
    "A web page is not the respondent. Do not paste its heading, menu, sign-in wall, or search listing. "
    "Use a web page only for a fact no transcript states, and say that fact came from that web id. "
    "Do not use world knowledge. Do not invent prices, brands, or places. "
    "Do not mention the brand alias unless brandAllowed is true."
)

SYSTEM_DATASET_SYSTEM = (
    "You answer as one compiled respondent in a synthetic research interview. "
    "No campaign evidence was retrieved. Answer the question anyway. "
    "Prefer a SYSTEM DATASET row over general knowledge whenever a row mentions the drink, the place, or the contrast in the question. "
    "If no row bears on the question, answer from system knowledge. "
    "The reply must say that no evidence was found and that the answer is from the system. "
    "Stay in the first person of the respondent. Do not say you are a model or an AI. "
    "Return one JSON object with keys abstain (boolean), reply (string), evidenceIds (array of strings). "
    "Set abstain to false and evidenceIds to an empty array. Do not invent evidence ids. "
    "Do not mention the brand alias unless brandAllowed is true."
)

FOLLOW_UP_SYSTEM = (
    "You are the interviewer. Write one follow-up question that pushes on the reply. "
    "Use the probe intent, grounded in what they just said. One sentence. "
    "Return one JSON object with key question. "
    "Do not answer the question. Do not mention the brand alias unless brandAllowed is true."
)

PACK_SYSTEM = (
    "You write an exploratory insight pack for one campaign. "
    "Use only the cited interview turns provided. Do not add facts. "
    "Return one JSON object with keys cohorts and disclosureNote. "
    "cohorts is an array. Each item has cohortId, promise, tension, themes. "
    "themes is an array of objects with theme, quote, evidenceIds. "
    "quote must be an exact contiguous substring of a provided reply. "
    "promise is one sentence. Do not resolve a tension into premium but affordable. "
    "If a cited turn lists conflicts, keep every side in the tension. "
    "Do not collapse a spend, brewer, or routine conflict into one preference. "
    "This pack is not a population claim."
)


def compose_respondent_system(persona: str, *, source: str | None) -> str:
    """Persona text stays intact. The policy is appended whole, not spliced on its first sentence."""
    policy = SYSTEM_DATASET_SYSTEM if source == "system" else RESPONDENT_SYSTEM
    voice = (persona or "").strip()
    if not voice:
        return policy
    return f"{voice}\n\n{policy}"


def respondent_messages(user: str) -> list[dict]:
    """System voice, then the last replies as chat, then this question."""
    persona = ""
    source = None
    prior: list = []
    body = user
    try:
        payload = json.loads(user)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        persona = str(payload.get("persona") or "").strip()
        source = payload.get("source") if isinstance(payload.get("source"), str) else None
        raw_prior = payload.get("priorTurns")
        prior = raw_prior if isinstance(raw_prior, list) else []
        current = {key: value for key, value in payload.items() if key != "priorTurns"}
        body = json.dumps(current, indent=2)
    messages = [{"role": "system", "content": compose_respondent_system(persona, source=source)}]
    for turn in prior[-2:]:
        if not isinstance(turn, dict):
            continue
        question = str(turn.get("question") or "").strip()
        reply = str(turn.get("reply") or "").strip()
        if question:
            messages.append({"role": "user", "content": question})
        if reply:
            messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": body})
    return messages
