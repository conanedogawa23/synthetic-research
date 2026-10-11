import json
import re
import time
import urllib.error
import urllib.request

_BUSY_STATUS = {429, 503}
_RETRY_WAITS = (1.5, 3.0)


def parse_model_json(text: str) -> dict:
    raw = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    body = fenced.group(1).strip() if fenced else raw
    start = body.find("{")
    end = body.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model did not return a JSON object.")
    return json.loads(body[start : end + 1])


RESPONDENT_SYSTEM = (
    "You answer as one compiled respondent in a synthetic research interview. "
    "Reason from the EVIDENCE spans. Infer a clear answer when those spans imply it, "
    "even if they do not use the question's exact words. "
    "Lead with the answer. Be sharp and specific: use the names, numbers, and examples in the transcripts. No preamble, no hedging, and no generic management language. "
    "The transcripts are the first pass and they decide the answer. Read all of them before any web or graph item. Cite those transcript ids. "
    "Use a web or graph item only for a fact that no transcript states, never instead of a transcript that already states it. "
    "When several transcripts add different facts, include each fact. Do not collapse them into one theme, and do not swap in a different transcript on a later ask. "
    "Return one JSON object with keys abstain (boolean), reply (string), evidenceIds (array of strings). "
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

GRADE_SYSTEM = (
    "You decide whether the transcripts are enough to answer the question as the respondent. "
    "sufficient is true only when those texts contain the facts the answer needs. "
    "A related question is not sufficient when the needed fact is absent. "
    "Shared words are not enough. A web search is not available to you. "
    "Return one JSON object with keys sufficient (boolean) and reason (string)."
)

VERIFY_SYSTEM = (
    "You check a respondent reply against the documents and nothing else. "
    "supported is true only when every factual claim is stated or clearly implied by the documents. "
    "Rewrite the reply so it keeps only those claims, in the first person, sharp and specific. "
    "If the question asks for a detail no document states, the reply must say the documents do not state it. "
    "If a document describes itself as an example, an exhibit, fictional, or not this organisation's figures, keep that status. "
    "A web document may support only a fact that no transcript states. Say that the fact came from that web id. "
    "evidenceIds must be chosen from the document ids provided. Do not invent an id. "
    "Return one JSON object with keys supported (boolean), reply (string), and evidenceIds (array of strings)."
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


class DeepInfraClient:
    def __init__(self, config: dict):
        self._config = config

    def complete_json(self, *, model: str, system: str, user: str, max_tokens: int) -> dict:
        content = self._chat(model, system, user, max_tokens)
        try:
            return parse_model_json(content)
        except (ValueError, json.JSONDecodeError):
            repaired = self._chat(
                model,
                "Return one JSON object and nothing else.",
                f"Convert this into one JSON object, keeping the same fields:\n{content}",
                max_tokens,
            )
            return parse_model_json(repaired)

    def respondent(self, user: str) -> dict:
        system = RESPONDENT_SYSTEM
        try:
            if json.loads(user).get("source") == "system":
                system = SYSTEM_DATASET_SYSTEM
        except json.JSONDecodeError:
            system = RESPONDENT_SYSTEM
        return self.complete_json(
            model=self._config["fast_model"],
            system=system,
            user=user,
            max_tokens=4096,
        )

    def pack(self, user: str) -> dict:
        return self.complete_json(
            model=self._config["reasoning_model"],
            system=PACK_SYSTEM,
            user=user,
            max_tokens=4096,
        )

    def follow_up(self, user: str) -> dict:
        return self.complete_json(
            model=self._config["fast_model"],
            system=FOLLOW_UP_SYSTEM,
            user=user,
            max_tokens=120,
        )

    def grade_transcripts(self, question: str, documents: list[dict]) -> dict:
        return self.complete_json(
            model=self._config["fast_model"],
            system=GRADE_SYSTEM,
            user=json.dumps({"question": question, "transcripts": documents}, indent=2),
            max_tokens=300,
        )

    def verify_reply(self, question: str, reply: str, documents: list[dict]) -> dict:
        return self.complete_json(
            model=self._config["fast_model"],
            system=VERIFY_SYSTEM,
            user=json.dumps({"question": question, "reply": reply, "documents": documents}, indent=2),
            max_tokens=4096,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._config.get("embed_model") or ""
        if not model:
            raise RuntimeError("DEEPINFRA_EMBED_MODEL is required for the session index.")
        body = json.dumps({"model": model, "input": texts}).encode("utf-8")
        payload = _read_json(
            lambda: urllib.request.Request(
                f"{self._config['base_url']}/embeddings",
                data=body,
                headers={
                    "Authorization": f"Bearer {self._config['api_key']}",
                    "Content-Type": "application/json",
                },
                method="POST",
            ),
            "embeddings",
        )
        rows = payload.get("data") or []
        rows.sort(key=lambda row: row.get("index", 0))
        vectors = [row.get("embedding") for row in rows]
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise RuntimeError("DeepInfra returned an incomplete embedding batch.")
        return vectors

    def _chat(self, model: str, system: str, user: str, max_tokens: int) -> str:
        body = json.dumps(
            {
                "model": model,
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        ).encode("utf-8")
        payload = _read_json(
            lambda: urllib.request.Request(
                f"{self._config['base_url']}/chat/completions",
                data=body,
                headers={
                    "Authorization": f"Bearer {self._config['api_key']}",
                    "Content-Type": "application/json",
                },
                method="POST",
            ),
            "chat",
        )
        content = (payload.get("choices") or [{}])[0].get("message", {}).get("content")
        if not content:
            raise RuntimeError("DeepInfra returned an empty completion.")
        return content


def _read_json(request_for, kind: str) -> dict:
    """POST to DeepInfra, waiting through a short busy response."""
    waits = (0.0, *_RETRY_WAITS)
    for attempt, pause in enumerate(waits):
        if pause:
            time.sleep(pause)
        try:
            with urllib.request.urlopen(request_for(), timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:400]
            busy = error.code in _BUSY_STATUS
            if busy and attempt < len(waits) - 1:
                print(f"[deepinfra] {kind} busy, retry {attempt + 1}", flush=True)
                continue
            if busy:
                raise RuntimeError("The interview model is busy. Ask again in a moment.") from None
            label = "DeepInfra embeddings" if kind == "embeddings" else "DeepInfra"
            raise RuntimeError(f"{label} {error.code}: {detail}") from None
    raise RuntimeError("The interview model is busy. Ask again in a moment.")
