import json
import re
import urllib.error
import urllib.request


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
    "Use only the EVIDENCE spans. Do not use world knowledge. "
    "Return one JSON object with keys abstain (boolean), reply (string), evidenceIds (array of strings). "
    "evidenceIds must be copied from the evidence ids you were given. "
    "If the evidence does not support an answer, set abstain true, reply to an empty string, and evidenceIds to []. "
    "Do not say what India thinks. Do not invent prices, brands, or places. "
    "Speak in the first person, in plain sentences. Do not mention the brand alias unless brandAllowed is true."
)

PACK_SYSTEM = (
    "You write an exploratory insight pack for one campaign. "
    "Use only the cited interview turns provided. Do not add facts. "
    "Return one JSON object with keys cohorts and disclosureNote. "
    "cohorts is an array. Each item has cohortId, promise, tension, themes. "
    "themes is an array of objects with theme, quote, evidenceIds. "
    "quote must be an exact contiguous substring of a provided reply. "
    "promise is one sentence. Do not resolve a tension into premium but affordable. "
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
        return self.complete_json(
            model=self._config["fast_model"],
            system=RESPONDENT_SYSTEM,
            user=user,
            max_tokens=700,
        )

    def pack(self, user: str) -> dict:
        return self.complete_json(
            model=self._config["reasoning_model"],
            system=PACK_SYSTEM,
            user=user,
            max_tokens=1800,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._config.get("embed_model") or ""
        if not model:
            raise RuntimeError("DEEPINFRA_EMBED_MODEL is required for the session index.")
        body = json.dumps({"model": model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self._config['base_url']}/embeddings",
            data=body,
            headers={
                "Authorization": f"Bearer {self._config['api_key']}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"DeepInfra embeddings {error.code}: {detail}") from None
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
        request = urllib.request.Request(
            f"{self._config['base_url']}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self._config['api_key']}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"DeepInfra {error.code}: {detail}") from None
        content = (payload.get("choices") or [{}])[0].get("message", {}).get("content")
        if not content:
            raise RuntimeError("DeepInfra returned an empty completion.")
        return content
