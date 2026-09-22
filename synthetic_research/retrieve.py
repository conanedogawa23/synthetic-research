import math
import re

STOP = {
    "the", "and", "for", "that", "with", "this", "from", "have", "what", "when",
    "your", "you", "are", "was", "were", "not", "but", "they", "them", "then",
    "into", "about", "just", "like", "does", "did", "can", "could", "would",
    "should", "there", "their", "here", "out", "how", "who", "why", "its",
}

SOURCE_KINDS = {"transcript", "review", "screener"}


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split()
        if len(token) > 2 and token not in STOP
    ]


def build_index(documents: list[dict]) -> dict:
    docs = []
    for document in documents:
        docs.append({**document, "tokens": tokenize(f"{document.get('title', '')} {document.get('text', '')}")})
    document_frequency: dict[str, int] = {}
    for document in docs:
        for token in set(document["tokens"]):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    average_length = sum(len(document["tokens"]) for document in docs) / max(len(docs), 1)
    return {
        "docs": docs,
        "document_frequency": document_frequency,
        "average_length": average_length,
        "count": len(docs),
    }


def _bm25(query_tokens: list[str], document: dict, index: dict) -> float:
    k1 = 1.2
    b = 0.75
    counts: dict[str, int] = {}
    for token in document["tokens"]:
        counts[token] = counts.get(token, 0) + 1
    score = 0.0
    length_norm = 1 - b + (b * len(document["tokens"])) / max(index["average_length"], 1)
    for token in query_tokens:
        frequency = counts.get(token, 0)
        if not frequency:
            continue
        df = index["document_frequency"].get(token, 0)
        idf = math.log(1 + (index["count"] - df + 0.5) / (df + 0.5))
        score += idf * ((frequency * (k1 + 1)) / (frequency + k1 * length_norm))
    return score


def retrieve(index: dict, query: str, *, cohort_id: str | None = None, limit: int = 4) -> list[dict]:
    query_tokens = tokenize(query)
    ranked = []
    for document in index["docs"]:
        if document.get("kind") not in SOURCE_KINDS:
            continue
        if cohort_id and cohort_id not in document.get("cohortIds", []):
            continue
        score = _bm25(query_tokens, document, index)
        if score > 0:
            ranked.append((score, document))
    ranked.sort(key=lambda row: row[0], reverse=True)
    return [
        {
            "id": document["id"],
            "kind": document["kind"],
            "title": document.get("title", ""),
            "text": document.get("text", ""),
            "cohortIds": document.get("cohortIds", []),
            "score": round(score, 4),
        }
        for score, document in ranked[:limit]
    ]
