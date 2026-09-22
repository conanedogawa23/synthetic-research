from synthetic_research.retrieve import build_index, retrieve


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def retrieve_evidence(
    documents: list[dict],
    query: str,
    *,
    cohort_id: str,
    query_vector: list[float] | None = None,
    limit: int = 4,
) -> list[dict]:
    index = build_index(documents)
    lexical_hits = retrieve(index, query, cohort_id=cohort_id, limit=max(len(index["docs"]), 1))
    lexical = {item["id"]: item["score"] for item in lexical_hits}
    lexical_max = max(lexical.values(), default=0) or 1.0
    use_vectors = bool(query_vector) and any(document.get("vector") for document in index["docs"])
    ranked = []
    for document in index["docs"]:
        if cohort_id not in document.get("cohortIds", []):
            continue
        lexical_score = lexical.get(document["id"], 0) / lexical_max
        semantic = cosine(query_vector or [], document.get("vector") or []) if use_vectors else 0.0
        if use_vectors:
            if semantic < 0.2 and lexical_score <= 0:
                continue
            score = (0.65 * semantic) + (0.35 * lexical_score)
        else:
            if lexical_score <= 0:
                continue
            score = lexical_score
        ranked.append((score, semantic, document))
    ranked.sort(key=lambda row: row[0], reverse=True)
    return [
        {
            "id": document["id"],
            "kind": document["kind"],
            "title": document.get("title", ""),
            "text": document.get("text", ""),
            "cohortIds": document.get("cohortIds", []),
            "score": round(score, 4),
            "semantic": round(semantic, 4),
        }
        for score, semantic, document in ranked[:limit]
    ]
