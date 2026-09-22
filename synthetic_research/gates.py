import re

PHASES = ["rapport", "unprompted", "deep_dive", "prompted"]


def mentions_brand(text: str, brand_alias: str) -> bool:
    if not brand_alias:
        return False
    pattern = re.compile(rf"\b{re.escape(brand_alias)}\b", re.IGNORECASE)
    return pattern.search(text or "") is not None


def kit_locked(phase: str) -> bool:
    return phase != "prompted"


def assert_brand_allowed(phase: str, text: str, brand_alias: str) -> None:
    if kit_locked(phase) and mentions_brand(text, brand_alias):
        raise ResearchError("BRAND_LOCKED", f"{brand_alias} is blocked until the Prompted phase.")


def next_phase(phase: str) -> str | None:
    try:
        index = PHASES.index(phase)
    except ValueError:
        return None
    if index == len(PHASES) - 1:
        return None
    return PHASES[index + 1]


def phase_has_citation(interview: dict, phase: str) -> bool:
    return any(
        turn["phase"] == phase and not turn["abstain"] and turn["evidenceIds"]
        for turn in interview["turns"]
    )


def normalize_quote(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text or "").strip().lower()
    return collapsed.replace("“", '"').replace("”", '"').replace("’", "'")


def quote_is_verbatim(quote: str, replies: list[str]) -> bool:
    needle = normalize_quote(quote)
    if len(needle) < 12:
        return False
    return any(needle in normalize_quote(reply) for reply in replies)


def public_evidence(document: dict, grant_id: str) -> dict:
    return {
        "id": document["id"],
        "kind": document["kind"],
        "title": document.get("title", ""),
        "text": document.get("text", ""),
        "cohortIds": document.get("cohortIds", []),
        "grantId": grant_id,
        "source": f"{document['kind']} · {grant_id}",
    }


class ResearchError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
