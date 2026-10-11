"""Build a campaign pack from a folder of synthetic-interview docx evidence."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from synthetic_research.gates import ResearchError

WORD_TEXT = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
WORD_PARA = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
QUOTE = re.compile(r"[“\"]([^”\"]{12,})[”\"]")
STIMULUS_NUMBER = re.compile(r"stimulus\s*(\d+)", re.IGNORECASE)
ALIAS = "Alder"

PHASES = ("rapport", "unprompted", "deep_dive", "prompted")
DEFAULT_MAINS = (
    ("rapport", "What are the biggest priorities for your sales organization over the next 12 to 24 months?"),
    ("unprompted", "How are you managing that today?"),
    ("deep_dive", "Why does that matter?"),
    ("prompted", "What would you need to see before you would support an initiative like this?"),
)


def load_campaign(path: Path) -> dict:
    if path.is_dir():
        return build_rag_pack(path)
    if path.suffix.lower() != ".json":
        raise ResearchError("PACK_INVALID", "Pass a campaign JSON file or a folder of docx evidence.")
    return json.loads(path.read_text(encoding="utf-8"))


def build_rag_pack(folder: Path) -> dict:
    files = sorted(path for path in folder.rglob("*.docx") if not path.name.startswith("~$"))
    stimuli = [path for path in files if "stimulus" in path.name.lower()]
    if not stimuli:
        raise ResearchError("PACK_INVALID", "The folder has no Stimulus docx files.")
    guide = next((path for path in files if "discussion guide" in path.name.lower()), None)
    documents = [_screener(), *_stimuli(stimuli)]
    _reject_alias(documents)
    return {
        "campaignId": "EVG-VP-SALES",
        "grantId": "rag-evergreen",
        "market": "GLOBAL",
        "category": "fmcg-sales",
        "kit": {
            "version": "v1",
            "locked": True,
            "brandAlias": ALIAS,
            "purpose": (
                "Synthetic interview of the Evergreen vice president of sales. "
                f"{ALIAS} is a session alias and is not a name in the evidence."
            ),
        },
        "marketPack": {
            "market": "GLOBAL",
            "household": "One commercial persona, not a household sample.",
            "taboos": [
                "Do not invent figures that are not in the evidence.",
                "Do not collapse a disagreement between documents into one fact.",
                "Do not pitch a tool before the persona has named the problem.",
            ],
        },
        "cohorts": [
            {
                "id": "vp-sales",
                "cardId": "R1",
                "label": "VP of Sales",
                "rule": "Evergreen VP of Sales · one synthetic persona",
                "bounds": ["VP of Sales", "Evergreen Consumer Products"],
                "traits": ["commercial priorities", "field execution", "proof before a new system"],
                "tension": "commercial productivity / proof before a new system",
                "affinity": ["route to market", "field reporting", "measurable outcomes"],
            }
        ],
        "guide": _guide(guide),
        "documents": documents,
    }


def docx_paragraphs(path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ResearchError("PACK_INVALID", f"{path.name} is not a readable docx.") from error
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as error:
        raise ResearchError("PACK_INVALID", f"{path.name} has no readable document body.") from error
    paragraphs = []
    for paragraph in root.iter(WORD_PARA):
        line = "".join(node.text or "" for node in paragraph.iter(WORD_TEXT)).strip()
        if line:
            paragraphs.append(line)
    return paragraphs


def _stimuli(paths: list[Path]) -> list[dict]:
    ordered = sorted(paths, key=lambda path: (_stimulus_number(path), path.name.lower()))
    documents = []
    for index, path in enumerate(ordered, start=1):
        paragraphs = docx_paragraphs(path)
        text = "\n".join(paragraphs).strip()
        if not text:
            raise ResearchError("PACK_INVALID", f"{path.name} has no text.")
        documents.append(
            {
                "id": f"ST-{index:02d}",
                "kind": "transcript",
                "cohortIds": ["vp-sales"],
                "title": path.stem,
                "text": text,
            }
        )
    return documents


def _screener() -> dict:
    return {
        "id": "SC-01",
        "kind": "screener",
        "cohortIds": ["vp-sales"],
        "members": 1,
        "title": "VP of Sales persona",
        "text": (
            "One synthetic respondent: the vice president of sales at Evergreen Consumer Products. "
            "Member count in this gold slice is 1. This is not a population claim."
        ),
    }


def _guide(path: Path | None) -> dict:
    quotes = _quoted_questions(path) if path else []
    chosen = quotes or [question for _phase, question in DEFAULT_MAINS]
    mains = []
    if len(chosen) >= len(PHASES):
        chunk = max(1, len(chosen) // len(PHASES))
        for index, question in enumerate(chosen):
            phase = PHASES[min(index // chunk, len(PHASES) - 1)]
            mains.append({"id": f"m{index + 1}", "phase": phase, "text": question})
    else:
        for index, question in enumerate(chosen):
            mains.append({"id": f"m{index + 1}", "phase": PHASES[index], "text": question})
        covered = {main["phase"] for main in mains}
        for phase, question in DEFAULT_MAINS:
            if phase in covered:
                continue
            mains.append({"id": f"m{len(mains) + 1}", "phase": phase, "text": question})
            covered.add(phase)
    return {
        "mains": mains,
        "probes": [
            {"id": "p1", "text": "Why does that matter?", "when": "A problem is named"},
            {"id": "p2", "text": "What happens when that doesn't work?", "when": "A breakdown is named"},
            {"id": "p3", "text": "What would you need to see before you would support that?", "when": "A change is proposed"},
        ],
    }


def _quoted_questions(path: Path) -> list[str]:
    questions = []
    seen = set()
    for paragraph in docx_paragraphs(path):
        for match in QUOTE.findall(paragraph):
            question = " ".join(match.split())
            if question in seen:
                continue
            seen.add(question)
            questions.append(question)
    return questions


def _stimulus_number(path: Path) -> int:
    match = STIMULUS_NUMBER.search(path.name)
    return int(match.group(1)) if match else 1000


def _reject_alias(documents: list[dict]) -> None:
    pattern = re.compile(rf"\b{re.escape(ALIAS)}\b", re.IGNORECASE)
    hits = [document["id"] for document in documents if pattern.search(document.get("text") or "")]
    if hits:
        raise ResearchError("PACK_INVALID", f"{ALIAS} appears in the evidence: {', '.join(hits)}.")
