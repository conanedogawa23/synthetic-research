"""Turn uploaded session files into text, then into a document graph."""

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from xml.etree import ElementTree

from synthetic_research.gates import ResearchError

WORD_TEXT = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"
WORD_PARA = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9&/-]{2,}")
_PHRASE = re.compile(r"\b[A-Z][a-zA-Z0-9&/-]{2,}(?:\s+[A-Z][a-zA-Z0-9&/-]{2,}){0,3}\b")
_RELATION = re.compile(
    r"\b([A-Z][A-Za-z0-9&/-]{2,}(?:\s+[A-Z][A-Za-z0-9&/-]{2,}){0,2})\s+"
    r"(is|are|uses|supports|sells|owns|manages|includes)\s+"
    r"(?:an?\s+|the\s+)?([A-Za-z][A-Za-z0-9&/-]+(?:\s+[A-Za-z][A-Za-z0-9&/-]+){0,3})",
    re.IGNORECASE,
)
_STOP = {
    "the", "and", "for", "that", "with", "this", "from", "have", "what", "when",
    "your", "you", "are", "was", "not", "but", "they", "this", "then", "into",
}


def read_upload(filename: str, payload: bytes) -> str:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix in {"txt", "md", "csv"}:
        text = payload.decode("utf-8", errors="replace")
    elif suffix == "docx":
        text = _docx_text(payload, filename)
    elif suffix == "pdf":
        text = _pdf_text(payload, filename)
    else:
        raise ResearchError(
            "UNSUPPORTED_TYPE",
            "Upload .txt, .md, .csv, .docx, or .pdf. This file type is not read yet.",
        )
    collapsed = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not collapsed:
        raise ResearchError("KNOWLEDGE_EMPTY", f"{filename} has no readable text.")
    return collapsed


def chunk_text(text: str, size: int = 900, overlap: int = 120) -> list[str]:
    """Split a file into overlapping passages so each vector is one passage, not the whole file."""
    paragraphs = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not paragraphs:
        return []
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if current and len(current) + len(paragraph) + 1 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap and len(current) > overlap else ""
            current = f"{tail}\n{paragraph}".strip() if tail else paragraph
        else:
            current = f"{current}\n{paragraph}".strip() if current else paragraph
        while len(current) > size:
            chunks.append(current[:size])
            current = current[size - overlap if overlap < size else size :].strip()
    if current:
        chunks.append(current)
    return chunks


def build_graph(documents: list[dict]) -> dict:
    """Documents become nodes. Repeated names become entity nodes, linked by mentions and verbs."""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()

    def add_node(node_id: str, label: str, kind: str) -> None:
        nodes.setdefault(node_id, {"id": node_id, "label": label, "kind": kind})

    def add_edge(source: str, relation: str, target: str) -> None:
        key = (source, relation, target)
        if source == target or key in seen_edges or source not in nodes or target not in nodes:
            return
        seen_edges.add(key)
        edges.append({"from": source, "relation": relation, "to": target})

    for document in documents:
        document_id = document.get("id")
        if not document_id:
            continue
        add_node(document_id, document.get("title") or document_id, "document")
        text = document.get("text") or ""
        mentioned: list[str] = []
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            for phrase in _PHRASE.findall(sentence):
                label = " ".join(phrase.split())
                if label.lower() in _STOP or len(label) < 3:
                    continue
                entity_id = _entity_id(label)
                add_node(entity_id, label, "entity")
                if entity_id not in mentioned:
                    mentioned.append(entity_id)
            for match in _RELATION.finditer(sentence):
                left = _entity_id(match.group(1))
                right = _entity_id(match.group(3))
                add_node(left, " ".join(match.group(1).split()), "entity")
                add_node(right, " ".join(match.group(3).split()), "entity")
                add_edge(left, match.group(2).lower(), right)
        for entity_id in mentioned[:8]:
            add_edge(document_id, "mentions", entity_id)
    return {"nodes": list(nodes.values()), "edges": edges}


def _entity_id(label: str) -> str:
    slug = "-".join(token.lower() for token in _TOKEN.findall(label))
    return f"ent-{slug or 'term'}"


def _docx_text(payload: bytes, filename: str) -> str:
    try:
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            xml = archive.read("word/document.xml")
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise ResearchError("UNSUPPORTED_TYPE", f"{filename} is not a readable Word file.") from error
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as error:
        raise ResearchError("UNSUPPORTED_TYPE", f"{filename} has no readable document body.") from error
    lines = []
    for paragraph in root.iter(WORD_PARA):
        line = "".join(node.text or "" for node in paragraph.iter(WORD_TEXT)).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _pdf_text(payload: bytes, filename: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ResearchError(
            "UNSUPPORTED_TYPE",
            "PDF reading is not installed. Upload .txt, .md, .csv, or .docx.",
        ) from error
    try:
        reader = PdfReader(BytesIO(payload))
    except Exception as error:  # noqa: BLE001
        raise ResearchError("UNSUPPORTED_TYPE", f"{filename} could not be read as a PDF.") from error
    return "\n".join(page.extract_text() or "" for page in reader.pages)
