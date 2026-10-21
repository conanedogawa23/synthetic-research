#!/usr/bin/env python3
"""Run a fixed question list through one interview and write a comparison score sheet."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from synthetic_research.chat import ChatClient
from synthetic_research.config import assert_inference_config, read_config
from synthetic_research.embeddings import EmbedClient
from synthetic_research.engine import ResearchEngine
from synthetic_research.mcp_client import ResearchMcp, enable_session_mcp
from synthetic_research.rag_pack import load_campaign
from synthetic_research.respondent_turn import prior_commitments
from synthetic_research.store import SessionStore

ROOT = Path(__file__).resolve().parents[1]
_NEGATION = re.compile(r"\b(?:not|never|no|won't|wont|don't|dont|cannot|can't|cant)\b")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_questions(path: Path | None, pack: dict) -> list[str]:
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
        if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
            return [str(item).strip() for item in payload["questions"] if str(item).strip()]
        raise ValueError("Questions file must be a JSON array or {\"questions\": [...]}.")
    questions = [str(main.get("text") or "").strip() for main in (pack.get("guide") or {}).get("mains") or []]
    return [item for item in questions if item]


def _turn_kind(turn: dict) -> str:
    if turn.get("abstain"):
        return "declined"
    if turn.get("source") == "system":
        return "system"
    if turn.get("evidenceIds"):
        return "cited"
    return "uncited"


def _ledger_from_turns(turns: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for turn in turns:
        if turn.get("abstain"):
            continue
        question = str(turn.get("question") or "").strip()
        reply = str(turn.get("reply") or "").strip()
        if not question or not reply:
            continue
        rows.append(
            {
                "question": question,
                "reply": reply,
                "evidenceIds": list(turn.get("evidenceIds") or []),
            }
        )
    return rows


def _contradictions(ledger: list[dict]) -> list[dict]:
    issues: list[dict] = []
    for index, early in enumerate(ledger):
        for late in ledger[index + 1 :]:
            shared = sorted(set(early.get("evidenceIds") or []) & set(late.get("evidenceIds") or []))
            if not shared:
                continue
            early_neg = bool(_NEGATION.search(early["reply"].lower()))
            late_neg = bool(_NEGATION.search(late["reply"].lower()))
            if early_neg == late_neg:
                continue
            issues.append(
                {
                    "sharedEvidenceIds": shared,
                    "earlierQuestion": early["question"],
                    "laterQuestion": late["question"],
                    "note": "Polarity differs on shared evidence.",
                }
            )
    return issues


def run_compare(
    *,
    pack_path: Path,
    cohort_id: str,
    questions_path: Path | None,
    output_path: Path,
    actor: str,
) -> Path:
    pack = load_campaign(pack_path)
    questions = _load_questions(questions_path, pack)
    if not questions:
        raise ValueError("No questions to run.")

    config = read_config(ROOT)
    enable_session_mcp(ROOT)
    assert_inference_config(config)
    chat = ChatClient(config)
    engine = ResearchEngine(SessionStore(ROOT / "data"), chat, embedder=EmbedClient(config), mcp=ResearchMcp())

    session = engine.open(pack, actor=actor)
    session_id = session["sessionId"]
    engine.grant(session_id, pack["grantId"], steward=actor)
    engine.compile_cards(session_id)
    engine.start_interview(session_id, cohort_id)

    rows: list[dict] = []
    for number, question in enumerate(questions, start=1):
        session = engine.store.read(session_id)
        interview = next(item for item in session["interviews"] if item["cohortId"] == cohort_id)
        prior_count = len(prior_commitments(question, interview.get("turns") or []))
        turn = engine.ask(session_id, cohort_id, question)
        synthesis = turn.get("synthesis") or {}
        rows.append(
            {
                "number": number,
                "question": question,
                "kind": _turn_kind(turn),
                "abstainReason": turn.get("abstainReason"),
                "evidenceIds": list(turn.get("evidenceIds") or []),
                "synthesisMode": synthesis.get("mode"),
                "priorCommitmentsSent": prior_count,
                "replyPreview": str(turn.get("reply") or "")[:240],
            }
        )

    interview = next(
        item for item in engine.store.read(session_id)["interviews"] if item["cohortId"] == cohort_id
    )
    ledger = _ledger_from_turns(interview.get("turns") or [])
    summary = {
        "generatedAt": _now(),
        "sessionId": session_id,
        "campaignId": pack.get("campaignId"),
        "cohortId": cohort_id,
        "questionCount": len(rows),
        "cited": sum(1 for row in rows if row["kind"] == "cited"),
        "system": sum(1 for row in rows if row["kind"] == "system"),
        "declined": sum(1 for row in rows if row["kind"] == "declined"),
        "contradictions": _contradictions(ledger),
        "turns": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a fixed interview question list for comparison runs.")
    parser.add_argument(
        "pack",
        nargs="?",
        default=str(ROOT / "gold" / "cam-1842.json"),
        help="Campaign JSON path.",
    )
    parser.add_argument("--cohort", default="office-sippers", help="Cohort id to interview.")
    parser.add_argument("--questions", type=Path, help="JSON file with questions array.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "compare" / "latest.json",
        help="Where to write the score sheet.",
    )
    parser.add_argument("--actor", default="Compare", help="Actor name for session open.")
    args = parser.parse_args()
    written = run_compare(
        pack_path=Path(args.pack),
        cohort_id=args.cohort,
        questions_path=args.questions,
        output_path=args.output,
        actor=args.actor,
    )
    print(written)


if __name__ == "__main__":
    main()
