import argparse
import shlex
from pathlib import Path

from synthetic_research.chat import ChatClient
from synthetic_research.config import assert_inference_config, read_config
from synthetic_research.embeddings import EmbedClient
from synthetic_research.engine import ResearchEngine
from synthetic_research.gates import ResearchError
from synthetic_research.mcp_client import ResearchMcp, enable_session_mcp
from synthetic_research.rag_pack import load_campaign
from synthetic_research.store import SessionStore
from synthetic_research.synthesis import format_synthesis

ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = ROOT / "gold" / "cam-1842.json"

def session_help(pack: dict) -> str:
    lines = ["Demo, in this order:", "  grant", "  cohorts", "  cards"]
    first = pack["cohorts"][0]
    lines.append(f"  interview {first['id']}")
    lines.append(f"  {pack['guide']['mains'][0]['text']}")
    lines.append("  end")
    for cohort in pack["cohorts"][1:]:
        lines.append(f"  skip {cohort['id']} Left for a later pass")
    lines.extend(
        [
            "  pack",
            "  approve Meera Iyer",
            "",
            "Other commands: chip ID, phase, improve TEXT, status, help, quit",
            "A plain line during an interview is the question.",
        ]
    )
    return "\n".join(lines) + "\n"


class SessionCli:
    def __init__(self, engine: ResearchEngine, pack: dict):
        self.engine = engine
        self.pack = pack
        self.session = engine.open(pack, actor="Meera")
        self.cohort_id = None

    def run(self) -> None:
        print(session_help(self.pack))
        print(self._prompt())
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not line:
                continue
            if line in {"quit", "exit"}:
                return
            try:
                self._handle(line)
            except ResearchError as error:
                print(f"{error.code}: {error}")
            print(self._prompt())

    def _handle(self, line: str) -> None:
        command, _, rest = line.partition(" ")
        known = {
            "help",
            "grant",
            "cohorts",
            "cards",
            "interview",
            "phase",
            "end",
            "skip",
            "chip",
            "improve",
            "pack",
            "approve",
            "status",
        }
        if command not in known:
            self._ask(line)
            return
        if command == "help":
            print(session_help(self.pack).strip())
        elif command == "grant":
            self._grant()
        elif command == "cohorts":
            self._cohorts()
        elif command == "cards":
            self._cards()
        elif command == "interview":
            self._interview(rest.strip())
        elif command == "phase":
            interview = self.engine.advance_phase(self._session_id(), self._require_cohort())
            print(f"Phase is now {interview['phase']}. Kit locked: {interview['kitLocked']}.")
        elif command == "end":
            interview = self.engine.end_interview(self._session_id(), self._require_cohort())
            print(f"Ended {interview['cohortId']}.")
            self.cohort_id = None
        elif command == "skip":
            self._skip(rest)
        elif command == "chip":
            self._chip(rest.strip())
        elif command == "improve":
            turn = self.engine.improve(self._session_id(), self._require_cohort(), rest.strip() or "Tighten this reply.")
            self._print_turn(turn)
        elif command == "pack":
            self._pack()
        elif command == "approve":
            self._approve(rest.strip() or "Meera Iyer")
        elif command == "status":
            print(self._prompt())

    def _grant(self) -> None:
        granted = self.engine.grant(self._session_id(), self.pack["grantId"], steward="Arjun")
        grant = granted["grant"]
        print(
            f"Grant {grant['grantId']} active. Documents: {grant['documentCount']}. "
            f"Retrieval: {grant['retrieval']}. Corpus: {grant['corpusHash'][:12]}."
        )

    def _cohorts(self) -> None:
        for cohort in self.engine.list_cohorts(self._session_id()):
            print(f"{cohort['id']}: {cohort['members']} members. {cohort['rule']}")

    def _cards(self) -> None:
        for card in self.engine.compile_cards(self._session_id()):
            provenance = card["provenance"]
            print(
                f"{card['cardId']} {card['label']} ({card['cohortId']}). Tension: {card['tension']}. "
                f"Provenance: {provenance['transcripts']} transcripts, {provenance['reviews']} reviews, "
                f"{provenance['sales']} sales rows."
            )

    def _interview(self, cohort_id: str) -> None:
        if not cohort_id:
            raise ResearchError("COHORT_MISSING", "Use interview office-sippers")
        interview = self.engine.start_interview(self._session_id(), cohort_id)
        self.cohort_id = cohort_id
        print(
            f"Interview {interview['cardId']} open. Phase {interview['phase']}. "
            f"Kit locked: {interview['kitLocked']}. Turns left: {interview['turnCap']}."
        )
        print("Type the question.")

    def _ask(self, question: str, from_followup: bool = False) -> None:
        if not self.cohort_id:
            raise ResearchError("INTERVIEW_MISSING", "Open an interview before asking. Try: interview office-sippers")
        turn = self.engine.ask(self._session_id(), self.cohort_id, question)
        self._print_turn(turn)
        if from_followup or turn["abstain"]:
            return
        follow = self.engine.follow_up_question(self._session_id(), self.cohort_id, turn)
        if not follow:
            return
        print(f"Follow-up: {follow}")
        self._ask(follow, from_followup=True)

    def _skip(self, rest: str) -> None:
        parts = shlex.split(rest) if rest else []
        if len(parts) < 2:
            raise ResearchError("SKIP_REASON", "Use skip cafe-switchers The base is too small for this pass")
        cohort_id, reason = parts[0], " ".join(parts[1:])
        self.engine.skip(self._session_id(), cohort_id, reason)
        print(f"Skipped {cohort_id}. The pack will show the hole.")

    def _chip(self, evidence_id: str) -> None:
        if not evidence_id:
            raise ResearchError("EVIDENCE_MISSING", "Use chip T-14")
        chip = self.engine.open_evidence(self._session_id(), evidence_id)
        print(f"{chip['id']} · {chip['source']}")
        print(chip["text"])

    def _pack(self) -> None:
        pack = self.engine.draft_pack(self._session_id())
        print(f"Badge: {pack['badge']}. Cited turns: {pack['citedTurnCount']}. Not a population claim.")
        for cohort in pack["cohorts"]:
            print(f"\n{cohort['cohortId']}")
            print(f"Promise: {cohort['promise']}")
            print(f"Tension: {cohort['tension']}")
            for theme in cohort["themes"]:
                evidence = ", ".join(theme.get("evidenceIds") or [])
                print(f"- {theme.get('theme')}: \"{theme.get('quote')}\" [{evidence}]")
        for skip in pack["skips"]:
            print(f"Skip {skip['cohortId']}: {skip['reason']}")

    def _approve(self, name: str) -> None:
        vault = self.engine.approve(self._session_id(), name)
        print(f"Approved by {vault['oversightName']}. Vault has the pack only. Raw gold stays out.")

    def _print_turn(self, turn: dict) -> None:
        if turn["abstain"]:
            print(f"Abstain ({turn['abstainReason']}).")
            return
        print(turn["reply"])
        self._print_meta(turn)
        if turn.get("source") == "system":
            names = ", ".join(turn.get("systemDatasets") or []) or "system"
            print(f"Evidence: none. Source: {names}.")
            return
        print("Evidence: " + ", ".join(turn["evidenceIds"]))

    def _print_meta(self, turn: dict) -> None:
        confidence = turn.get("confidence") or {}
        if confidence:
            print(
                f"Confidence: {confidence.get('score')} {confidence.get('label')} "
                f"({confidence.get('basis')})."
            )
        for call in turn.get("mcp") or []:
            status = "ok" if call.get("ok") else call.get("error")
            print(f"MCP {call.get('database')} {call.get('tool')}: {status}")
        for line in format_synthesis(turn.get("synthesis")):
            print(line)

    def _require_cohort(self) -> str:
        if not self.cohort_id:
            raise ResearchError("INTERVIEW_MISSING", "No interview is open.")
        return self.cohort_id

    def _session_id(self) -> str:
        return self.session["sessionId"]

    def _prompt(self) -> str:
        session = self.engine.store.read(self._session_id())
        grant = (session.get("grant") or {}).get("status")
        if self.cohort_id:
            interview = next(
                (
                    item
                    for item in reversed(session["interviews"])
                    if item["cohortId"] == self.cohort_id and item["status"] == "open"
                ),
                None,
            )
            if interview:
                locked = "kit locked" if interview["kitLocked"] else "kit open"
                left = interview["turnCap"] - interview["turnsUsed"]
                campaign = session.get("campaignId") or "session"
                return f"{campaign} · {self.cohort_id} · {interview['phase']} · {locked} · {left} left"
        campaign = session.get("campaignId") or "session"
        grant_id = (session.get("grant") or {}).get("grantId")
        if grant == "active" and grant_id:
            return f"{campaign} · {grant_id} · cohorts"
        return f"{campaign} · no grant"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic research interview in the terminal.")
    parser.add_argument(
        "pack",
        nargs="?",
        default=str(PACK_PATH),
        help="Campaign JSON, or a folder of Stimulus docx files.",
    )
    args = parser.parse_args(argv)
    config = read_config(ROOT)
    enable_session_mcp(ROOT)
    assert_inference_config(config)
    chat = ChatClient(config)
    engine = ResearchEngine(SessionStore(ROOT / "data"), chat, embedder=EmbedClient(config), mcp=ResearchMcp())
    pack = load_campaign(Path(args.pack))
    SessionCli(engine, pack).run()


if __name__ == "__main__":
    main()
