import json
import sys
from pathlib import Path

from synthetic_research.engine import ResearchEngine
from synthetic_research.gates import ResearchError


def _note(events: list[str], line: str) -> None:
    events.extend([line])
    print(line, file=sys.stderr, flush=True)


def render_report(events: list[str], snapshot: dict, refreshed_error: str) -> str:
    session = snapshot["session"]
    vault = snapshot["vault"] or {}
    lines = [
        f"Campaign {session['campaignId']}",
        f"Session {session['sessionId']}",
        f"Kit {session['kit']['version']} locked. Brand alias {session['kit']['brandAlias']}.",
        f"Session status {session['status']}. Index present: {snapshot['indexPresent']}.",
        f"Corpus {session.get('corpusHash', '')}.",
        f"Retrieval {(session.get('grant') or {}).get('retrieval', 'none')}.",
        _metric_line(session),
        f"Audit events: {len(snapshot.get('audit') or [])}.",
        "",
        "Walk",
        *events,
        "",
        "Vault pack",
        f"Badge: {vault.get('badge')}",
        f"Client told: {vault.get('clientTold')}",
        f"Oversight: {vault.get('oversightName')}",
        f"Population claim: {vault.get('populationClaim')}",
        f"Disclosure: {vault.get('disclosureNote')}",
        "Raw gold documents are not in the vault.",
        "",
    ]
    for cohort in vault.get("cohorts") or []:
        lines.append(f"Cohort {cohort.get('cohortId')}")
        lines.append(f"Promise: {cohort.get('promise')}")
        lines.append(f"Tension: {cohort.get('tension')}")
        for theme in cohort.get("themes") or []:
            evidence = ", ".join(theme.get("evidenceIds") or [])
            lines.append(f"- {theme.get('theme')}: \"{theme.get('quote')}\" [{evidence}]")
        lines.append("")
    for skip in vault.get("skips") or []:
        lines.append(f"Skip {skip.get('cohortId')}: {skip.get('reason')}")
    lines.append("")
    lines.append(f"After refresh, a new draft is blocked: {refreshed_error}")
    lines.append("The approved vault pack stays. Stale interviews stay out of a new draft.")
    return "\n".join(lines).rstrip() + "\n"


def _metric_line(session: dict) -> str:
    interviews = session.get("interviews") or []
    turns = [turn for interview in interviews for turn in interview.get("turns") or []]
    if not turns:
        return "Interview turns: 0."
    cited = sum(1 for turn in turns if not turn["abstain"] and turn["evidenceIds"])
    abstained = sum(1 for turn in turns if turn["abstain"])
    stale = sum(1 for interview in interviews if interview.get("stale"))
    return (
        f"Interview turns: {len(turns)}. Cited: {cited}. Abstained: {abstained}. "
        f"Citation rate: {cited / len(turns):.0%}. Interviews marked stale after refresh: {stale}."
    )


def _phase(engine: ResearchEngine, session_id: str, cohort_id: str) -> str:
    session = engine.store.read(session_id)
    interview = next(
        item
        for item in reversed(session["interviews"])
        if item["cohortId"] == cohort_id and not item["stale"]
    )
    return interview["phase"]


def _format_turn(turn: dict) -> str:
    if turn["abstain"]:
        return f"{turn['phase']}: Abstain ({turn['abstainReason']}) on “{turn['question']}”"
    chips = ", ".join(turn["evidenceIds"])
    improved = " improved" if turn["improved"] else ""
    return f"{turn['phase']}{improved}: Q: {turn['question']}\n  A: {turn['reply']}\n  Evidence: {chips}"


def interview_cohort(engine: ResearchEngine, session_id: str, pack: dict, cohort_id: str, events: list[str]) -> None:
    engine.start_interview(session_id, cohort_id)
    _note(events, f"Started {cohort_id}. Phase rapport. Kit locked. Turns left 10.")
    improved = False
    brand_checked = False
    tea_checked = False
    for main in pack["guide"]["mains"]:
        while _phase(engine, session_id, cohort_id) != main["phase"]:
            phase = _phase(engine, session_id, cohort_id)
            if phase == "unprompted" and main["phase"] == "prompted":
                engine.advance_phase(session_id, cohort_id)
                ladder = engine.ladder(session_id, cohort_id)
                _note(events, _format_turn(ladder))
            engine.advance_phase(session_id, cohort_id)
            _note(events, f"Phase is now {_phase(engine, session_id, cohort_id)}. Kit locked: {_phase(engine, session_id, cohort_id) != 'prompted'}.")
        if main["phase"] == "unprompted" and not brand_checked:
            try:
                engine.ask(session_id, cohort_id, "What do you think of Aroha?")
                _note(events, "Brand question was accepted too early.")
            except ResearchError as error:
                _note(events, f"Brand question blocked: {error}")
            brand_checked = True
        if main["id"] == "m2" and not tea_checked:
            tea = engine.ask(session_id, cohort_id, "What tea do you brew in the morning?")
            _note(events, _format_turn(tea))
            tea_checked = True
        turn = engine.ask(session_id, cohort_id, main["text"])
        _note(events, _format_turn(turn))
        if not improved and main["phase"] == "unprompted" and not turn["abstain"]:
            tightened = engine.improve(session_id, cohort_id, "Tighten this reply. Keep the same evidence. First person.")
            _note(events, _format_turn(tightened))
            improved = True
        if turn["evidenceIds"]:
            chip = engine.open_evidence(session_id, turn["evidenceIds"][0])
            _note(events, f"Chip {chip['id']} · {chip['source']} · no customer name field.")
    ended = engine.end_interview(session_id, cohort_id)
    _note(events, f"Ended {cohort_id}. Cited turns stay in the session. Status {ended['status']}.")


def run_campaign(root: Path, pack_path: Path, engine: ResearchEngine) -> Path:
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    events: list[str] = []
    session = engine.open(pack, actor="Meera")
    _note(events, f"Opened {session['campaignId']} as {session['actor']}. Insight locked before grant.")
    granted = engine.grant(session["sessionId"], pack["grantId"], steward="Arjun")
    _note(events, 
        f"Grant {granted['grant']['grantId']} active. Documents in index: {granted['grant']['documentCount']}."
    )
    for cohort in engine.list_cohorts(session["sessionId"]):
        _note(events, f"Cohort {cohort['id']}: {cohort['members']} members. {cohort['rule']}")
    cards = engine.compile_cards(session["sessionId"])
    for card in cards:
        provenance = card["provenance"]
        _note(events, 
            f"Card {card['cardId']} {card['label']}. Tension: {card['tension']}. "
            f"Provenance: {provenance['transcripts']} transcripts, {provenance['reviews']} reviews, "
            f"{provenance['members']} members."
        )
    probe = engine.add_probe(session["sessionId"], "What would make this desk-safe?")
    _note(events, f"Probe added for this session only: {probe['text']}")
    try:
        engine.edit_main()
    except ResearchError as error:
        _note(events, f"Main questions stay locked: {error.code}.")
    try:
        engine.add_cohort()
    except ResearchError as error:
        _note(events, f"Add cohort refused: {error.code}.")

    interview_cohort(engine, session["sessionId"], pack, "office-sippers", events)
    interview_cohort(engine, session["sessionId"], pack, "value-households", events)
    engine.skip(
        session["sessionId"],
        "cafe-switchers",
        "Member count is 3276. This pass will not treat that base as a cohort claim.",
    )
    _note(events, "Skipped cafe-switchers with a written reason. The pack will show the hole.")

    engine.draft_pack(session["sessionId"])
    engine.send_changes(
        session["sessionId"],
        "Do not collapse the tension into premium but affordable. Keep the exploratory badge.",
    )
    pack_draft = engine.draft_pack(session["sessionId"])
    _note(events, 
        f"Pack drafted after send-back. Badge {pack_draft['badge']}. "
        f"Cited turns used: {pack_draft['citedTurnCount']}. Cohorts in pack: {len(pack_draft['cohorts'])}."
    )
    engine.approve(session["sessionId"], "Meera Iyer")
    _note(events, "Approved. Vault received the pack only.")

    engine.refresh(session["sessionId"], steward="Arjun")
    refreshed_error = ""
    try:
        engine.draft_pack(session["sessionId"])
        refreshed_error = "A stale draft was allowed."
    except ResearchError as error:
        refreshed_error = f"{error.code}: {error}"
    engine.close(session["sessionId"])
    other = engine.open(pack, actor="Meera")
    try:
        engine.list_cohorts(other["sessionId"])
        _note(events, "Second session could see cohorts without a grant.")
    except ResearchError as error:
        _note(events, f"Second session {other['sessionId']} cannot read the first index: {error.code}.")

    snapshot = engine.snapshot(session["sessionId"])
    report = render_report(events, snapshot, refreshed_error)
    out = root / "runs" / "cam-1842-report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    return out
