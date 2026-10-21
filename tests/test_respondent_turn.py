import json
import tempfile
import unittest
from pathlib import Path

from synthetic_research.engine import ResearchEngine
from synthetic_research.prompts import RESPONDENT_SYSTEM, compose_respondent_system, respondent_messages
from synthetic_research.respondent_turn import (
    LEDGER_LIMIT,
    bounded_model_evidence,
    enforce_required_statements,
    prior_commitments,
    prior_turns,
    required_statements,
)
from synthetic_research.store import SessionStore

PACK = {
    "campaignId": "CAM-1842",
    "grantId": "gold-1842",
    "kit": {"version": "v3", "locked": True, "brandAlias": "Aroha", "purpose": "RTD coffee"},
    "marketPack": {"market": "IN", "taboos": []},
    "cohorts": [
        {
            "id": "office-sippers",
            "cardId": "R1",
            "label": "Office sipper",
            "rule": "Office sippers 25–34",
            "bounds": ["25–34", "not tea-only"],
            "traits": [],
            "tension": "cafe taste / value price",
            "affinity": [],
        }
    ],
    "guide": {
        "mains": [{"id": "m1", "phase": "rapport", "text": "Walk me through a weekday morning drink."}],
        "probes": [],
    },
    "documents": [
        {
            "id": "T-14",
            "kind": "transcript",
            "cohortIds": ["office-sippers"],
            "title": "Desk",
            "text": "A can looks like I packed breakfast. I can drink it at the desk.",
        },
        {
            "id": "R-882",
            "kind": "review",
            "cohortIds": ["office-sippers"],
            "title": "Medicinal",
            "text": "The sweetness sits on top of a chalky note.",
        },
        {
            "id": "S-01",
            "kind": "screener",
            "cohortIds": ["office-sippers"],
            "members": 12410,
            "title": "Incidence",
            "text": "Member count in this gold slice is 12410.",
        },
    ],
}


class FakeLlm:
    last_payload: dict | None = None

    def respondent(self, user: str) -> dict:
        FakeLlm.last_payload = json.loads(user)
        evidence = FakeLlm.last_payload["evidence"]
        return {"abstain": False, "reply": evidence[0]["text"][:120], "evidenceIds": [evidence[0]["id"]]}

    def pack(self, user: str) -> dict:
        del user
        return {"cohorts": [], "disclosureNote": "Exploratory. Not a population claim."}


class RespondentTurnTests(unittest.TestCase):
    def test_direct_statement_is_appended_when_the_model_omits_it(self):
        statements = required_statements(
            {
                "mode": "direct",
                "reply": "Weekday morning is Bru at the flat.",
                "directDocumentId": "T-14",
            }
        )
        reply, evidence_ids = enforce_required_statements(
            "I start with whatever is already on the desk.",
            [],
            statements,
            {"T-14"},
        )
        self.assertIn("already on the desk", reply)
        self.assertIn("Weekday morning is Bru at the flat.", reply)
        self.assertEqual(evidence_ids, ["T-14"])

    def test_a_statement_already_present_is_not_repeated(self):
        statements = required_statements(
            {
                "mode": "direct",
                "reply": "Weekday morning is Bru at the flat.",
                "directDocumentId": "T-14",
            }
        )
        reply, evidence_ids = enforce_required_statements(
            "Weekday morning is Bru at the flat.",
            ["T-14"],
            statements,
            {"T-14"},
        )
        self.assertEqual(reply.count("Weekday morning is Bru at the flat."), 1)
        self.assertEqual(evidence_ids, ["T-14"])

    def test_indirect_plan_has_no_required_statement(self):
        statements = required_statements(
            {
                "mode": "indirect",
                "reply": "",
                "conflicts": [{"sides": [{"excerpt": "The morning drink is mine.", "documentIds": ["T-06"]}]}],
            }
        )
        self.assertEqual(statements, [])

    def test_conflict_appends_each_missing_side(self):
        statements = required_statements(
            {
                "mode": "conflict",
                "reply": "",
                "conflicts": [
                    {
                        "sides": [
                            {
                                "excerpt": "I will not pay cafe money for a can.",
                                "excerptDocumentId": "T-14",
                                "documentIds": ["T-14"],
                            },
                            {
                                "excerpt": "I would pay cafe money for a can.",
                                "excerptDocumentId": "T-31",
                                "documentIds": ["T-31"],
                            },
                        ]
                    }
                ],
            }
        )
        reply, evidence_ids = enforce_required_statements(
            "I am fine with the price.",
            [],
            statements,
            {"T-14", "T-31"},
        )
        self.assertIn("I am fine with the price.", reply)
        self.assertIn("will not pay", reply)
        self.assertIn("would pay", reply)
        self.assertEqual(evidence_ids, ["T-14", "T-31"])

    def test_history_keeps_the_last_two_cited_replies(self):
        turns = [{"question": f"q{index}", "reply": f"a{index}", "abstain": False} for index in range(6)]
        turns.append({"question": "skip", "reply": "", "abstain": True})
        history = prior_turns(turns)
        self.assertEqual([item["question"] for item in history], ["q4", "q5"])

    def test_ledger_pulls_an_early_desk_commitment_on_a_later_desk_question(self):
        turns = []
        for index in range(16):
            turns.append(
                {
                    "question": f"Unrelated topic {index}",
                    "reply": f"Generic answer {index}",
                    "abstain": False,
                    "evidenceIds": ["T-99"],
                }
            )
        turns[1] = {
            "question": "What do you drink at the desk?",
            "reply": "A can looks like I packed breakfast at the desk.",
            "abstain": False,
            "evidenceIds": ["T-14"],
        }
        ledger = prior_commitments("Tell me again about your desk drink routine.", turns)
        questions = [row["question"] for row in ledger]
        self.assertIn("What do you drink at the desk?", questions)
        self.assertLessEqual(len(ledger), LEDGER_LIMIT)

    def test_bounded_evidence_keeps_required_documents_and_caps_the_rest(self):
        plan = {"mode": "direct", "directDocumentId": "T-1", "evidenceIds": ["T-1"]}
        evidence = [
            {"id": f"T-{index}", "kind": "transcript", "text": f"row {index}", "score": index / 10}
            for index in range(1, 20)
        ]
        bounded = bounded_model_evidence(evidence, plan, extra_limit=8)
        self.assertIn("T-1", [row["id"] for row in bounded])
        self.assertEqual(len(bounded), 9)

    def test_persona_keeps_the_full_policy_and_history_is_chat(self):
        persona = "VP of Sales. Speak from the field."
        messages = respondent_messages(
            json.dumps(
                {
                    "persona": persona,
                    "source": None,
                    "priorTurns": [{"question": "What do you drink?", "reply": "A can at the desk."}],
                    "priorCommitments": [
                        {
                            "question": "What do you drink?",
                            "reply": "A can at the desk.",
                            "evidenceIds": ["T-14"],
                        }
                    ],
                    "question": "And after that?",
                    "evidence": [],
                }
            )
        )
        system = messages[0]["content"]
        self.assertTrue(system.startswith(persona))
        self.assertIn(RESPONDENT_SYSTEM, system)
        self.assertEqual(compose_respondent_system(persona, source=None), system)
        self.assertEqual(messages[1], {"role": "user", "content": "What do you drink?"})
        self.assertEqual(messages[2], {"role": "assistant", "content": "A can at the desk."})
        current = json.loads(messages[-1]["content"])
        self.assertNotIn("priorTurns", current)
        self.assertNotIn("priorCommitments", current)
        self.assertEqual(current["question"], "And after that?")

    def test_engine_sends_prior_commitments_and_bounded_evidence(self):
        class Recording(FakeLlm):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            engine = ResearchEngine(SessionStore(Path(tmp)), Recording())
            session = engine.open(PACK, actor="Meera")
            engine.grant(session["sessionId"], "gold-1842", steward="Arjun")
            engine.compile_cards(session["sessionId"])
            engine.start_interview(session["sessionId"], "office-sippers")
            engine.ask(session["sessionId"], "office-sippers", "What do you drink at the desk?")
            engine.ask(session["sessionId"], "office-sippers", "Tell me again about your desk drink routine.")
        payload = Recording.last_payload or {}
        self.assertIn("priorCommitments", payload)
        self.assertGreaterEqual(len(payload["priorCommitments"]), 1)
        self.assertLessEqual(len(payload["evidence"]), 9)


if __name__ == "__main__":
    unittest.main()
