import json
import unittest

from synthetic_research.prompts import RESPONDENT_SYSTEM, compose_respondent_system, respondent_messages
from synthetic_research.respondent_turn import enforce_required_statements, prior_turns, required_statements


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

    def test_history_keeps_the_last_four_cited_replies(self):
        turns = [{"question": f"q{index}", "reply": f"a{index}", "abstain": False} for index in range(6)]
        turns.append({"question": "skip", "reply": "", "abstain": True})
        history = prior_turns(turns)
        self.assertEqual([item["question"] for item in history], ["q2", "q3", "q4", "q5"])

    def test_persona_keeps_the_full_policy_and_history_is_chat(self):
        persona = "VP of Sales. Speak from the field."
        messages = respondent_messages(
            json.dumps(
                {
                    "persona": persona,
                    "source": None,
                    "priorTurns": [{"question": "What do you drink?", "reply": "A can at the desk."}],
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
        self.assertEqual(current["question"], "And after that?")
