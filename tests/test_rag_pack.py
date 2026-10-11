import tempfile
import unittest
import zipfile
from pathlib import Path

from synthetic_research.corpus import validate_campaign
from synthetic_research.gates import ResearchError
from synthetic_research.rag_pack import build_rag_pack, docx_paragraphs


def _docx(path: Path, paragraphs: list[str]) -> None:
    body = []
    for paragraph in paragraphs:
        text = paragraph.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        body.append(f"<w:p><w:r><w:t>{text}</w:t></w:r></w:p>")
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body)}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document)


class RagPackTests(unittest.TestCase):
    def test_stimulus_folder_becomes_a_valid_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            _docx(
                folder / "Stimulus 1 - Earnings.docx",
                ["Management priorities for 2026 are profitable growth and commercial productivity."],
            )
            _docx(
                folder / "Print - Discussion Guide Aid For the Interview.docx",
                ["Suggested opening", "“What are the biggest priorities for your sales organization?”"],
            )
            _docx(folder / "Persona Instructions - System Prompt.docx", ["You are the vice president. Do not invent facts."])
            pack = build_rag_pack(folder)
            validate_campaign(pack)
        self.assertEqual(pack["campaignId"], "EVG-VP-SALES")
        self.assertEqual(pack["cohorts"][0]["id"], "vp-sales")
        self.assertEqual([document["id"] for document in pack["documents"]], ["SC-01", "ST-01"])
        self.assertNotIn("Do not invent facts", pack["documents"][1]["text"])
        self.assertEqual(pack["guide"]["mains"][0]["text"], "What are the biggest priorities for your sales organization?")
        self.assertEqual(pack["kit"]["brandAlias"], "Alder")

    def test_missing_stimuli_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ResearchError) as error:
                build_rag_pack(Path(tmp))
        self.assertEqual(error.exception.code, "PACK_INVALID")

    def test_paragraphs_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Stimulus 2 - Roles.docx"
            _docx(path, ["Route to market covers general trade and modern trade."])
            self.assertEqual(docx_paragraphs(path), ["Route to market covers general trade and modern trade."])


if __name__ == "__main__":
    unittest.main()
