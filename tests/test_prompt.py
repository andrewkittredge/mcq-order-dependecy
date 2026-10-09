"""Tests for prompt rendering and reply parsing."""

import json
import tempfile
import unittest
from pathlib import Path

from order_dependency.harnesses.build_prompt import build_prompt
from order_dependency.harnesses.parse_reply import _parse_choice, parse_reply
from order_dependency.mcq import MCQ, MCQS, load_questions

# An extraction item: three invoice lines, the total is the second canonical segment.
INVOICE = MCQ(
    "inv",
    "What is the total due?",
    ("Subtotal: $8,400.00", "Total due: $10,080.00", "VAT: $1,680.00"),
    answer=1,
    values=("$8,400.00", "$10,080.00", "$1,680.00"),
)


class PromptTests(unittest.TestCase):
    """Prompt rendering and letter parsing."""

    def test_prompt_relabels_options_in_presented_order(self):
        """Options are relabelled A, B, C in presented order."""
        mcq = MCQ("q", "Pick one.", ("red", "green", "blue"), answer=2)
        self.assertIn("A. blue\nB. red\nC. green", build_prompt(mcq, (2, 0, 1)))

    def test_parse_choice(self):
        """Common reply shapes parse to the right presented position."""
        cases = [
            ("B", 1),
            ("(C)", 2),
            ("**A**", 0),
            ("D.", 3),
            ("The answer is C.", 2),
            ("A: because ...", 0),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(_parse_choice(text, 4), expected)


class ExtractionTests(unittest.TestCase):
    """Extraction items render as an unlabelled excerpt and parse by matching the stated figure."""

    def test_prompt_lists_segments_in_presented_order_without_letters(self):
        """Segments follow an ``Excerpt:`` heading, in the permuted order, with no letter labels."""
        prompt = build_prompt(INVOICE, (2, 0, 1))
        self.assertIn("Excerpt:\nVAT: $1,680.00\nSubtotal: $8,400.00\nTotal due: $10,080.00\n", prompt)
        self.assertNotIn("A.", prompt)

    def test_parse_reply_returns_presented_position_of_the_stated_figure(self):
        """Different spellings of the same figure map to the segment that carried it."""
        perm = (2, 0, 1)  # total due is shown last
        for text in ("$10,080.00", "10,080", "10080", "The total due is $10,080.00."):
            with self.subTest(text=text):
                self.assertEqual(parse_reply(INVOICE, perm, text), 2)

    def test_answer_may_be_given_as_a_value(self):
        """Loading resolves an ``answer`` written as the value string to the canonical index."""
        record = {
            "id": "x",
            "question": "?",
            "options": ["Fee: 1%", "Charge: 2%"],
            "values": ["1%", "2%"],
            "answer": "2%",
        }
        loaded = MCQS.validate_python([record])
        self.assertEqual(loaded[0].answer, 1)
        self.assertEqual(loaded[0].correct_text, "2%")
        self.assertTrue(loaded[0].is_extraction)


class ContextTests(unittest.TestCase):
    """Questions with a context show the document first and load it from a shared file."""

    FILING = "Revenue was $4,812 million.\nNet income was $612 million."

    def test_prompt_puts_document_before_question_and_lettered_options(self):
        """The context sits under ``Document:``; the question and ``A.``/``B.`` options follow."""
        mcq = MCQ("q", "What was revenue?", ("$4,812 million", "$612 million"), answer=0, context=self.FILING)
        expected = f"Document:\n{self.FILING}\n\nWhat was revenue?\n\nA. $612 million\nB. $4,812 million\n\nAnswer:"
        self.assertEqual(build_prompt(mcq, (1, 0)), expected)

    def test_load_questions_reads_context_file_relative_to_question_file(self):
        """``context_file`` is replaced by the text of that file, resolved next to the JSON."""
        records = [
            {"id": "a", "question": "?", "options": ["1", "2"], "answer": 0, "context_file": "filing.txt"},
            {"id": "b", "question": "?", "options": ["1", "2"], "answer": 1},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "filing.txt").write_text(self.FILING, encoding="utf-8")
            questions_path = Path(tmp) / "questions.json"
            questions_path.write_text(json.dumps(records), encoding="utf-8")
            loaded = load_questions(questions_path)
        self.assertEqual(loaded[0].context, self.FILING)
        self.assertEqual(loaded[1].context, "")


if __name__ == "__main__":
    unittest.main()
