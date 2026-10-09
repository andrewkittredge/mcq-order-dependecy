"""Tests for the report module: the file it writes and the Markdown it renders."""

import itertools
import tempfile
import unittest
from pathlib import Path

from order_dependency.analysis.report import write_report
from order_dependency.harnesses.parse_reply import LABELS
from order_dependency.mcq import MCQ
from order_dependency.runner.experiment_run import ExperimentRun
from order_dependency.runner.llm_config import LLMConfig
from order_dependency.trial import Trial

# Four options, correct answer at canonical index 1 ("b").
MCQ4 = MCQ("q", "Pick one.", ("a", "b", "c", "d"), answer=1, topic="colours", difficulty="easy")
CONFIG = LLMConfig(backend="claude", model="claude-opus-5")


def make_trials(mcq: MCQ, choose_position) -> list[Trial]:
    """Simulate an answerer over every permutation of ``mcq``.

    Args:
        mcq: The question.
        choose_position: Callable ``perm -> presented position`` standing in for the model.

    Returns:
        One answered ``Trial`` per permutation, with the chosen letter as its reply.
    """
    trials = []
    for i, perm in enumerate(itertools.permutations(range(mcq.k))):
        pos = choose_position(perm)
        canon = perm[pos]
        trials.append(Trial(mcq.id, i, perm, pos, canon, canon == mcq.answer, reply=LABELS[pos]))
    return trials


def make_run(trials: list[Trial]) -> ExperimentRun:
    """Wrap trials for ``MCQ4`` in a run record.

    Args:
        trials: The trials.

    Returns:
        A run with ``CONFIG`` and one sample per ordering.
    """
    return ExperimentRun(config=CONFIG, samples=1, questions=[MCQ4], trials=trials)


def render(trials: list[Trial]) -> str:
    """Write the report for trials of ``MCQ4`` to a temporary directory and return its text.

    Args:
        trials: The trials.

    Returns:
        The contents of ``report.md``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        write_report(make_run(trials), Path(tmp))
        return (Path(tmp) / "report.md").read_text(encoding="utf-8")


class WriteReportTests(unittest.TestCase):
    """``write_report`` writes ``report.md`` and returns the summary."""

    def test_writes_report_and_returns_summary(self):
        """The output directory is created, report.md is the rendered report, and the summary comes back."""
        run = make_run(make_trials(MCQ4, lambda perm: perm.index(1)))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "run"
            summary = write_report(run, out)
            names = [p.name for p in out.iterdir()]
            report = (out / "report.md").read_text(encoding="utf-8")
        self.assertEqual(names, ["report.md"])
        self.assertEqual(summary["n_trials"], 24)
        self.assertAlmostEqual(summary["mean_ods"], 0.0)
        self.assertTrue(report.startswith("# Order dependency report"))


class RenderMarkdownTests(unittest.TestCase):
    """The Markdown report has every section and the tables reflect the trials."""

    def test_sections_and_header(self):
        """Every section heading is present and the header names the model and prompt count."""
        report = render(make_trials(MCQ4, lambda perm: perm.index(1)))
        for heading in (
            "## Headline metrics",
            "## Answer-moving attack (paper Table 1)",
            "## Position bias",
            "## Per-question results",
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, report)
        self.assertIn("- Model: `claude-opus-5`", report)
        self.assertIn("- Prompts: 24 (0 unanswered/unparseable)", report)

    def test_invariant_answerer_scores_zero_ods(self):
        """A model that always finds the answer scores ODS 0 with 100% consistency."""
        report = render(make_trials(MCQ4, lambda perm: perm.index(1)))
        self.assertIn("| **0.000**", report)
        self.assertIn("| 100.0%", report)

    def test_always_a_answerer_fills_position_table(self):
        """An always-A model scores ODS 1 and puts 100% of picks on A."""
        report = render(make_trials(MCQ4, lambda perm: 0))
        self.assertIn("| **1.000**", report)
        self.assertRegex(report, r"\| Chosen share\s+\| 100\.0%\s+\| 0\.0%\s+\| 0\.0%\s+\| 0\.0%\s+\|")

    def test_per_question_section(self):
        """Each question shows its metrics, marks the answer, and lists every ordering with the model's reply."""
        report = render(make_trials(MCQ4, lambda perm: 0))
        self.assertIn("### q (colours, easy)\n\nPick one.\n\nODS 1.000 · consistency 25.0% · answered 24", report)
        self.assertIn("- b  ✅", report)
        self.assertIn("| A. a · B. b · C. c · D. d | A | A. a | ❌ |", report)
        self.assertIn("| A. b · B. a · C. c · D. d | A | A. b | ✅ |", report)

    def test_unanswered_trial_row(self):
        """A trial without an answer shows its reply text and dashes for the choice and verdict."""
        refused = Trial(MCQ4.id, 0, (0, 1, 2, 3), None, None, None, reply="I cannot say.")
        answered = Trial(MCQ4.id, 1, (1, 0, 2, 3), 0, 1, True, reply="A")
        report = render([refused, answered])
        self.assertIn("| A. a · B. b · C. c · D. d | I cannot say. | — | — |", report)


if __name__ == "__main__":
    unittest.main()
