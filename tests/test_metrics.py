"""Tests for ODS and the aggregate metrics, using synthetic answerers."""

import itertools
import json
import tempfile
import unittest
from pathlib import Path

from order_dependency.analysis.aggregate import _ods_from_matrix, aggregate
from order_dependency.analysis.report import write_report
from order_dependency.analysis.summarize import summarize
from order_dependency.mcq import MCQ
from order_dependency.runner.experiment_run import ExperimentRun
from order_dependency.runner.llm_config import LLMConfig
from order_dependency.trial import Trial

# Four options, correct answer at canonical index 1.
MCQ4 = MCQ("q", "?", ("a", "b", "c", "d"), answer=1)


def make_trials(mcq: MCQ, choose_position) -> list[Trial]:
    """Simulate an answerer over every full permutation of ``mcq``.

    Args:
        mcq: The question.
        choose_position: Callable ``perm -> presented position`` standing in for the model.

    Returns:
        One answered ``Trial`` per permutation.
    """
    trials = []
    for i, perm in enumerate(itertools.permutations(range(mcq.k))):
        pos = choose_position(perm)
        canon = perm[pos]
        trials.append(Trial(mcq.id, i, perm, pos, canon, canon == mcq.answer))
    return trials


class OdsTests(unittest.TestCase):
    """ODS must hit its documented endpoints."""

    def test_two_options_with_disjoint_picks_give_max_ods(self):
        """Two orderings, two disjoint one-hot picks over K=2: variance 0.25*2 / (1-1/2) = 1."""
        self.assertAlmostEqual(_ods_from_matrix({0: [1, 0], 1: [0, 1]}, 2), 1.0)

    def test_soft_probabilities_reduce_ods(self):
        """A stable modal answer with mild probability wobble yields a small positive ODS."""
        ods = _ods_from_matrix({0: [0.9, 0.1, 0, 0], 1: [0.8, 0.2, 0, 0]}, 4)
        self.assertGreater(ods, 0)
        self.assertLess(ods, 0.05)


class ExperimentRunTests(unittest.TestCase):
    """The run record must build from the runner's arguments and survive a JSON round trip."""

    def test_round_trip(self):
        """model_dump -> JSON -> model_validate_json reproduces the run, tuples included."""
        run = ExperimentRun(
            config=LLMConfig(),
            samples=1,
            questions=[MCQ4],
            trials=make_trials(MCQ4, lambda perm: 0),
        )
        loaded = ExperimentRun.model_validate_json(json.dumps(run.model_dump()))
        self.assertEqual(loaded, run)
        self.assertIsInstance(loaded.trials[0].perm, tuple)

    def test_report_with_unanswered_trial(self):
        """A refused prompt (no position/canonical) is counted as unanswered and the report still renders."""
        trials = make_trials(MCQ4, lambda perm: perm.index(1))
        trials[0] = Trial(MCQ4.id, 0, trials[0].perm, None, None, None)
        run = ExperimentRun(
            config=LLMConfig(),
            samples=1,
            questions=[MCQ4],
            trials=trials,
        )
        with tempfile.TemporaryDirectory() as out:
            summary = write_report(run, Path(out))
            report = (Path(out) / "report.md").read_text(encoding="utf-8")
        self.assertEqual(summary["n_unanswered"], 1)
        self.assertEqual(summary["n_answered"], 23)
        self.assertIn("- Prompts: 24 (1 unanswered/unparseable)", report)


class AggregateTests(unittest.TestCase):
    """Dataset-level summaries."""

    def test_aggregate_majority_vote_and_position_recall(self):
        """An always-A answerer is only right when the answer is shown first."""
        trials = make_trials(MCQ4, lambda perm: 0)
        per_question = list(aggregate([MCQ4], trials))
        self.assertEqual([q.question_id for q in per_question], ["q"])
        self.assertAlmostEqual(per_question[0].ods, 1.0)
        agg = summarize([MCQ4], trials, per_question)
        self.assertAlmostEqual(agg["mean_ods"], 1.0)
        self.assertEqual(agg["position_share"][0], 1.0)
        self.assertEqual(agg["recall_by_position"], {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0})
        self.assertEqual(agg["original_accuracy"], 0.0)  # as authored, the answer sits at B
        self.assertGreater(agg["recall_std"], 0.4)
        self.assertEqual(agg["questions_with_disagreement"], 1)


if __name__ == "__main__":
    unittest.main()
