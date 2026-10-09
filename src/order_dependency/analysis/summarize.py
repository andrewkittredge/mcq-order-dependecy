from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from statistics import mean, median, pstdev
from typing import Any

import numpy as np

from order_dependency.analysis.question_metrics import QuestionMetrics
from order_dependency.analysis.shares import shares
from order_dependency.mcq import MCQ
from order_dependency.trial import Trial


def summarize(mcqs: list[MCQ], trials: list[Trial], per_question: Sequence[QuestionMetrics]) -> dict[str, Any]:
    """Compute dataset-level summaries from the trials and the per-question metrics.

    Args:
        mcqs: All questions in the run.
        trials: All recorded trials.
        per_question: Output of ``aggregate`` for the same run.

    Returns:
        A JSON-serialisable dict with counts (``n_questions``, ``n_trials``,
        ``n_answered``, ``n_unanswered``), ODS statistics (``mean_ods``, ``median_ods``,
        ``max_ods``), ``questions_with_disagreement``, ``mean_consistency``,
        ``trial_accuracy``, ``original_accuracy`` (accuracy on the as-authored ordering
        only), ``position_share``, ``recall_by_position``,
        ``recall_std``, and ``per_question`` (each ``QuestionMetrics`` as a dict).
    """
    answered = [t for t in trials if t.canonical is not None]
    n = len(answered)
    recall = _recall_by_position(answered, {m.id: m for m in mcqs})
    original = [
        t.correct for t in answered if list(t.perm) == sorted(t.perm) and t.correct is not None
    ]  # identity ordering = as authored
    ods = [q.ods for q in per_question]
    consistency = [q.consistency for q in per_question]
    # A question with no answers has consistency 0.0 but is not a disagreement.
    disagreement = sum(q.n_answered > 0 and q.consistency < 1.0 for q in per_question)
    k = max(m.k for m in mcqs)
    return {
        "n_questions": len(per_question),
        "n_trials": len(trials),
        "n_answered": n,
        "n_unanswered": len(trials) - n,
        "mean_ods": mean(ods),
        "median_ods": median(ods),
        "max_ods": max(ods),
        "questions_with_disagreement": disagreement,
        "mean_consistency": mean(consistency),
        "trial_accuracy": np.mean([t.correct for t in answered if t.correct is not None]),
        "original_accuracy": np.mean(original),
        "position_share": shares([t.position for t in answered], k),
        "recall_by_position": recall,
        "recall_std": pstdev(recall.values()) if len(recall) > 1 else 0.0,
        "per_question": [asdict(q) for q in per_question],
    }


def _recall_by_position(answered: list[Trial], mcqs_by_id: dict[str, MCQ]) -> dict[int, float]:
    """Accuracy conditioned on where the correct option was displayed.

    Args:
        answered: Trials with a ``canonical`` choice, across all questions.
        mcqs_by_id: Lookup from question id to ``MCQ``.

    Returns:
        Presented position -> accuracy of the trials whose correct option sat there,
        sorted by position. Positions never hosting a correct option are absent.
    """
    hits: dict[int, list[bool]] = defaultdict(list)
    for t in answered:
        correct_position = t.perm.index(mcqs_by_id[t.question_id].answer)
        hits[correct_position].append(t.correct)
    return {position: sum(flags) / len(flags) for position, flags in sorted(hits.items())}
