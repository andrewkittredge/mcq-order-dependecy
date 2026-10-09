"""Order-dependency metrics computed from recorded trials.

Notation: a question has K canonical options and was shown under P permutations,
each answered n times (samples). ``P_p(o)`` is the empirical probability that the
model selected canonical option ``o`` when shown permutation ``p``.

Option Dependency Score (ODS)
-----------------------------
ODS measures how much the *selection distribution over options* moves when only
the ordering changes::

    ODS = sum_o Var_p[ P_p(o) ]  /  (1 - 1/K)

The numerator is the total variance of the selection probabilities across
permutations. It is 0 when every permutation yields the same distribution
(perfect order invariance). Its maximum, 1 - 1/K, is reached when each
permutation deterministically selects an option and the options are chosen
equally often overall - i.e. the answer is fully determined by the ordering
(e.g. the model always picks "A"). Dividing by that maximum normalises the score
to [0, 1].

``P_p(o)`` treats every trial as a one-hot pick of its chosen option, averaged over
the samples of a permutation, so it is the same quantity for every backend and,
with one sample per permutation, measures disagreement between permutations.

Supporting metrics
------------------
* consistency        - share of trials agreeing with the modal (majority) answer
* accuracy           - share of trials that are correct (order-averaged accuracy)
* position share     - how often each presented position (A, B, C, ...) is chosen;
                       uniform (1/K) under a balanced permutation set if unbiased
* recall by position - accuracy when the correct option sits at each position;
                       its std (RStd, Zheng et al. 2024) is another sensitivity index
"""

from collections import Counter, defaultdict
from itertools import groupby
from statistics import pvariance

from order_dependency.analysis.question_metrics import QuestionMetrics
from order_dependency.mcq import MCQ
from order_dependency.trial import Trial


def aggregate(mcqs: list[MCQ], trials: list[Trial]):
    """Compute per-question metrics for every question that received trials.

    Args:
        mcqs: All questions in the run (questions without trials are skipped).
        trials: All recorded trials.

    Returns:
        One row per question, in the order questions first appear in ``trials`` (the run
        asks them in ``mcqs`` order), with a column per ``QuestionMetrics`` field.
    """
    question_id_to_trials = {qid: list(group) for qid, group in groupby(trials, key=lambda t: t.question_id)}
    mcq_to_trials = {mcq: question_id_to_trials.get(mcq.id, []) for mcq in mcqs}

    for mcq, question_trials in mcq_to_trials.items():
        if question_trials:
            metrics = _question_metrics(mcq, question_trials)
            yield metrics


def _question_metrics(mcq: MCQ, trials: list[Trial]) -> QuestionMetrics:
    """Compute all per-question metrics from the question's trials.

    Args:
        mcq: The question.
        trials: Every trial recorded for ``mcq`` (any ordering, any sample).

    Returns:
        The populated ``QuestionMetrics``. ODS and consistency use answered trials
        only; ties for the modal option go to the lowest canonical index.
    """
    answered = [t for t in trials if t.canonical is not None]
    n = len(answered)
    counts = Counter(t.canonical for t in answered)
    majority = min(counts, key=lambda option: (-counts[option], option)) if n else None
    consistency = counts[majority] / n if n else 0.0
    matrix = _selection_matrix(answered, mcq.k)
    ods = _ods_from_matrix(matrix, mcq.k)
    return QuestionMetrics(question_id=mcq.id, n_answered=n, ods=ods, consistency=consistency)


def _selection_matrix(answered: list[Trial], k: int) -> dict[int, list[float]]:
    """Estimate ``P_p(o)`` from answered trials.

    Args:
        answered: One question's trials that have a ``canonical`` choice.
        k: Number of options.

    Returns:
        Permutation index -> list of length ``k`` giving the share of that permutation's
        answered trials choosing each canonical option. Permutations with no answers
        are absent.
    """
    picks: dict[int, Counter[int]] = defaultdict(Counter)
    for t in answered:
        picks[t.perm_index][t.canonical] += 1
    matrix = {}
    for perm_index, counts in picks.items():
        n = counts.total()
        matrix[perm_index] = [counts[option] / n for option in range(k)]
    return matrix


def _ods_from_matrix(matrix: dict[int, list[float]], k: int) -> float:
    """Compute the Option Dependency Score from a selection matrix.

    Args:
        matrix: Output of ``_selection_matrix``: permutation index -> ``P_p(o)`` per option.
        k: Number of options.

    Returns:
        ``sum_o Var_p[P_p(o)] / (1 - 1/k)`` in ``[0, 1]``. Returns 0.0 when fewer
        than two permutations were answered (no variance can be measured) or
        ``k < 2``.
    """
    if len(matrix) < 2 or k < 2:
        return 0.0
    columns = zip(*matrix.values())  # one sequence per option: P_p(o) across permutations
    total_variance = sum(pvariance(column) for column in columns)
    return total_variance / (1 - 1 / k)
