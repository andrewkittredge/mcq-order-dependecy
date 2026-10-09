from operator import itemgetter

from order_dependency.harnesses.parse_reply import LABELS
from order_dependency.mcq import MCQ
from order_dependency.runner.permutations import Permutation


def build_prompt(mcq: MCQ, perm: Permutation) -> str:
    """Render the question with options in the order given by ``perm``.

    Args:
        mcq: The question to render.
        perm: Permutation controlling which canonical option appears at each position.

    Returns:
        The user-turn text: question, blank line, the options, blank line, ``"Answer:"``.
        Multiple-choice options are rendered as ``"X. option"`` lines; extraction
        segments are rendered verbatim, one per line, under an ``"Excerpt:"`` heading.
        A context is placed first under a ``"Document:"`` heading.
    """
    presented = itemgetter(*perm)(mcq.options)
    if mcq.is_extraction:
        body = ["Excerpt:", *presented]
    else:
        body = [f"{label}. {text}" for label, text in zip(LABELS, presented)]
    head = ["Document:", mcq.context, ""] if mcq.context else []
    return "\n".join([*head, mcq.question, "", *body, "", "Answer:"])
