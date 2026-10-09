"""One answered prompt: a question shown under one ordering, answered once."""

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class Trial:
    """The persisted record of a single prompt/answer pair.

    A trial knows which question and ordering were shown, so the chosen letter is
    already resolved to a canonical option and a correctness flag. Only what the
    metrics need is kept, plus the raw reply so the report can show what the model
    said: ODS, consistency and accuracy use ``canonical`` per ``perm_index``; the
    position-bias tables use ``position`` and ``perm``.

    Attributes:
        question_id: Id of the ``MCQ`` that was asked.
        perm_index: Index of the ordering within the question's permutation list.
        perm: The ordering itself (canonical option at each presented position).
        position: Presented position chosen (0 = "A"), or ``None`` if unanswered.
        canonical: Canonical option chosen, or ``None`` if unanswered.
        correct: Whether the chosen option is the answer, or ``None`` if unanswered.
        reply: The model's reply text (the chosen letter for the local backend); empty
            when the request failed.
    """

    question_id: str
    perm_index: int
    perm: tuple[int, ...]
    position: int | None
    canonical: int | None
    correct: bool | None
    reply: str = ""
