"""Order-dependency metrics for a single question."""

from pydantic.dataclasses import dataclass


@dataclass
class QuestionMetrics:
    """What the dataset-level summary needs from one question.

    Attributes:
        question_id: Id of the ``MCQ`` these metrics describe.
        n_answered: Prompts that yielded a parseable letter; a question with none is
            not counted as a disagreement.
        ods: Option Dependency Score in ``[0, 1]`` (see the ``aggregate`` module docstring).
        consistency: Share of answered trials agreeing with the modal option.
    """

    question_id: str
    n_answered: int
    ods: float
    consistency: float
