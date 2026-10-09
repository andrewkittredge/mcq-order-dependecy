"""Multiple-choice question model.

Options are stored in a *canonical* order (the order in the input file). Every
permutation the experiment presents to the model is expressed as a tuple of
canonical indices, so answers can always be mapped back to "which option" rather
than "which letter".

A question file is a JSON list of objects; load it with ``MCQS.validate_json``::

    [
      {"id": "q1", "question": "...", "options": ["...", "..."],
       "answer": 2 | "option text", "topic": "...", "difficulty": "easy|medium|hard"},
      ...
    ]

An *extraction item* uses the same record with an extra ``values`` list: each option is
then a segment of a document (a line of an invoice, a sentence of an earnings release)
and ``values[i]`` is the figure that segment carries. The segments are permuted like
options, the model is asked to extract a figure rather than pick a letter, and its reply
is matched back to the segment the figure came from. ``answer`` may name the value.

A record may carry a ``context``: a document the question is about, shown to the model
before the question. ``load_questions`` also accepts ``context_file``, a path relative to
the question file, so one long document can be shared by many records.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, model_validator
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class MCQ:
    """A multiple-choice question with its options in canonical order.

    Attributes:
        id: Stable identifier used to join trials back to the question.
        question: The question stem shown to the model.
        options: Answer options in canonical (input-file) order.
        answer: Canonical index of the correct option. In input data it may also be
            given as the option's text, which is resolved to its index on load.
        topic: Free-text subject area, used only for reporting.
        difficulty: Free-text difficulty label, used only for reporting.
        values: For extraction items, the figure carried by each option (same order as
            ``options``); empty for ordinary multiple-choice questions.
        context: Document text shown before the question; empty for a bare question.
    """

    id: str
    question: str
    options: tuple[str, ...]
    answer: int
    topic: str = ""
    difficulty: str = ""
    values: tuple[str, ...] = ()
    context: str = ""

    @model_validator(mode="before")
    @classmethod
    def _answer_text_to_index(cls, data: Any) -> Any:
        """Accept ``answer`` as option text (or, for extraction items, a value) and convert it to its index.

        Args:
            data: Raw input; only dict input (e.g. from JSON) is inspected.

        Returns:
            The input with ``answer`` as an ``int``.
        """
        if isinstance(data, dict) and isinstance(data.get("answer"), str):
            options = list(data["options"])
            candidates = options if data["answer"] in options else list(data.get("values", ()))
            data = {**data, "answer": candidates.index(data["answer"])}
        return data

    @property
    def is_extraction(self) -> bool:
        """Whether this is an extraction item rather than a lettered multiple-choice question.

        Returns:
            ``True`` when ``values`` is non-empty.
        """
        return bool(self.values)

    @property
    def display(self) -> tuple[str, ...]:
        """Short label for each option, in canonical order.

        Returns:
            The values for an extraction item (segments are too long for tables), else the options.
        """
        return self.values or self.options

    @property
    def k(self) -> int:
        """Number of answer options.

        Returns:
            The option count ``K`` used throughout the metrics.
        """
        return len(self.options)

    @property
    def correct_text(self) -> str:
        """Short label of the correct option.

        Returns:
            The ``display`` entry at the canonical ``answer`` index.
        """
        return self.display[self.answer]


MCQS = TypeAdapter(list[MCQ])
"""Loader for a question file: ``MCQS.validate_json(path.read_text())``."""


def load_questions(path: Path) -> list[MCQ]:
    """Load a question file, reading any ``context_file`` into ``context``.

    Args:
        path: The JSON question file; ``context_file`` entries are resolved relative to its directory.

    Returns:
        The questions, in file order.
    """
    records = json.loads(path.read_text(encoding="utf-8"))
    documents: dict[str, str] = {}
    for record in records:
        name = record.pop("context_file", None)
        if name is not None:
            if name not in documents:
                documents[name] = (path.parent / name).read_text(encoding="utf-8")
            record["context"] = documents[name]
    return MCQS.validate_python(records)
