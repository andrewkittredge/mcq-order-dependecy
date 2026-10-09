"""Render a question under a given ordering and parse the model's reply.

The prompt is identical across permutations except for the order of the option
lines, so any change in the answer is attributable to ordering alone. For a
multiple-choice question the options are lettered and the model is asked for a
single letter; for an extraction item the options are the lines of a document
excerpt, shown without labels, and the model is asked for the requested figure. A
question with a ``context`` is preceded by that document, so the model must find
the answer in it rather than recall it.
Structured outputs would be more robust but would also change the decoding path
we are trying to study, so we keep the free-text reply and parse it (the raw text
is always recorded).
"""

import re
from operator import itemgetter
from typing import Sequence

from order_dependency.mcq import MCQ
from order_dependency.runner.permutations import Permutation

# Letters assigned to presented positions 0, 1, 2, ...; caps the supported option count at 10.
LABELS = "ABCDEFGHIJ"

# A letter at the very start of the reply, optionally wrapped in punctuation: "B", "(B)", "**B**", "B."
_LEADING = re.compile(r"^\W*([A-J])(?![A-Za-z])")
# Any standalone capital letter that is not part of a word: catches "The answer is C."
_ANYWHERE = re.compile(r"(?<![A-Za-z])([A-J])(?![A-Za-z])")
# A number with optional thousands separators and decimals: "4,812", "0.47", "12.5" (from "12.5%").
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_reply(mcq: MCQ, perm: Permutation, text: str) -> int | None:
    """Parse a reply to ``build_prompt(mcq, perm)`` into a presented position.

    Args:
        mcq: The question that was asked.
        perm: The ordering it was shown under.
        text: The model's reply.

    Returns:
        The presented position the reply selects (a letter for multiple choice, the
        segment whose figure was extracted for extraction items), or ``None``.
    """
    if mcq.is_extraction:
        presented_values = itemgetter(*perm)(mcq.values)
        return _parse_value(text, presented_values)
    return _parse_choice(text, mcq.k)


def _parse_choice(text: str, k: int) -> int | None:
    """Extract the chosen option letter from a free-text reply.

    A letter at the very start of the reply wins; otherwise the first standalone
    valid letter anywhere in the text is used. Lower-case letters are ignored so
    that words such as "a" or "I" are never mistaken for answers.

    Args:
        text: The model's reply.
        k: Number of options, which bounds the valid letters to ``LABELS[:k]``.

    Returns:
        The presented position (0 = "A") of the chosen letter, or ``None`` if no
        valid letter was found.
    """
    valid = LABELS[:k]
    text = text.strip()
    m = _LEADING.match(text)
    if m and m.group(1) in valid:
        return valid.index(m.group(1))
    for m in _ANYWHERE.finditer(text):
        if m.group(1) in valid:
            return valid.index(m.group(1))
    return None


def _numbers(text: str) -> set[float]:
    """Collect every number in a string, ignoring currency symbols, units and thousands separators.

    Args:
        text: Any text, e.g. ``"$4,812 million"``.

    Returns:
        The numbers found, e.g. ``{4812.0}``.
    """
    return {float(m.replace(",", "")) for m in _NUMBER.findall(text)}


def _mentions(text: str, value: str) -> bool:
    """Whether a reply states ``value``.

    Numeric values match on their numbers, so ``"4,812"``, ``"$4,812 million"`` and
    ``"4812"`` all agree; non-numeric values match as a case-insensitive substring.

    Args:
        text: The model's reply.
        value: A candidate value from an extraction item.

    Returns:
        ``True`` if the reply contains the value.
    """
    wanted = _numbers(value)
    if wanted:
        return wanted <= _numbers(text)
    return value.lower() in text.lower()


def _parse_value(text: str, values: Sequence[str]) -> int | None:
    """Find which candidate value a free-text reply states.

    Args:
        text: The model's reply.
        values: Candidate values in the order their segments were presented.

    Returns:
        The presented position of the single value the reply mentions, or ``None``
        if the reply mentions none of them or more than one.
    """
    matches = [j for j, value in enumerate(values) if _mentions(text, value)]
    return matches[0] if len(matches) == 1 else None
