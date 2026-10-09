"""Share of each index among a sequence of choices."""

from collections import Counter
from collections.abc import Iterable


def shares(values: Iterable[int], k: int) -> dict[int, float]:
    """Share of each index ``0..k-1`` among ``values``.

    Args:
        values: Integer choices (positions or canonical options).
        k: Number of options.

    Returns:
        Index -> share of ``values`` equal to it; 0.0 for indices never chosen.
    """
    counts = Counter(values)
    n = counts.total()
    return {j: counts[j] / n if n else 0.0 for j in range(k)}
