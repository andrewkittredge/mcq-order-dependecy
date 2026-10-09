"""Option orderings for a question: the correct option at every position.

A permutation is a tuple ``perm`` of canonical option indices where ``perm[j]``
is the canonical option shown at presented position ``j`` (position 0 = "A").
"""

Permutation = tuple[int, ...]


def place_correct(k: int, answer: int) -> list[Permutation]:
    """Move only the correct option through every position; distractors keep their order.

    This is the "answer-moving attack" of Zheng et al. (2024): it isolates pure
    *position* bias from effects caused by which distractors sit next to each other.

    Args:
        k: Number of options.
        answer: Canonical index of the correct option.

    Returns:
        ``k`` permutations, the ``p``-th showing the correct option at position ``p``.
    """
    others = [i for i in range(k) if i != answer]
    perms = []
    for pos in range(k):
        order = others[:pos] + [answer] + others[pos:]
        perms.append(tuple(order))
    return perms
