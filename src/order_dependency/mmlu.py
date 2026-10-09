"""The MMLU test split (``cais/mmlu`` on Hugging Face) as MCQs, fetched on demand.

``-q mmlu`` on the command line calls ``load_mmlu``. The parquet file is downloaded once by
``huggingface_hub`` into its cache (``~/.cache/huggingface``) and reused afterwards, so no
question file needs to be kept in the repo. Ids are ``mmlu-<subject>-<row>`` (row = position in
the full test split), so runs made from different subsets can be joined on question id.
"""

import random

import pandas as pd
from huggingface_hub import hf_hub_download

from order_dependency.mcq import MCQ

REPO_ID = "cais/mmlu"
FILENAME = "all/test-00000-of-00001.parquet"


def load_mmlu(per_subject: int | None = None) -> list[MCQ]:
    """Download the MMLU test split if it is not cached yet and convert it to MCQs.

    Args:
        per_subject: Keep at most this many questions per subject (random, seeded);
            ``None`` keeps all 14,042.
        seed: Seed for the subset draw.

    Returns:
        MCQs grouped by subject in alphabetical order, with the subject as ``topic``.
    """
    parquet = hf_hub_download(REPO_ID, FILENAME, repo_type="dataset")
    rows = pd.read_parquet(parquet)
    return _to_mcqs(rows, per_subject)


def _to_mcqs(rows: pd.DataFrame, per_subject: int | None) -> list[MCQ]:
    """Convert MMLU rows to MCQs, optionally drawing a stratified subset.

    Args:
        rows: Frame with ``question``, ``subject``, ``choices`` and ``answer`` columns.
        per_subject: Keep at most this many questions per subject; ``None`` keeps all.
        seed: Seed for the subset draw.

    Returns:
        MCQs grouped by subject in alphabetical order, with the subject as ``topic``.
    """
    rng = random.Random()
    mcqs = []
    for subject, group in rows.groupby("subject", sort=True):
        records = list(group.to_dict("index").items())
        if per_subject:
            n = min(per_subject, len(records))
            records = rng.sample(records, n)
        for i, row in records:
            options = tuple(str(choice) for choice in row["choices"])
            mcq = MCQ(
                id=f"mmlu-{subject}-{i}",
                question=row["question"],
                options=options,
                answer=int(row["answer"]),
                topic=subject,
            )
            mcqs.append(mcq)
    return mcqs
