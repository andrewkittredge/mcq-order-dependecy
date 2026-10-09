import argparse
import asyncio
import logging
from itertools import product
from pathlib import Path
from typing import Sequence

from tqdm.asyncio import tqdm

from order_dependency.analysis.report import write_report
from order_dependency.mcq import MCQ, load_questions
from order_dependency.mmlu import load_mmlu
from order_dependency.runner.experiment_run import ExperimentRun
from order_dependency.runner.llm_config import LLMConfig
from order_dependency.runner.permutations import Permutation, place_correct
from order_dependency.trial import Trial

logger = logging.getLogger(__name__)

# Anchored to the project root (bin/order-dependency.py -> root) so the CLI works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = PROJECT_ROOT / "data" / "questions.json"
RESULTS_DIR = PROJECT_ROOT / "results"


def run_experiment(args: argparse.Namespace) -> int:
    """Query the LLM under every ordering and write the report.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code: 0 on success, 1 if the first request failed.
    """
    if str(args.questions) == "mmlu":
        mcqs = load_mmlu(args.per_subject)
    else:
        mcqs = load_questions(args.questions)
    config = LLMConfig(
        backend=args.backend,
        model=args.model,
        quantized=args.quantize,
        thinking=args.thinking,
        effort=None if args.effort == "none" else args.effort,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
    )

    run = asyncio.run(_run_experiment(mcqs, config, samples=args.samples))

    out = RESULTS_DIR / args.out  # an absolute --out is kept as-is by pathlib
    summary = write_report(run, out)
    logger.info(
        f"""
        Mean ODS: {summary["mean_ods"]:.3f}
        questions with order-induced changes: {summary["questions_with_disagreement"]}/{summary["n_questions"]}
        accuracy: {summary["trial_accuracy"]:.1%}
        Report written to {out / "report.md"} (plus results.json, trials.csv)
    """
    )
    logger.info("Report written to %s (plus results.json, trials.csv)", out / "report.md")

    return 0


async def _run_experiment(
    mcqs: list[MCQ],
    config: LLMConfig,
    samples: int = 1,
) -> ExperimentRun:
    """Run the full experiment.

    Args:
        mcqs: Questions to ask.
        config: LLM request settings.
        samples: Times each ordering is asked.

    Returns:
        The completed run. Feed it to ``report.write_report`` to compute metrics.

    """
    # Every prompt the run will send: (question, permutation index, permutation), `samples` times each.
    jobs: list[tuple[MCQ, int, Permutation]] = []
    for mcq in mcqs:
        perms = place_correct(mcq.k, mcq.answer)
        for (p, perm), _ in product(enumerate(perms), range(samples)):
            jobs.append((mcq, p, perm))
    logger.info(
        f"Running {len(jobs)} prompts against {config.model} ({config.backend}, samples={samples})",
    )
    answerer = config.answerer  # built once: for `hf` each construction loads another copy of the model onto the GPU
    answers = (answerer.answer(*job) for job in jobs)
    trials: Sequence[Trial] = await tqdm.gather(*answers, initial=1, total=len(jobs), unit="prompt")

    return ExperimentRun(config=config, samples=samples, questions=mcqs, trials=trials)
