"""Query the LLM under every ordering of each question and write a report.

Examples:
    order-dependency -o opus5-nothink --thinking disabled     # writes <repo>/results/opus5-nothink
    order-dependency -q mmlu --per-subject 20 -o mmlu-subset  # MMLU from HF
"""

import argparse
import logging
import sys
from pathlib import Path

from order_dependency.runner.run_experiment import run_experiment


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``order-dependency`` console script.

    Args:
        argv: Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code from ``run_experiment``.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # progress and the headline result, on stderr
    parser = argparse.ArgumentParser(
        prog="order-dependency",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-q",
        "--questions",
        type=Path,
        help="MCQ JSON file, or `mmlu` to fetch the MMLU test split from Hugging Face (cached after the first run)",
    )
    parser.add_argument(
        "--per-subject",
        type=int,
        default=None,
        help="with -q mmlu: questions per subject, drawn with --seed (default: all)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("latest"),
        help="run name or output directory; relative paths are placed under",
    )
    parser.add_argument(
        "--backend",
        choices=["claude", "hf"],
        default="claude",
        help="claude = Anthropic API; hf = local Hugging Face model scored from option-letter logits "
        "(needs `uv sync --group hf`)",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-5",
        help="Anthropic model id (e.g. claude-opus-5, claude-haiku-4-5, claude-sonnet-4-5, claude-opus-4-5), "
        "or HF repo id for --backend hf (e.g. Qwen/Qwen2.5-0.5B, huggyllama/llama-7b, meta-llama/Llama-3.1-8B)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="with --backend hf: load the model in 4-bit (bitsandbytes NF4); needed for 7B+ models on an 8 GB GPU",
    )
    parser.add_argument(
        "--thinking",
        choices=["adaptive", "disabled", "omit"],
        default="adaptive",
        help="adaptive = model decides how much to think; disabled = answer directly; "
        "omit = don't send the parameter (for models without adaptive thinking)",
    )
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high", "none"],
        default="low",
        help="output_config.effort; 'none' omits it (required for Haiku 4.5)",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=1, help="answers per ordering (estimates P_p(o))")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)
    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
