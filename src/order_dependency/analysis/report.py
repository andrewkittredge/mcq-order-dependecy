"""Write the human-readable report: ``report.md`` rendered from a Jinja2 template."""

from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader

from order_dependency.analysis.aggregate import aggregate
from order_dependency.analysis.summarize import summarize
from order_dependency.harnesses.parse_reply import LABELS
from order_dependency.runner.experiment_run import ExperimentRun

Summary = dict[str, Any]

# trim_blocks/lstrip_blocks drop the newline and indentation around `{% %}` lines, so loops in the
# template add no blank lines of their own.
TEMPLATES = Environment(loader=PackageLoader("order_dependency.analysis"), trim_blocks=True, lstrip_blocks=True)


def write_report(run: ExperimentRun, out_dir: Path) -> Summary:
    """Compute metrics for a run and write the report.

    Args:
        run: A completed run, fresh from ``runner.run_experiment`` or rebuilt from a
            previous ``results.json`` with ``ExperimentRun.model_validate_json``.
        out_dir: Directory to write into; created if missing.

    Returns:
        The summary dict from ``metrics.summarize``. Side effect: writes ``report.md``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_question = list(aggregate(run.questions, run.trials))
    summary = summarize(run.questions, run.trials, per_question)
    template = TEMPLATES.get_template("report.md.j2")
    markdown = template.render(run=run, s=summary, labels=LABELS)
    (out_dir / "report.md").write_text(markdown, encoding="utf-8")
    return summary
