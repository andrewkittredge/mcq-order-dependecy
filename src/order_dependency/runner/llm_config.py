"""Model and request settings for the LLM under test."""

from typing import TYPE_CHECKING

from pydantic.dataclasses import dataclass

from order_dependency.harnesses.claude_answerer import ClaudeAnswerer

if TYPE_CHECKING:  # local_answerer imports torch at module level; only import it when the hf backend is used
    from order_dependency.harnesses.local_answerer import LocalAnswerer


@dataclass(frozen=True)
class LLMConfig:
    """Everything that shapes a request except the question itself.

    Thinking is configurable. Claude Opus 5 thinks by default; ``disabled`` gives a
    "fast, first-impression" answer that is typically more order-sensitive, so the
    two settings form a natural before/after for the *reasoning as mitigation*
    experiment. No temperature is sent: current Claude models reject sampling
    parameters, so repeated samples use the default sampling distribution.

    Attributes:
        backend: ``"claude"`` (Anthropic API) or ``"hf"`` (local Hugging Face model
            scored from its option-letter logits).
        model: Anthropic model id, or Hugging Face repo id for ``hf``.
        quantized: Load the ``hf`` model with 4-bit (bitsandbytes NF4) weights instead of fp16.
        thinking: ``"adaptive"`` (model decides how much to think), ``"disabled"``
            (answer directly), or ``"omit"`` (send no ``thinking`` parameter, for
            models without adaptive thinking).
        effort: ``output_config.effort`` level, or ``None`` to omit the parameter.
        max_tokens: Response token cap. Must leave room for thinking tokens when
            thinking is on, otherwise the reply is cut off before the letter.
        concurrency: Maximum in-flight requests.
    """

    backend: str | None = None
    model: str | None = None
    quantized: bool = False
    thinking: str = "adaptive"
    effort: str | None = "low"
    max_tokens: int = 4096
    concurrency: int = 8

    @property
    def answerer(self) -> "LocalAnswerer | ClaudeAnswerer":
        """Build the backend this config describes.

        Constructs a new instance on each access (loading the model for ``hf``), so
        call it once per run.

        Returns:
            A ``LocalAnswerer`` for ``backend == "hf"``, otherwise a ``ClaudeAnswerer``.
        """
        if self.backend == "hf":
            from order_dependency.harnesses.local_answerer import LocalAnswerer

            return LocalAnswerer(self)
        return ClaudeAnswerer(self)

    def request_kwargs(self) -> dict:
        """Build the model-specific keyword arguments for ``messages.create``.

        Kept in one place so the exact request shape can be recorded in the report.

        Returns:
            A dict with ``model`` and ``max_tokens`` plus, when configured,
            ``thinking`` and ``output_config``.
        """
        kwargs: dict = {"model": self.model, "max_tokens": self.max_tokens}
        if self.thinking == "adaptive":
            kwargs["thinking"] = {"type": "adaptive"}
        elif self.thinking == "disabled":
            kwargs["thinking"] = {"type": "disabled"}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        return kwargs
