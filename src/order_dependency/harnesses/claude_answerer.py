"""Ask Claude to answer one rendered MCQ."""

import asyncio
from typing import TYPE_CHECKING

import anthropic

from order_dependency.harnesses.build_prompt import build_prompt
from order_dependency.harnesses.parse_reply import parse_reply
from order_dependency.harnesses.prompts import DOCUMENT_SYSTEM_PROMPT, EXTRACTION_SYSTEM_PROMPT, SYSTEM_PROMPT
from order_dependency.mcq import MCQ
from order_dependency.runner.permutations import Permutation
from order_dependency.trial import Trial

if TYPE_CHECKING:  # llm_config imports this module, so the type is only needed statically
    from order_dependency.runner.llm_config import LLMConfig


class ClaudeAnswerer:
    """Thin async wrapper around the Anthropic Messages API with bounded concurrency.

    One API call per (question, ordering). Refusals, max-token cut-offs and API
    errors are recorded as unanswered rather than retried on another model, so a
    run measures exactly one model.

    Attributes:
        config: Request settings shared by every call.
        client: The async Anthropic client. Credentials come from
            ``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile.
    """

    def __init__(
        self,
        config: "LLMConfig",
    ):
        """Create an answerer.

        Args:
            config: Request settings.
        """
        self.config = config
        self.client = anthropic.AsyncAnthropic(max_retries=5)
        self._sem = asyncio.Semaphore(config.concurrency)

    async def answer(self, mcq: MCQ, perm_index: int, perm: Permutation) -> Trial:
        """Ask the model one question under one ordering and record the outcome.

        Args:
            mcq: The question.
            perm_index: Index of ``perm`` in the question's permutation list.
            perm: Ordering of the options to present.

        Returns:
            The trial. API and connection failures leave it unanswered rather than
            raising, so a single bad request does not abort a run.
        """
        response = None
        choice_system = DOCUMENT_SYSTEM_PROMPT if mcq.context else SYSTEM_PROMPT
        system = EXTRACTION_SYSTEM_PROMPT if mcq.is_extraction else choice_system
        async with self._sem:  # cap in-flight requests to stay under rate limits
            try:
                response = await self.client.messages.create(
                    system=system,
                    messages=[{"role": "user", "content": build_prompt(mcq, perm)}],
                    **self.config.request_kwargs(),
                )
            except (anthropic.APIStatusError, anthropic.APIConnectionError):  # 4xx, 5xx after retries, transport
                pass

        text = "".join(b.text for b in response.content if b.type == "text") if response else ""
        # Only a naturally finished turn counts: a refusal or a max_tokens cut-off means
        # the model never committed to an answer.
        finished = response is not None and response.stop_reason == "end_turn"
        position = parse_reply(mcq, perm, text) if finished else None
        canonical = None if position is None else perm[position]
        correct = None if canonical is None else canonical == mcq.answer
        return Trial(
            question_id=mcq.id,
            perm_index=perm_index,
            perm=perm,
            position=position,
            canonical=canonical,
            correct=correct,
            reply=text,
        )
