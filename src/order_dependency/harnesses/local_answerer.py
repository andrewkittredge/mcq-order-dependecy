"""Answer MCQs with a local Hugging Face causal LM by reading option-letter logits.

This is the evaluation protocol of Zheng et al. (2024) for open-weight models: one
forward pass per prompt, then compare the next-token logits of the option IDs
(``A``/``B``/``C``/``D``) after ``Answer:``. No text is generated, so the answer is
deterministic.

``torch`` and ``transformers`` come from the optional ``hf`` dependency group and are
imported at module level, so this module must only be imported when the local backend
is selected (``llm_config`` imports it inside the ``hf`` branch).

The forward pass is synchronous and would otherwise run every queued job before any
completion callback, leaving the tqdm bar at ``1/N`` until the end. ``answer`` therefore
takes a one-slot semaphore and yields to the event loop (``asyncio.sleep(0)``) before
scoring, so completions are processed between prompts and the bar advances.
"""

import asyncio
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase
from transformers.modeling_outputs import CausalLMOutputWithPast

from order_dependency.harnesses.build_prompt import build_prompt
from order_dependency.harnesses.parse_reply import LABELS
from order_dependency.mcq import MCQ
from order_dependency.runner.permutations import Permutation
from order_dependency.trial import Trial

if TYPE_CHECKING:  # llm_config imports this module, so the type is only needed statically
    from order_dependency.runner.llm_config import LLMConfig

# The paper's 0-shot MMLU header (Appendix A); base models get no system prompt.
MMLU_HEADER = "The following are multiple choice questions (with answers) about {topic}.\n\n"


class LocalAnswerer:
    """Local-model counterpart of ``ClaudeAnswerer`` with the same ``answer`` signature.

    Attributes:
        config: Run settings; ``model`` is a Hugging Face repo id, loaded in 4-bit when
            ``config.quantized`` is set, else fp16.
        tokenizer: The model's tokenizer.
        model: The causal LM, sharded across available GPUs by ``device_map="auto"``.
        label_ids: Token id of each option letter when it follows ``Answer:``.
    """

    def __init__(self, config: "LLMConfig"):
        """Load the model once.

        Args:
            config: Run settings.
        """
        quantization = None
        if config.quantized:
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        self.config = config
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(config.model)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,  # pyright: ignore[reportArgumentType]
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization,
        )
        self.model.eval()
        # Tokenise "Answer: X" and keep the token after "Answer:" - usually the " X" variant.
        stem_length = len(self.tokenizer("Answer:").input_ids)
        self.label_ids = [self._label_id(label, stem_length) for label in LABELS]
        self._sem = asyncio.Semaphore(1)

    def _label_id(self, label: str, stem_length: int) -> int:
        """Token id the tokenizer gives ``label`` directly after ``Answer:``.

        Args:
            label: An option letter.
            stem_length: Number of tokens in ``"Answer:"`` alone.

        Returns:
            The token id.
        """
        ids = self.tokenizer(f"Answer: {label}").input_ids
        return ids[stem_length]

    async def answer(self, mcq: MCQ, perm_index: int, perm: Permutation) -> Trial:
        """Score one question under one ordering from the option-letter logits.

        Args:
            mcq: The question.
            perm_index: Index of ``perm`` in the question's permutation list.
            perm: Ordering of the options to present.

        Returns:
            The trial, with ``position`` the argmax over the ``k`` option-letter logits.
        """
        prompt = build_prompt(mcq, perm)
        if mcq.topic:
            header = MMLU_HEADER.format(topic=mcq.topic.replace("_", " "))
            prompt = header + prompt
        async with self._sem:
            await asyncio.sleep(0)  # let finished trials update the progress bar before blocking on the GPU
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                # Last-position logits only, no KV cache: both grow with prompt length and nothing reads them.
                output: CausalLMOutputWithPast = self.model(**inputs, logits_to_keep=1, use_cache=False)
        next_token_logits = output.logits[0, -1]  # pyright: ignore[reportOptionalSubscript]
        letter_logits = next_token_logits[self.label_ids[: mcq.k]]
        position = int(letter_logits.argmax())
        canonical = perm[position]
        return Trial(
            question_id=mcq.id,
            perm_index=perm_index,
            perm=perm,
            position=position,
            canonical=canonical,
            correct=canonical == mcq.answer,
            reply=LABELS[position],
        )
