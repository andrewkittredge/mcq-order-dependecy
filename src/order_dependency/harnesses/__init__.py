"""Answer backends. Each exposes ``async answer(mcq, perm_index, perm) -> Trial``.

* ``claude_answerer.ClaudeAnswerer`` - Anthropic Messages API, parses the sampled letter.
* ``local_answerer.LocalAnswerer`` - local Hugging Face model, argmax of the option-letter logits.
  Imports ``torch`` at module level, so ``llm_config`` only imports it when the ``hf`` backend is selected.
"""
