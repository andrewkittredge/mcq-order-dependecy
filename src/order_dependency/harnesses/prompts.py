"""System prompts for the Claude backend, one per question kind.

Each tells the model to reply with nothing but the answer, so the free-text reply can
be parsed without a structured-output constraint that would change the decoding path
under study. The local backend reads logits instead and sends no system prompt.
"""

SYSTEM_PROMPT = (
    "You are taking a multiple-choice test. Exactly one option is correct. "
    "Reply with only the letter of the best option and nothing else."
)

DOCUMENT_SYSTEM_PROMPT = (
    "You are answering a multiple-choice question about the document provided. Exactly one option is correct. "
    "Reply with only the letter of the best option and nothing else."
)

EXTRACTION_SYSTEM_PROMPT = (
    "You are extracting a figure from an excerpt of a financial document. "
    "Reply with only the requested figure, written as it appears in the excerpt, and nothing else."
)
