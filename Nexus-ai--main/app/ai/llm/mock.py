"""Deterministic, offline mock LLM provider.

Grounds its answer strictly in the retrieved context passed via the system
message so that RAG behaviour (citations, "no answer" fallback) is meaningful in
tests and local development without any API key or network access.
"""

import re
from collections.abc import AsyncIterator, Sequence

from app.ai.llm.base import LLMMessage, LLMResult

_NO_CONTEXT = "(none)"
_FALLBACK = "I couldn't find relevant information in the knowledge base to answer that."
_WORD_RE = re.compile(r"\S+")


def _count(text: str) -> int:
    return len(_WORD_RE.findall(text))


class MockLLMProvider:
    """Rule-based, context-grounded chat completion."""

    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    
    def _answer(self, messages: Sequence[LLMMessage]) -> str:
        system = next(
            (m.content for m in messages if m.role == "system"), ""
        )
        question = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )

        if "[1]" not in system or _NO_CONTEXT in system:
            return _FALLBACK

        q = question.lower()
        context = system.lower()

        # Answer questions about the sample support policy.
        if "working hours" in q or "support team's hours" in q:
            if "9:00 am to 5:00 pm" in context:
                return (
                    "The Demo Support Team works Monday through Friday, "
                    "9:00 AM to 5:00 PM. [1]"
                )

        if (
            "how quickly" in q
            or "response time" in q
            or "how soon" in q
            or "respond" in q
        ):
            if "one business day" in context:
                return (
                    "The Demo Support Team aims to respond within "
                    "one business day. [1]"
                )

        if (
            "password" in q
            or "api key" in q
            or "api keys" in q
            or "secrets" in q
        ):
            if "passwords" in context and "api keys" in context:
                return (
                    "Users should never include passwords, API keys, "
                    "or other secrets in support requests. [1]"
                )

        if "workspace" in q:
            if "organizing documents and conversations" in context:
                return (
                    "A workspace is a separate area for organizing "
                    "documents and conversations. [1]"
                )

        # Default response for other questions.
        match = re.search(r"\[1\]\s*(.+)", system)
        snippet = (match.group(1).strip() if match else "")[:280]
        return (
            f'Based on the knowledge base, regarding '
            f'"{question.strip()}": {snippet} [1]'
        )
        
    async def complete(self, messages: Sequence[LLMMessage]) -> LLMResult:
        text = self._answer(messages)
        prompt_tokens = sum(_count(m.content) for m in messages)
        return LLMResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=_count(text),
        )

    async def stream(self, messages: Sequence[LLMMessage]) -> AsyncIterator[str]:
        text = self._answer(messages)
        for token in text.split(" "):
            yield token + " "

