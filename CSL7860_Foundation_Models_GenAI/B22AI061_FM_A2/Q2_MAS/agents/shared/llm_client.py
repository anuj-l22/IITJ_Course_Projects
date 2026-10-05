from __future__ import annotations
import os
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

USE_MOCK = os.getenv("MOCK", "1") == "1"

def _has_any_key() -> bool:
    keys = ["OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "AZURE_API_KEY"]
    return any(os.getenv(k) for k in keys)

class LLMClient:
    def __init__(self, model: str | None = None):
        self.model = model or os.getenv("MODEL", "gemini/gemini-2.5-flash")
        self.use_mock = USE_MOCK or not _has_any_key()

    def _mock_complete(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        user = next((m for m in reversed(messages) if m.get("role") == "user"), {"content": ""})
        context = user.get("content", "")[:1200]
        return (
            "Novelty: Modular multi-agent pipeline grounded in the provided PDF context.\n"
            "Method: Reader extracts text; MetaReviewer and Critic use it plus arXiv retrieval.\n"
            "Results: Deterministic MOCK summary.\n"
            "Takeaways:\n"
            "1) Grounding summaries in PDF text changes outputs per document.\n"
            "2) Local arXiv CSV speeds retrieval.\n"
            "3) Tool-call metrics aid debugging.\n"
            "Readings:\n"
            "- arXiv:1506.0454 — Seq2Seq Attention\n"
            "- arXiv:2306.00981 — Toolformer\n"
            f"\nContext preview: {context}"
        )

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        if self.use_mock:
            return self._mock_complete(messages, temperature=temperature)
        try:
            from litellm import completion  # type: ignore
        except Exception:
            user = next((m for m in reversed(messages) if m.get("role") == "user"), {"content": ""})
            return "[LiteLLM missing] " + user.get("content","")[:400]

        @retry(wait=wait_exponential(multiplier=1, min=1, max=6), stop=stop_after_attempt(3))
        def _call():
            resp = completion(model=self.model, messages=messages, temperature=temperature)
            return resp.choices[0].message["content"]

        try:
            return _call()
        except Exception as e:
            return "[LLM fallback to MOCK due to error: %s] %s" % (e.__class__.__name__, self._mock_complete(messages))
