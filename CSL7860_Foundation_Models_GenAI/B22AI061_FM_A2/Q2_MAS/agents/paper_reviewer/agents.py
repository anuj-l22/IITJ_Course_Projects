from __future__ import annotations
from typing import Optional
from agents.shared.agent_base import MiniAgent
from agents.shared import tools

def make_reader(pdf_path: Optional[str] = None) -> MiniAgent:
    path = pdf_path or "data/sample_papers/sample1.pdf"
    return MiniAgent(
        name="Reader",
        system_prompt=(
            "Extract and lightly clean text from the PDF; identify Abstract/Intro if present.\n"
            "Return raw text for downstream agents."
        ),
        toolmap={"pdf_extract": (lambda: tools.pdf_extract(path))}
    )

def make_metareviewer(query: str = "multi-agent research paper summarization") -> MiniAgent:
    return MiniAgent(
        name="MetaReviewer",
        system_prompt=(
            "Given reader context, draft bullets for Novelty, Method, and Results.\n"
            "Also retrieve 1–3 related arXiv items relevant to the topic/keywords."
        ),
        toolmap={"search_arxiv": (lambda: tools.search_arxiv(query, max_results=3))}
    )

def make_critic(query: str = "student learning agents") -> MiniAgent:
    return MiniAgent(
        name="Critic",
        system_prompt=(
            "Condense to a student‑friendly summary with 3 Takeaways and 2 follow‑up readings.\n"
            "Ensure clarity and avoid jargon where possible."
        ),
        toolmap={"search_arxiv": (lambda: tools.search_arxiv(query, max_results=2))}
    )
