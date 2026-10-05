from __future__ import annotations
import argparse, json, os, time, re
from collections import Counter
from typing import Dict, Any

from agents.paper_reviewer.agents import make_reader, make_metareviewer, make_critic
from agents.shared import tools

_STOP = {"a","an","and","the","of","for","to","in","on","with","by","is","are","as","that","this",
    "we","they","it","from","at","be","or","if","into","via","our","their","you","your","using",
    "these","those","can","will","may","might","have","has","had","but","not","than","such"}

def _keywords(text: str, top_k: int = 8) -> str:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", (text or "").lower())
    freq = Counter(w for w in words if w not in _STOP and len(w) > 2)
    return " ".join([w for w, _ in freq.most_common(top_k)]) or "research paper"

def run_pipeline(pdf: str | None = None) -> Dict[str, Any]:
    t0 = time.time()
    pdf_path = pdf or "data/sample_papers/sample1.pdf"

    pdf_text = tools.pdf_extract(pdf_path, max_pages=12)

    reader = make_reader(pdf_path).run(
        "Extract raw text from this PDF. Keep the text faithful to the document so downstream agents stay grounded."
    )

    query = _keywords(pdf_text, 8)
    evidence = pdf_text[:8000]

    meta = make_metareviewer(query=query).run(
        "Given the following PDF text, draft novelty/method/results bullets and cite 1–3 related works:\n"
        + evidence
    )

    critic = make_critic(query=query).run(
        "Write a student‑friendly review of THIS PDF only (not generic). "
        "Use the text below as main evidence and include 3 Takeaways + 2 Readings:\n"
        + evidence
    )

    total_tools = sum((m.metrics.get("tool_calls", 0) if hasattr(m, "metrics") else 0) for m in [reader, meta, critic])

    result = {
        "reader": reader.metrics,
        "metareviewer": meta.metrics,
        "critic": critic.metrics,
        "final": critic.final,
        "citations": [c.model_dump() for c in (critic.citations or [])],
        "tool_call_count": total_tools,
        "latency_ms_total": int((time.time() - t0) * 1000),
        "arxiv_query": query,
        "pdf_used": pdf_path,
    }
    outdir = "data/reviews"; os.makedirs(outdir, exist_ok=True)
    base = os.path.basename(pdf_path).rsplit(".", 1)[0]
    with open(os.path.join(outdir, f"{base}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="data/sample_papers/sample1.pdf")
    args = ap.parse_args()
    print(json.dumps(run_pipeline(args.pdf), indent=2, ensure_ascii=False))
