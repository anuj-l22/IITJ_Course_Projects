# Research Note: Multi‑Agent Research Paper Reviewer

**Objective.** Build a lightweight multi‑agent system that reviews academic PDFs and returns concise, student‑friendly summaries with actionable *Takeaways* and *Readings*.

---

## 1. Setup & Implementation

- **Language/stack:** Python 3.10+, minimal dependencies. Orchestration in `agents/paper_reviewer/orchestrator.py`.  
- **Agents:** Reader (PDF extract), MetaReviewer (structure + related work), Critic (student‑friendly edit).  
- **Tools:** `pdf_extract` (pypdf/pdfplumber), `search_arxiv` (arxiv client/HTTP), `save_json` (file I/O).  
- **LLM:** Pluggable via LiteLLM; choose a cost‑effective model (e.g., `gemini/gemini-2.5-flash`, `gpt-4o-mini`, or local).  
- **MCP:** FastAPI + fastmcp for per‑agent servers, plus a unified pipeline endpoint.

### Data

- Sample PDFs under `data/sample_papers/` (now 4 distinct mini PDFs for eval variety).  
- For broader runs, use the ArXiv metadata/API (see links in README).

---

## 2. Evaluation Protocol

We provide a programmatic harness in `eval/` that runs **≥6 cases** and computes the following:

- **Success rate**: proportion of cases passing all constraints (*Takeaways ≥3*, *Readings ≥2*, length bounds).  
- **Constraint violations**: count of failed checks per case.  
- **End‑to‑end latency (ms)**: measured around the full pipeline.  
- **Tool‑call count**: aggregated across agents by the orchestrator.

### Repro command

```bash
make eval
# Or explicitly:
python -m eval.make_cases --min 6
python eval/run_eval.py --cases eval/cases --out eval/out
python eval/compute_metrics.py --log eval/out/results.jsonl --csv eval/out/metrics.csv --plot eval/out/latency.png
```

### Case schema

```json
{ "id": "paper_case_01", "pdf_path": "data/sample_papers/sample1.pdf" }
```

### Metrics artifacts

- `eval/out/results.jsonl` — raw per‑case result (includes constraints field).  
- `eval/out/metrics.csv` — flattened metrics table.  
- `eval/out/latency.png` — histogram of end‑to‑end latency.

> The harness includes a **graceful fallback** path when the full agent stack is not importable (e.g., CI without API keys). In regular usage it runs the actual pipeline.

---

## 3. Results (latest run)

- **Success rate:** 66.67% (4/6) with `gemini/gemini-2.5-flash`; two cases failed on missing Takeaways (now mitigated with output post-checks).  
- **Avg latency:** 75,390 ms end‑to‑end (LLM + PDF parsing).  
- **Avg tool calls:** 3.0 per case (one per agent).  
- **Constraint violations:** 2 total across the six cases.  
- **Figure:** `eval/out/latency.png` shows the latency spread across cases.

---

## 4. Design Insights

1. **Role separation reduces prompt bloat.** The Reader focuses on extraction, MetaReviewer on structure, Critic on pedagogy—leading to shorter prompts and more predictable outputs.  
2. **Grounding via “just enough” retrieval.** A tiny related‑work lookup nudges the summary toward citations without bloating latency.  
3. **Strict output contracts help grading.** Enforcing *Takeaways*/*Readings* lets us write deterministic checks and compare runs across models.  
4. **Fail‑soft tools preserve UX.** When a tool fails, the agent still returns a valid answer with degraded grounding—avoiding dead‑ends.

---

## 5. Threats to Validity & Future Work

- **PDF quality variance.** Scanned PDFs with complex math degrade extraction quality. Use OCR where needed.  
- **Model variability.** Smaller LLMs can fluctuate; add few‑shot exemplars or sampling controls.  
- **Domain drift.** ArXiv categories evolve; consider simple category filters per query.  
- **Broader tasks.** The same scaffolding extends to *Academic Copilot* and *News Fact‑Checker* by swapping tools and agent prompts.

---

## 6. Key Takeaways

- A minimal, tool‑using multi‑agent pipeline is sufficient for student‑oriented reviews.  
- Programmatic evaluation with hard contracts yields fast feedback and comparable metrics.  
- MCP deployment enables modular reuse of agents in other clients and assignments.
