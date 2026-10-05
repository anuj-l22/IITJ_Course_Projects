# Architecture 

Summary of the **Multi-Agent Research Paper Reviewer**: agent roles, orchestration, schema/contracts, tools, and MCP integration.

---

## Agents
- **Reader** — extract PDF text, normalize whitespace. Tool: `pdf_extract` (pypdf + pdfplumber fallback). Output: raw text buffer.
- **MetaReviewer** — structure Novelty/Method/Results and surface related work. Tool: `search_arxiv` (local CSV or arXiv API). Output: draft bullets + citations.
- **Critic** — edit for students; ensure 3 Takeaways + 2 Readings. Tool: `search_arxiv` (secondary lookup). Output: final review + citations.

## Orchestration
```
PDF -> [Reader] -> [MetaReviewer] -> [Critic] -> Final summary
```
- Implemented in `agents/paper_reviewer/orchestrator.py`.
- Aggregates per-agent metrics and `tool_call_count`, writes `data/reviews/<pdf>.json`.
- Fail-soft: if a tool fails, proceed with degraded context but still emit a valid summary.

## Message schema & contracts
- Message (`agents/shared/schemas.py`): `{role: system|user|assistant|tool, content, tool_calls?, citations?, metrics?}`.
- AgentResult: `{messages: [...], final, citations: [{source, snippet?}], metrics: {agent_latency_ms, tool_calls, ...}}`.
- Hard constraints (also enforced in eval): Takeaways section with ≥3 bullets; Readings section with ≥2 links/IDs; length 256–4096 chars.

## Tools
- `pdf_extract(path, max_pages)` — PDF text extraction with tolerant parsing.
- `search_arxiv(query, max_results)` — related work from local CSVs or arXiv API, returns title/url/entry_id.
- (Optional) `save_json` — persist artifacts for debugging.

## MCP integration
- Per-agent MCP servers: `reader_server.py`, `metareviewer_server.py`, `critic_server.py` (ports 8001–8003).
- Unified pipeline tool: `pipeline_server.py` (port 8004) and FastAPI gateway `app.py` at `/mcp/reviewer/{reader|metareviewer|critic|pipeline}` (port 8000 when using `make mcp-api`).
- `mcp_server/run_all.py` starts the four standalone servers via fastmcp.
- Contract: `analyze_paper({pdf_path}) -> {final, citations, tool_call_count, latency_ms_total}`.

## Observability & evaluation
- Tool latency tracked in `agents/shared/tools.py`; per-agent latency/tool_calls added in `MiniAgent.run()`.
- Eval harness (`eval/run_eval.py`, `eval/compute_metrics.py`) runs ≥6 JSON cases, reports success rate, constraint violations, latency, and tool-call count; plot saved to `eval/out/latency.png`.

## Assumptions / limits
- Linear, synchronous pipeline to minimize cost/complexity.
- Best-effort retrieval: if arXiv is unreachable, stub citations are used to keep outputs valid.
- Intended for local MCP use; wrap with a proxy for remote access if needed.
