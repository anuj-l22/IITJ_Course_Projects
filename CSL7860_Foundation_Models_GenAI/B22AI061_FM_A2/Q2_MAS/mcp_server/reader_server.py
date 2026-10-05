from __future__ import annotations
import os, sys
# add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try both import locations for compatibility
try:
    from fastmcp import FastMCP
except Exception:
    from mcp.server.fastmcp import FastMCP

from agents.paper_reviewer.agents import make_reader

mcp = FastMCP(name="reader-server", json_response=True)

@mcp.tool()
def run_agent(pdf_path: str | None = None, query: str = "Read the paper and prepare raw text.") -> dict:
    agent = make_reader(pdf_path)
    res = agent.run(query)
    return {"final": res.final, "citations": [c.model_dump() for c in res.citations], "metrics": res.metrics}
