from __future__ import annotations
import os, sys
# add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try both import locations for compatibility
try:
    from fastmcp import FastMCP
except Exception:
    from mcp.server.fastmcp import FastMCP

from agents.paper_reviewer.agents import make_critic

mcp = FastMCP(name="critic-server", json_response=True)

@mcp.tool()
def run_agent(query: str = "Student-friendly review (3 Takeaways + 2 Readings)") -> dict:
    agent = make_critic(query=query)
    res = agent.run(query)
    return {"final": res.final, "citations": [c.model_dump() for c in res.citations], "metrics": res.metrics}
