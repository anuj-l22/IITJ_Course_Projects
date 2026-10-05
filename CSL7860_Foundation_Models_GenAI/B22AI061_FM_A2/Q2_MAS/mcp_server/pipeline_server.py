from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from fastmcp import FastMCP
except Exception:
    from mcp.server.fastmcp import FastMCP

from agents.paper_reviewer.orchestrator import run_pipeline

mcp = FastMCP(name="pipeline-server", json_response=True)

@mcp.tool(name="analyze_paper")
def analyze_paper(pdf_path: str) -> dict:
    """Run the full multi-agent pipeline as one MCP call."""
    return run_pipeline(pdf_path)
