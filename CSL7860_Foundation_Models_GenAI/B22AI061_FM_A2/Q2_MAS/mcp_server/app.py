from __future__ import annotations
import contextlib
from starlette.applications import Starlette
from starlette.routing import Mount
from mcp.server.fastmcp import FastMCP

from agents.paper_reviewer.agents import make_reader, make_metareviewer, make_critic
from agents.paper_reviewer.orchestrator import run_pipeline

reader_mcp = FastMCP(name="reader-server", stateless_http=True, json_response=True, streamable_http_path="/")
meta_mcp = FastMCP(name="metareviewer-server", stateless_http=True, json_response=True, streamable_http_path="/")
critic_mcp = FastMCP(name="critic-server", stateless_http=True, json_response=True, streamable_http_path="/")
pipeline_mcp = FastMCP(name="pipeline-server", stateless_http=True, json_response=True, streamable_http_path="/")

@reader_mcp.tool(name="run_agent")
def run_agent_reader(pdf_path: str | None = None, query: str = "Read the paper and prepare raw text.") -> dict:
    agent = make_reader(pdf_path)
    res = agent.run(query)
    return {"final": res.final, "citations": [c.model_dump() for c in res.citations], "metrics": res.metrics}

@meta_mcp.tool(name="run_agent")
def run_agent_meta(query: str = "Draft novelty/method/results + related work") -> dict:
    agent = make_metareviewer(query=query)
    res = agent.run(query)
    return {"final": res.final, "citations": [c.model_dump() for c in res.citations], "metrics": res.metrics}

@critic_mcp.tool(name="run_agent")
def run_agent_critic(query: str = "Student-friendly review (3 Takeaways + 2 Readings)") -> dict:
    agent = make_critic(query=query)
    res = agent.run(query)
    return {"final": res.final, "citations": [c.model_dump() for c in res.citations], "metrics": res.metrics}

@pipeline_mcp.tool(name="analyze_paper")
def analyze_paper(pdf_path: str) -> dict:
    """Run the full multi-agent pipeline as a single MCP call."""
    return run_pipeline(pdf_path)

@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    async with contextlib.AsyncExitStack() as stack:
        await stack.enter_async_context(reader_mcp.session_manager.run())
        await stack.enter_async_context(meta_mcp.session_manager.run())
        await stack.enter_async_context(critic_mcp.session_manager.run())
        await stack.enter_async_context(pipeline_mcp.session_manager.run())
        yield

app = Starlette(
    routes=[
        Mount("/mcp/reviewer/reader", app=reader_mcp.streamable_http_app()),
        Mount("/mcp/reviewer/metareviewer", app=meta_mcp.streamable_http_app()),
        Mount("/mcp/reviewer/critic", app=critic_mcp.streamable_http_app()),
        Mount("/mcp/reviewer/pipeline", app=pipeline_mcp.streamable_http_app()),
    ],
    lifespan=lifespan,
)
