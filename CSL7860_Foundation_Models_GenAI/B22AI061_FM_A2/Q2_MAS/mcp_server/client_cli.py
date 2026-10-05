from __future__ import annotations
import asyncio, json, argparse
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def call_tool(base: str, args: dict):
    base = base if base.endswith("/") else base + "/"
    async with streamablehttp_client(base) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("run_agent", args)
            print(json.dumps(result.dict(), indent=2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8001/", help="e.g. http://localhost:8001/")
    ap.add_argument("--args", default="{}", help='JSON args, e.g. {"pdf_path": "data/sample_papers/sample1.pdf"}')
    a = ap.parse_args()
    asyncio.run(call_tool(a.base, json.loads(a.args)))

if __name__ == "__main__":
    main()
