import subprocess, sys, shlex

def launch(spec: str, port: int):
    # Use fastmcp module entrypoint (cli submodule is not executable as a module)
    cmd = f"{shlex.quote(sys.executable)} -m fastmcp run {spec} --transport streamable-http --host 0.0.0.0 --port {port}"
    print("+", cmd, flush=True)
    return subprocess.Popen(cmd, shell=True)

def main():
    procs = []
    try:
        procs.append(launch("mcp_server.reader_server:mcp", 8001))
        procs.append(launch("mcp_server.metareviewer_server:mcp", 8002))
        procs.append(launch("mcp_server.critic_server:mcp", 8003))
        procs.append(launch("mcp_server.pipeline_server:mcp", 8004))
        print("\nAll MCP servers started: reader=8001, metareviewer=8002, critic=8003, pipeline=8004")
        print("Press Ctrl+C to stop.", flush=True)
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            try:
                p.terminate()
            except Exception:
                pass

if __name__ == "__main__":
    main()
