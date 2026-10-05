.PHONY: ui mcp mcp-api eval fmt lint

# Launch the Streamlit UI for the paper reviewer
ui:
	streamlit run agents/paper_reviewer/ui_streamlit.py

# Run all MCP agent servers (Reader, MetaReviewer, Critic)
mcp:
	python -m mcp_server.run_all

# Optional: run the unified FastAPI app (HTTP gateway)
mcp-api:
	uvicorn mcp_server.app:app --host 0.0.0.0 --port 8000 --reload

# End-to-end evaluation harness
eval:
	python -m eval.make_cases --min 6 && \
	python eval/run_eval.py --cases eval/cases --out eval/out && \
	python eval/compute_metrics.py --log eval/out/results.jsonl --csv eval/out/metrics.csv --plot eval/out/latency.png && \
	ls -lh eval/out

fmt:
	python -m pip install ruff black && ruff format . && black .

lint:
	python -m pip install ruff && ruff check .
