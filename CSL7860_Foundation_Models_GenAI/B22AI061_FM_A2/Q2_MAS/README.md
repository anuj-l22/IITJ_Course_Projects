# Multi-Agent Research Paper Reviewer with MCP

**Course:** CSL7860 – Foundation Models and Generative AI
**Author:** Anuj

A **multi-agent LLM system** that reviews academic papers and produces student-friendly summaries. Each agent runs as its own **Model Context Protocol (MCP) server**, and a Python pipeline orchestrates them. The system takes a PDF and returns a structured summary covering novelty, method, results, takeaways and further reading, all checked against a fixed output contract.

## Highlights
- **100% success rate** on 6 programmatic evaluation cases, with **zero constraint violations** and an average of **3 tool calls per case**.
- Three cooperating agents with clear roles, each deployed as an MCP server.
- Includes an interactive **Streamlit** demo and an automated evaluation harness that reports success rate, violations, latency and tool-call counts.

## Architecture
- **Reader:** parses the PDF and extracts the title, abstract, sections and figures into a normalised JSON outline.
- **MetaReviewer:** fetches context from the arXiv API, resolves references and drafts the student-facing summary.
- **Critic:** checks that the draft covers the core contribution, includes at least one verifiable citation and makes no speculative claims. It then validates the JSON schema and publishes the final output.
- **Contracts:** at least one verifiable citation, no unsupported claims, output that matches the schema, and logging of every tool call.
- **LLM backend:** pluggable through LiteLLM (for example `gemini-2.5-flash`), with a deterministic mock mode for testing.

## Key Insights
- **Keeping the agents separate matters.** Folding the Critic's checks into the MetaReviewer increased hallucinations and schema violations.
- **Defining the output schema first** made automated success checks simple.
- **Caching parsed PDFs** reduced latency without affecting the constraint checks.

## Skills & Tools
LLM Agents · Model Context Protocol (MCP) · Multi-Agent Orchestration · LiteLLM · arXiv API · JSON Schema · Streamlit · Python
