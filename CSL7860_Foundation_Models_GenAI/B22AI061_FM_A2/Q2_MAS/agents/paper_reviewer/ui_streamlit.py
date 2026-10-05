import os, sys, glob
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.paper_reviewer.orchestrator import run_pipeline

st.set_page_config(page_title="Multi-Agent Research Paper Reviewer", layout="wide")
st.title("📄 Multi‑Agent Research Paper Reviewer")

with st.sidebar:
    st.header("LLM Setup (Optional)")
    st.caption("Leave empty for deterministic MOCK mode.")
    model = st.text_input("MODEL (LiteLLM)", value=os.environ.get("MODEL", "gemini/gemini-2.5-flash"))
    openai = st.text_input("OPENAI_API_KEY", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
    gemini = st.text_input("GEMINI_API_KEY", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
    google = st.text_input("GOOGLE_API_KEY", type="password", value=os.environ.get("GOOGLE_API_KEY", ""))
    groq = st.text_input("GROQ_API_KEY", type="password", value=os.environ.get("GROQ_API_KEY", ""))
    if model: os.environ["MODEL"] = model
    if openai: os.environ["OPENAI_API_KEY"] = openai
    if gemini: os.environ["GEMINI_API_KEY"] = gemini
    if google: os.environ["GOOGLE_API_KEY"] = google
    if groq: os.environ["GROQ_API_KEY"] = groq
    mock = st.checkbox("Use MOCK summarizer", value=True if not (openai or gemini or google or groq) else False)
    os.environ["MOCK"] = "1" if mock else "0"

st.write("Select or upload a PDF to review. Results will be saved under `data/reviews/`.")

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded is not None:
    upath = os.path.join("data", "uploads", uploaded.name)
    os.makedirs(os.path.dirname(upath), exist_ok=True)
    with open(upath, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Uploaded to {upath}")

choices = sorted(glob.glob("data/uploads/*.pdf") + glob.glob("data/sample_papers/*.pdf"))
if not choices:
    st.warning("No PDFs found. Please upload or place PDFs under data/sample_papers/.")
idx = 0 if choices else None
pdf = st.selectbox("Select a PDF", choices, index=idx)

if st.button("▶ Run Reviewer", disabled=not bool(pdf)):
    with st.spinner("Reviewing..."):
        result = run_pipeline(pdf)
    st.session_state["result"] = result

if "result" in st.session_state:
    result = st.session_state["result"]
    st.subheader("Student‑Friendly Summary")
    st.write(result.get("final", "(empty)"))

    st.subheader("Citations (from arXiv tool)")
    cites = result.get("citations", [])
    if cites:
        for c in cites:
            st.markdown(f"- [{c.get('source')}]({c.get('source')})")
    else:
        st.write("No citations captured.")

    st.subheader("Metrics")
    st.json({k:v for k,v in result.items() if k not in ["final", "citations"]})
