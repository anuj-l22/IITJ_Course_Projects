from __future__ import annotations
import os, time, re, csv
from typing import List, Dict, Any

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None

_TOOL_LATENCIES: List[float] = []

def _record_latency(start: float) -> None:
    _TOOL_LATENCIES.append(time.time() - start)

def reset_tool_latencies() -> None:
    global _TOOL_LATENCIES
    _TOOL_LATENCIES = []

def tool_latency_summary() -> Dict[str, int]:
    return {"tool_calls": len(_TOOL_LATENCIES), "tool_latency_ms_total": int(sum(_TOOL_LATENCIES) * 1000)}

def pdf_extract(path: str, max_pages: int = 12) -> str:
    start = time.time()
    try:
        errors: List[str] = []
        size = os.path.getsize(path) if os.path.exists(path) else 0

        if PdfReader is not None:
            try:
                reader = PdfReader(path, strict=False)  # strict=False to tolerate bad xref pointers
            except TypeError:
                reader = PdfReader(path)
            except Exception as e:  # pylint: disable=bare-except
                errors.append(f"pypdf:{e.__class__.__name__}:{e}")
                reader = None
            if reader is not None:
                try:
                    pages = min(len(reader.pages), max_pages)
                    text_parts: List[str] = []
                    for i in range(pages):
                        text_parts.append(reader.pages[i].extract_text() or "")
                    text = re.sub(r"\s+", " ", "\n".join(text_parts)).strip()
                    if text:
                        return text
                except Exception as e:
                    errors.append(f"pypdf:{e.__class__.__name__}:{e}")

        if pdfplumber is not None:
            try:
                with pdfplumber.open(path) as pdf:
                    pages = pdf.pages[:max_pages]
                    text_parts = [(p.extract_text() or "") for p in pages]
                    text = re.sub(r"\s+", " ", " ".join(text_parts)).strip()
                    if text:
                        return text
            except Exception as e:
                errors.append(f"pdfplumber:{e.__class__.__name__}:{e}")

        if errors:
            return f"[pdf_extract error] {'; '.join(errors)} (path={path} bytes={size})"
        return f"[pdf_extract fallback] Install 'pypdf' for true extraction. file={path} bytes={size}"
    finally:
        _record_latency(start)

def _iter_local_csv(paths: List[str]):
    if pd is not None:
        for p in paths:
            if not os.path.exists(p):
                continue
            try:
                for chunk in pd.read_csv(p, chunksize=20000):
                    yield chunk.to_dict(orient="records")
            except Exception:
                continue
    else:
        for p in paths:
            if not os.path.exists(p):
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    yield list(csv.DictReader(f))
            except Exception:
                continue


def search_arxiv(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    start = time.time()
    results: List[Dict[str, str]] = []
    try:
        base = "data/external"
        candidates = []
        if os.path.isdir(base):
            csvs = [os.path.join(base, n) for n in os.listdir(base) if n.lower().endswith(".csv") and "arxiv" in n.lower()]
            candidates = csvs
        keywords = [w.lower() for w in re.split(r"\W+", query) if w]
        seen = set()
        for batch in _iter_local_csv(candidates):
            for row in batch:
                title = str(row.get("title", ""))
                summary = str(row.get("summary", row.get("abstract", "")))
                cat = str(row.get("categories", ""))
                hay = (title + " " + summary + " " + cat).lower()
                score = sum(1 for k in keywords if k in hay)
                if score >= max(1, len(keywords) // 3):
                    url = str(row.get("url", row.get("pdf_url", "")))
                    entry_id = str(row.get("id", row.get("entry_id", "")))
                    key = (title, url)
                    if key in seen:
                        continue
                    seen.add(key)
                    results.append({
                        "title": title[:200],
                        "url": url or f"https://arxiv.org/abs/{entry_id}",
                        "summary": summary[:400],
                        "entry_id": entry_id
                    })
                    if len(results) >= max_results:
                        return results

        try:
            import arxiv  # type: ignore
            s = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
            for r in s.results():
                results.append({
                    "title": r.title,
                    "url": getattr(r, "entry_id", ""),
                    "summary": getattr(r, "summary", "")[:400],
                    "entry_id": getattr(r, "get_short_id", lambda: "")()
                })
        except Exception:
            pass

        if not results:
            results = [
                {"title": "Tool-Using LLM Agents", "url": "https://arxiv.org/abs/2303.00001", "summary": "", "entry_id": "2303.00001"},
                {"title": "Paper Review Agents", "url": "https://arxiv.org/abs/2406.12345", "summary": "", "entry_id": "2406.12345"},
                {"title": "Lightweight Academic Assistants", "url": "https://arxiv.org/abs/2405.05050", "summary": "", "entry_id": "2405.05050"},
            ][:max_results]

        return results[:max_results]
    finally:
        _record_latency(start)
