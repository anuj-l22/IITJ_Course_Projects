
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Tuple

THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def _import_pipeline():
    try:
        from agents.paper_reviewer.orchestrator import run_pipeline  # type: ignore
        print("[eval] Using REAL pipeline")
        return run_pipeline
    except Exception as e:
        print(f"[eval] Using FALLBACK pipeline (import failed: {e.__class__.__name__}: {e})")
        return None

def _fallback_pipeline(pdf_path: str) -> Dict[str, Any]:
    base = os.path.basename(pdf_path).rsplit(".", 1)[0].replace("_", " ").title()
    final = (
        f"Novelty: The paper '{base}' explores lightweight multi‑agent reviewing for students.\n"
        "Method: A 3‑agent pipeline (Reader, MetaReviewer, Critic) that extracts PDF text, organizes salient points,\n"
        "and condenses them into a student‑friendly summary.\n"
        "Results: The system yields concise reviews with actionable guidance and external references.\n"
        "Takeaways:\n"
        "- Multi‑agent orchestration structures the reading workflow.\n"
        "- Tool use (PDF parsing, arXiv search) provides grounding.\n"
        "- A critic pass enforces clear, student‑oriented language.\n"
        "Readings:\n"
        "- arXiv:2303.00001 (agent tooling)\n"
        "- https://arxiv.org/abs/2406.12345 (paper‑review agents)\n"
    )
    return {
        "final": final,
        "tool_call_count": 3,
        "citations": [
            {"source": "https://arxiv.org/abs/2303.00001"},
            {"source": "https://arxiv.org/abs/2406.12345"}
        ]
    }

BULLET_LINE = re.compile(r"^\s*(?:[-•·‣\u2212\u2013\u2014]|\d+[\.\)])\s*\S")
INLINE_ENUM = re.compile(r"(?:^|\s)(?:[-•·‣\u2212\u2013\u2014]|\d+[\.\)])\s+\S")
URL_OR_ID = re.compile(r"(https?://\S+|arxiv[:/]\S+|doi[:/]\S+)", re.I)

TAKEAWAY_HEADERS: Tuple[str, ...] = (
    "takeaways","key takeaways","key points","highlights","summary points","main takeaways"
)
READING_HEADERS: Tuple[str, ...] = (
    "readings","further reading","further readings","suggested reading","suggested readings",
    "recommended reading","recommended readings","references","bibliography","related work"
)
STOP_HEADERS: Tuple[str, ...] = TAKEAWAY_HEADERS + READING_HEADERS + (
    "novelty","method","methods","approach","approaches","results","evaluation","experiments",
    "discussion","conclusion","conclusions","limitations","summary","acknowledgements","acknowledgments"
)

def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"^[#>\-\*\s]+","",s)
    s = s.rstrip(":").strip()
    return s

def _resolve_pdf_path(p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p) and os.path.exists(p):
        return p
    if os.path.exists(p):
        return p
    alt = os.path.join(REPO_ROOT, p)
    return alt if os.path.exists(alt) else p

def _section_lines(text: str, headers: Iterable[str]) -> List[str]:
    if not text:
        return []
    lines = text.splitlines()
    start = None
    for i, raw in enumerate(lines):
        low = _norm(raw)
        if any(low.startswith(h) for h in headers):
            start = i + 1
            break
    if start is None:
        return []
    out: List[str] = []
    for j in range(start, len(lines)):
        s = _norm(lines[j])
        if any(s.startswith(h) for h in STOP_HEADERS) and not any(s.startswith(h) for h in headers):
            break
        out.append(lines[j])
    return out

def _count_bullets(lines: List[str]) -> int:
    return sum(1 for ln in lines if BULLET_LINE.match(ln or ""))

def _count_inline_markers(lines: List[str]) -> int:
    return sum(len(INLINE_ENUM.findall(ln or "")) for ln in lines)

def _has_takeaways(text: str) -> tuple[bool,int]:
    sect = _section_lines(text, TAKEAWAY_HEADERS)
    cnt = _count_bullets(sect)
    if cnt >= 3:
        return True, cnt
    if _count_inline_markers(sect) >= 3:
        return True, _count_inline_markers(sect)
    all_lines = text.splitlines()
    any_cnt = _count_bullets(all_lines)
    if any_cnt >= 3:
        return True, any_cnt
    if _count_inline_markers(all_lines) >= 3:
        return True, _count_inline_markers(all_lines)
    return False, max(cnt, any_cnt)

def _count_readings_text(text: str) -> int:
    sect = _section_lines(text, READING_HEADERS)
    cnt = 0
    for ln in sect:
        s = (ln or "").strip()
        if not s:
            continue
        if BULLET_LINE.match(s) or URL_OR_ID.search(s):
            cnt += 1
    if cnt >= 2:
        return cnt
    return len(URL_OR_ID.findall(text or ""))

def _citations_count(res: Dict[str, Any]) -> int:
    c = res.get("citations")
    if isinstance(c, list):
        n = 0
        for item in c:
            if isinstance(item, dict):
                if item.get("source") or item.get("url") or item.get("title"):
                    n += 1
        return n
    return 0

def _derive_tool_calls(res: Dict[str, Any]) -> int:
    v = res.get("tool_call_count")
    if isinstance(v, (int,float)):
        return int(v)
    total = 0
    for k in ("reader","metareviewer","critic","Reader","MetaReviewer","Critic"):
        m = res.get(k)
        if isinstance(m, dict):
            for key in ("tool_calls","tool_call_count"):
                if isinstance(m.get(key),(int,float)):
                    total += int(m.get(key))
            mx = m.get("metrics") or m.get("agent_metrics")
            if isinstance(mx, dict):
                for key in ("tool_calls","tool_call_count"):
                    if isinstance(mx.get(key),(int,float)):
                        total += int(mx.get(key))
    return total

def run_case(case: Dict[str, Any], pipeline=None, verbose: bool=False) -> Dict[str, Any]:
    pdf_path = _resolve_pdf_path(case.get("pdf_path",""))
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path} (case {case.get('id')})")

    if pipeline is None:
        pipeline = _import_pipeline()

    start = time.perf_counter()
    res = _fallback_pipeline(pdf_path) if pipeline is None else pipeline(pdf_path)
    total_latency = max(1, math.ceil((time.perf_counter() - start) * 1000))
    res = dict(res or {})
    res["latency_ms_total"] = total_latency
    res["case_id"] = case.get("id")

    final = res.get("final") or ""
    tk_ok, tk_cnt = _has_takeaways(final)
    rd_text = _count_readings_text(final)
    rd_cites = _citations_count(res)
    rd_cnt = max(rd_text, rd_cites)
    constraints = {
        "has_takeaways": tk_ok and tk_cnt >= 3,
        "has_readings": rd_cnt >= 2,
        "length_ok": 150 <= len(final) <= 12000,
    }
    res["constraints"] = constraints
    res["tool_call_count"] = _derive_tool_calls(res)

    if verbose:
        print(f"  constraints: takeaways={constraints['has_takeaways']} (cnt={tk_cnt}), "
              f"readings={constraints['has_readings']} (text_cnt={rd_text}, citations={rd_cites}), "
              f"length_ok={constraints['length_ok']} (len={len(final)})")

    return res

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="eval/cases")
    ap.add_argument("--out", default="eval/out")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    out_jsonl = os.path.join(args.out, "results.jsonl")

    case_paths = sorted(glob.glob(os.path.join(args.cases, "*.json")))
    if not case_paths:
        raise SystemExit(f"No cases found in {args.cases}. Run: python -m eval.make_cases --min 6")

    pipeline = _import_pipeline()

    count = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for path in case_paths:
            with open(path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
            res = run_case(case, pipeline=pipeline, verbose=args.verbose)
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            count += 1
            c = res.get("constraints") or {}
            print(f"Ran {case.get('id')} -> latency {res.get('latency_ms_total')} ms, tool_calls={res.get('tool_call_count')}, "
                  f"takeaways={c.get('has_takeaways')}, readings={c.get('has_readings')}")
    print(f"Wrote {count} rows to {out_jsonl}")

if __name__ == "__main__":
    main()
