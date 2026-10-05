
from __future__ import annotations
import os, json, argparse, glob
from pathlib import Path

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _write_case(out_dir: str, idx: int, pdf_path: str) -> None:
    case = {"id": f"paper_case_{idx:02d}", "pdf_path": pdf_path}
    path = os.path.join(out_dir, f"paper_case_{idx:02d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(case, f, indent=2)

def _find_pdfs() -> list[str]:
    # Prefer curated sample papers in the repo
    patterns = [
        "data/sample_papers/*.pdf",
        "data/*.pdf",
        "data/external/*.pdf",
    ]
    pdfs: list[str] = []
    for pat in patterns:
        pdfs.extend(sorted(glob.glob(pat)))
    # Normalize to posix relative paths for portability
    pdfs = [str(Path(p)) for p in pdfs if os.path.isfile(p)]
    return pdfs

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=6, help="minimum number of cases to ensure")
    ap.add_argument("--out", default="eval/cases", help="cases output directory")
    args = ap.parse_args()

    _ensure_dir(args.out)
    existing = sorted(glob.glob(os.path.join(args.out, "*.json")))

    if len(existing) >= args.min:
        # Nothing to do
        print(f"{len(existing)} existing cases found in {args.out}; no new cases needed.")
        return

    pdfs = _find_pdfs()
    if not pdfs:
        # As a last resort, synthesize a tiny valid PDF with placeholder text
        synth_dir = "data/sample_papers"
        _ensure_dir(synth_dir)
        placeholder = os.path.join(synth_dir, "sample_auto.pdf")
        _write_minimal_pdf(placeholder, text="Sample PDF for evaluation")
        pdfs = [placeholder]

    # Cycle over available PDFs until we meet the minimum number of cases
    needed = args.min - len(existing)
    start_idx = 1
    if existing:
        # compute next index based on existing filenames like paper_case_01.json
        try:
            names = [os.path.basename(p) for p in existing]
            nums = [int(n.split("_")[-1].split(".")[0]) for n in names if n.startswith("paper_case_")]
            if nums:
                start_idx = max(nums) + 1
        except Exception:
            pass

    i = start_idx
    j = 0
    while needed > 0:
        pdf = pdfs[j % len(pdfs)]
        # Store relative path for portability
        rel = str(Path(pdf))
        _write_case(args.out, i, rel)
        print(f"Created case paper_case_{i:02d} -> {rel}")
        i += 1
        j += 1
        needed -= 1

def _write_minimal_pdf(path: str, text: str = "Hello, PDF") -> None:
    # Minimal PDF with a single page and a simple text draw. Offsets are not strictly accurate,
    # but modern parsers (including pypdf) can read this form. Keep it tiny.
    content = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 144]/Resources<</Font<</F1 5 0 R>>>>/Contents 4 0 R>>endobj\n"
        b"4 0 obj<</Length 44>>stream\n"
        b"BT /F1 18 Tf 72 100 Td (" + text.encode("latin-1", errors="ignore") + b") Tj ET\n"
        b"endstream\n"
        b"endobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000010 00000 n \n0000000056 00000 n \n0000000114 00000 n \n0000000240 00000 n \n0000000383 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n497\n%%EOF\n"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)

if __name__ == "__main__":
    main()
