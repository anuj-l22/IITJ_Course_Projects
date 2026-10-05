#!/usr/bin/env python3
# dataset_prep.py — robust GOOD/OGB prep with automatic fallback to generate=True
import argparse, os, sys, importlib, traceback, inspect
from typing import Iterable, List
# at the top of dataset_prep.py
import importlib, sys, os

def import_good(modname: str):
    """
    Try to import GOOD modules from either:
      - 'GOOD.data.good_datasets.*' (some installs),
      - 'graph_ood.data.good_datasets.*' (others).
    """
    try:
        return importlib.import_module(modname)
    except ModuleNotFoundError:
        if modname.startswith("GOOD."):
            alt = "graph_ood." + modname.split("GOOD.", 1)[1]
            return importlib.import_module(alt)
        raise

def add_repo_root():
    here = os.getcwd()
    if os.path.isdir(os.path.join(here, "GOOD")) and here not in sys.path:
        sys.path.insert(0, here)

def need(mod, hint=""):
    try:
        importlib.import_module(mod)
    except Exception:
        print(f"[ERROR] Cannot import '{mod}'. {hint}")
        raise

def parse_list(csv: str) -> List[str]:
    return [x.strip().lower() for x in csv.split(",") if x.strip()]

def import_good(modname: str):
    try:
        return importlib.import_module(modname)
    except Exception:
        print(f"[ERROR] Import failed: {modname}")
        traceback.print_exc()
        print("Tip: run from UIL repo root so 'GOOD/' is on sys.path, or install GOOD via:")
        print("  pip install git+https://github.com/divelab/GOOD")
        raise

def ctor_supports_kw(ctor, kw: str) -> bool:
    try:
        sig = inspect.signature(ctor)
        return kw in sig.parameters
    except Exception:
        return False

def try_build(builder, **kwargs):
    """Try building; if it fails due to download quota, retry with generate=True when supported."""
    try:
        return builder(**kwargs)
    except Exception as e:
        # If the ctor supports 'generate', try again with generate=True
        if ctor_supports_kw(builder, "generate"):
            print("[INFO] Falling back to generate=True …")
            kwargs2 = dict(kwargs)
            kwargs2["generate"] = True
            return builder(**kwargs2)
        raise

def rdkit_ok() -> bool:
    try:
        from rdkit import Chem  # noqa
        return True
    except Exception:
        return False

def summary(root: str):
    print("\n[SUMMARY] Checked paths under:", root)
    for sub in [
        "GOODCMNIST",
        "GOODMotif",
        os.path.join("OGB","ogbg_molhiv"),
        os.path.join("OGB","ogbg_molbbbp"),
    ]:
        p = os.path.join(root, sub)
        print("  -", p, "exists" if os.path.exists(p) else "missing")

# ---------------- GOOD builders ----------------
def prepare_good_cmnist(root: str, shifts: Iterable[str], subsets: Iterable[str], prefer_generate: bool):
    mod = import_good("GOOD.data.good_datasets.good_cmnist")
    GOODCMNIST = getattr(mod, "GOODCMNIST")
    for shift in shifts:
        for subset in subsets:
            print(f"[GOOD-CMNIST] domain=color shift={shift} subset={subset}")
            kwargs = dict(root=root, domain="color", shift=shift, subset=subset)
            if prefer_generate and ctor_supports_kw(GOODCMNIST, "generate"):
                kwargs["generate"] = True
            _ = try_build(GOODCMNIST, **kwargs)

def prepare_good_motif(root: str, domains: Iterable[str], shifts: Iterable[str], subsets: Iterable[str], prefer_generate: bool):
    mod = import_good("GOOD.data.good_datasets.good_motif")
    GOODMotif = getattr(mod, "GOODMotif")
    for domain in domains:        # e.g., 'basis'/'base', 'size'
        for shift in shifts:
            for subset in subsets:
                print(f"[GOOD-Motif] domain={domain} shift={shift} subset={subset}")
                kwargs = dict(root=root, domain=domain, shift=shift, subset=subset)
                if prefer_generate and ctor_supports_kw(GOODMotif, "generate"):
                    kwargs["generate"] = True
                _ = try_build(GOODMotif, **kwargs)

def prepare_good_hiv(root, domains, shifts, prefer_generate):
    mod = import_good("GOOD.data.good_datasets.good_hiv")
    # patch RDKit symbols into whichever module was imported
    from rdkit import Chem as _Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold as _MS
    setattr(mod, "Chem", _Chem)
    setattr(mod, "MurckoScaffold", _MS)

    GOODHIV = getattr(mod, "GOODHIV")  # class is the same in both layouts
    for domain in domains:
        for shift in shifts:
            print(f"[GOOD-HIV] domain={domain} shift={shift}")
            try:
                GOODHIV.load(root, domain=domain, shift=shift, generate=prefer_generate)
            except Exception:
                print("[INFO] Falling back to generate=True …")
                GOODHIV.load(root, domain=domain, shift=shift, generate=True)

def prepare_good_bbbp(root, domains, shifts, prefer_generate):
    mod = import_good("GOOD.data.good_datasets.good_bbbp")
    # patch RDKit symbols
    from rdkit import Chem as _Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold as _MS
    setattr(mod, "Chem", _Chem)
    setattr(mod, "MurckoScaffold", _MS)

    GOODBBBP = getattr(mod, "GOODBBBP")
    for domain in domains:
        for shift in shifts:
            print(f"[GOOD-BBBP] domain={domain} shift={shift}")
            try:
                GOODBBBP.load(root, domain=domain, shift=shift, generate=prefer_generate)
            except Exception:
                print("[INFO] Falling back to generate=True …")
                GOODBBBP.load(root, domain=domain, shift=shift, generate=True)


# ---------------- OGB caching ----------------
def prepare_ogb(root: str, names: Iterable[str], skip_rdkit_check=False):
    if not skip_rdkit_check and not rdkit_ok():
        print("[ERROR] RDKit not found. Required for molecule datasets (MolHIV/BBBP/PCBA).")
        print("Try: conda install -c conda-forge rdkit=2022.09.1")
        raise SystemExit(1)
    from ogb.graphproppred import PygGraphPropPredDataset
    ogb_root = os.path.join(root, "OGB")
    os.makedirs(ogb_root, exist_ok=True)
    for name in names:
        canonical = name.lower()
        if not canonical.startswith("ogbg-"):
            canonical = "ogbg-" + canonical
        print(f"[OGB] Caching dataset: {canonical}")
        _ = PygGraphPropPredDataset(name=canonical, root=ogb_root)

def main():
    add_repo_root()

    ap = argparse.ArgumentParser(description="Prepare/download datasets for UIL (GOOD + OGB).")
    ap.add_argument("--root", type=str, default="./datasets",
                    help="Datasets root (default: ./datasets)")
    ap.add_argument("--good", type=str, default="cmnist,motif",
                    help="Comma-separated GOOD datasets: {cmnist,motif,none}")
    ap.add_argument("--motif-domains", type=str, default="basis,size",
                    help="Domains for GOOD-Motif (e.g., basis,size OR base,size)")
    ap.add_argument("--shifts", type=str, default="covariate,concept",
                    help="Shifts to prepare: {covariate,concept}")
    ap.add_argument("--subsets", type=str, default="train,val,test,id_val,id_test",
                    help="Subsets to materialize for GOOD")
    ap.add_argument("--ogb", type=str, default="none",
                    help="OGB datasets: {molhiv,molbbbp,molpcba,none}")
    ap.add_argument("--prefer-generate", action="store_true",
                    help="Force generate=True when supported (avoids Google Drive).")
    ap.add_argument("--skip-rdkit-check", action="store_true",
                    help="Skip RDKit check (only if you know it’s installed).")
    ap.add_argument("--good-mols", type=str, default="none",
                help="GOOD molecule datasets to prepare: {hiv,bbbp,none}")
    ap.add_argument("--mol-domains", type=str, default="scaffold,size",
                    help="Domains for GOOD molecules (e.g., scaffold,size)")

    args = ap.parse_args()
    root = os.path.abspath(args.root)
    os.makedirs(root, exist_ok=True)
    print(f"[INFO] Using dataset root: {root}")

    need("torch_geometric", "Install PyTorch Geometric first (torch-scatter/sparse/cluster wheels).")
    need("ogb", "pip install ogb")

    good_list = [] if args.good.lower() in ("", "none") else parse_list(args.good)
    ogb_list  = [] if args.ogb.lower()  in ("", "none") else parse_list(args.ogb)
    mol_domains = parse_list(args.mol_domains)
    good_mols   = [] if args.good_mols.lower() in ("", "none") else parse_list(args.good_mols)
    shifts    = parse_list(args.shifts)
    subsets   = parse_list(args.subsets)
    motif_domains = parse_list(args.motif_domains)

    if "cmnist" in good_list:
        prepare_good_cmnist(root, shifts=shifts, subsets=subsets, prefer_generate=args.prefer_generate)
    if "motif" in good_list:
        prepare_good_motif(root, domains=motif_domains, shifts=shifts, subsets=subsets, prefer_generate=args.prefer_generate)
    if "hiv" in good_mols:
        prepare_good_hiv(root, domains=mol_domains, shifts=shifts, prefer_generate=args.prefer_generate)
    if "bbbp" in good_mols:
        prepare_good_bbbp(root, domains=mol_domains, shifts=shifts, prefer_generate=args.prefer_generate)
    if ogb_list:
        prepare_ogb(root, names=ogb_list, skip_rdkit_check=args.skip_rdkit_check)

    summary(root)
    print("\n[DONE] Dataset preparation finished.")

if __name__ == "__main__":
    main()
