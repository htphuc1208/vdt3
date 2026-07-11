"""Run the WWW'25 RCAEval BARO reproduction without importing unrelated baselines."""
from __future__ import annotations

import argparse
import importlib.util
import os
import runpy
import sys
import types
from pathlib import Path


BASELINE_NAMES = (
    "baro",
    "causalrca",
    "circa",
    "cloudranger",
    "cmlp_pagerank",
    "dummy",
    "e_diagnosis",
    "easyrca",
    "fci_pagerank",
    "fci_randomwalk",
    "ges_pagerank",
    "granger_pagerank",
    "granger_randomwalk",
    "lingam_pagerank",
    "lingam_randomwalk",
    "micro_diag",
    "microcause",
    "microrank",
    "mscred",
    "nsigma",
    "ntlr_pagerank",
    "ntlr_randomwalk",
    "pc_pagerank",
    "pc_randomwalk",
    "run",
    "tracerca",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--tdelta", type=int, default=0)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--vendor-root", default="vendor/RCAEval-www25")
    parser.add_argument("--out-root", default="results/reproductions/rcaeval_www25")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    vendor_root = (project_root / args.vendor_root).resolve()
    source_data = (project_root / "data" / "rcaeval").resolve()
    run_root = (project_root / args.out_root / args.dataset / "baro").resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    data_link = run_root / "data"
    if not data_link.exists():
        data_link.symlink_to(_paper_data_layout(run_root, source_data), target_is_directory=True)

    sys.path.insert(0, str(vendor_root))
    baro = _load_baro(vendor_root)
    e2e_stub = types.ModuleType("RCAEval.e2e")
    e2e_stub.rca = _identity_rca
    for name in BASELINE_NAMES:
        setattr(e2e_stub, name, baro)
    sys.modules["RCAEval.e2e"] = e2e_stub

    forwarded = [
        str(vendor_root / "main.py"),
        "--method",
        "baro",
        "--dataset",
        args.dataset,
        "--length",
        str(args.length),
        "--tdelta",
        str(args.tdelta),
    ]
    if args.test:
        forwarded.append("--test")
    sys.argv = forwarded
    os.chdir(run_root)
    runpy.run_path(str(vendor_root / "main.py"), run_name="__main__")
    return 0


def _load_baro(vendor_root: Path):
    path = vendor_root / "RCAEval" / "e2e" / "baro.py"
    spec = importlib.util.spec_from_file_location("rcaeval_www25_baro", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load official BARO implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.baro


def _identity_rca(func):
    return func


def _paper_data_layout(run_root: Path, source_data: Path) -> Path:
    layout = run_root / ".paper-data"
    for suite in ("RE1", "RE2", "RE3"):
        suite_dir = layout / suite
        suite_dir.mkdir(parents=True, exist_ok=True)
        for system in ("OB", "SS", "TT"):
            target = source_data / f"{suite}-{system}"
            link = suite_dir / f"{suite}-{system}"
            if target.exists() and not link.exists():
                link.symlink_to(target, target_is_directory=True)
    return layout


if __name__ == "__main__":
    raise SystemExit(main())
