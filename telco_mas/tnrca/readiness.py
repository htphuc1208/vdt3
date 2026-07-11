"""CLI for TN-RCA-style dataset integrity and readiness checks."""
from __future__ import annotations

import json
from pathlib import Path

from .dataset import TNRCADataset, TNRCADatasetError


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Audit a TN-RCA-style telecom alarm dataset")
    parser.add_argument("--root", required=True)
    parser.add_argument("--minimum-cases", type=int, default=100)
    parser.add_argument("--out", default=None)
    parser.add_argument("--strict", action="store_true", help="return exit 2 when confirmatory data are not ready")
    args = parser.parse_args(argv)
    try:
        report = TNRCADataset(args.root).readiness(minimum_confirmatory_cases=args.minimum_cases)
    except TNRCADatasetError as exc:
        report = {"confirmatory_ready": False, "error": str(exc), "root": str(Path(args.root).resolve())}
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("confirmatory_ready") or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
