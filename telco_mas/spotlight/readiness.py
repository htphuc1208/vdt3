"""Validate whether public SpotLight data can support paper-grade evaluation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EXPECTED_BASELINE_ARCHIVES = {"1UE": 35, "5UE": 20, "7UE-Ping": 2}
EXPECTED_ANOMALY_PAIRS = 16
PUBLISHED_OVERALL_F1 = 0.95


def spotlight_readiness(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    dataset_root = root_path / "Anomaly Dataset" if (root_path / "Anomaly Dataset").is_dir() else root_path
    anomaly_root = dataset_root / "Anomaly"
    baseline_root = dataset_root / "Baseline"

    pairs = []
    if anomaly_root.is_dir():
        for platform_path in sorted(anomaly_root.rglob("platform.csv")):
            radio_path = platform_path.with_name("radio.csv")
            if radio_path.is_file():
                pairs.append(platform_path.parent)

    baseline_counts = {
        name: len(list((baseline_root / name).glob("*.zip"))) if (baseline_root / name).is_dir() else 0
        for name in EXPECTED_BASELINE_ARCHIVES
    }
    label_files = sorted(
        path for path in dataset_root.rglob("*")
        if path.is_file() and (path.name.lower() == "label" or "label" in path.stem.lower())
    ) if dataset_root.is_dir() else []

    anomaly_complete = len(pairs) == EXPECTED_ANOMALY_PAIRS
    baseline_complete = baseline_counts == EXPECTED_BASELINE_ARCHIVES
    # The Drive listing has no point-label artifacts. A label elsewhere must be
    # explicitly associated with every experiment before point-level F1 is valid.
    labels_complete = len(label_files) >= EXPECTED_ANOMALY_PAIRS
    warnings = []
    if not anomaly_complete:
        warnings.append(f"Expected {EXPECTED_ANOMALY_PAIRS} anomaly platform/radio pairs, found {len(pairs)}.")
    if not baseline_complete:
        warnings.append(
            f"Baseline archive counts are incomplete: expected {EXPECTED_BASELINE_ARCHIVES}, found {baseline_counts}."
        )
    if not labels_complete:
        warnings.append(
            "Point-level anomaly labels are incomplete in the public artifact. Do not reproduce or compare "
            "published point-level F1 by inferring labels from KPI values, filenames, or the nominal injection schedule."
        )

    return {
        "dataset": "SpotLight Open RAN",
        "root": str(dataset_root),
        "domain": "commercial-grade 5G Open RAN testbed (CU/DU/RU, radio, fronthaul, platform)",
        "service_layer": False,
        "anomaly_pair_count": len(pairs),
        "expected_anomaly_pair_count": EXPECTED_ANOMALY_PAIRS,
        "baseline_archive_counts": baseline_counts,
        "expected_baseline_archive_counts": EXPECTED_BASELINE_ARCHIVES,
        "point_label_file_count": len(label_files),
        "point_label_paths": [str(path.relative_to(dataset_root)) for path in label_files[:20]],
        "published_spotlight_overall_f1": PUBLISHED_OVERALL_F1,
        "artifact_complete": anomaly_complete and baseline_complete,
        "published_protocol_reproducible": anomaly_complete and baseline_complete and labels_complete,
        "confirmatory_ready": anomaly_complete and baseline_complete and labels_complete,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Audit SpotLight Open RAN benchmark readiness")
    parser.add_argument("--root", default="data/spotlight")
    parser.add_argument("--out", default=None)
    parser.add_argument("--strict", action="store_true", help="return exit 2 when confirmatory data are not ready")
    args = parser.parse_args(argv)
    report = spotlight_readiness(args.root)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["confirmatory_ready"] or not args.strict else 2


if __name__ == "__main__":
    raise SystemExit(main())
