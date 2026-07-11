"""Validate the preserved challenge artifact before it can be evaluated."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from .dataset import load_observations, load_split
from .protocol import protocol_hash


EXPECTED = {"train": 1407, "test": 600}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(root: str | Path, *, validate_test_labels: bool = True) -> dict[str, object]:
    root = Path(root)
    required = [
        "train_for_ml.csv",
        "test_for_ml.csv",
        "train_label.csv",
        "test_label.csv",
        "train_for_textcnn.zip",
        "test_for_textcnn.zip",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    report: dict[str, object] = {
        "root": str(root.resolve()),
        "protocol_hash": protocol_hash(),
        "missing": missing,
        "ready": False,
    }
    if missing:
        return report
    counts = {}
    train = load_split(root, "train")
    counts["train"] = {
        "cases": len(train.sample_ids),
        "observation_rows": len(train.observations),
        "label_sums": {key: int(value) for key, value in train.labels.sum().items()},
    }
    if validate_test_labels:
        test = load_split(root, "test")
        counts["test"] = {
            "cases": len(test.sample_ids),
            "observation_rows": len(test.observations),
            "label_sums": {key: int(value) for key, value in test.labels.sum().items()},
        }
        test_labels = pd.read_csv(root / "test_label.csv")
        inactive_roots_zero: bool | None = all(
            column in test_labels and int(test_labels[column].sum()) == 0
            for column in ("Root4", "Root5", "Root6")
        )
    else:
        test_observations = load_observations(root, "test")
        counts["test"] = {
            "cases": int(test_observations["sample_index"].nunique()),
            "observation_rows": len(test_observations),
            "label_sums": "deferred_until_after_fit",
        }
        inactive_roots_zero = None
    zip_counts = {}
    for split in ("train", "test"):
        with ZipFile(root / f"{split}_for_textcnn.zip") as archive:
            zip_counts[split] = sum(name.endswith(".csv") for name in archive.namelist())
    hash_names = required if validate_test_labels else [
        name for name in required if name != "test_label.csv"
    ]
    hashes = {name: _sha256(root / name) for name in hash_names}
    ready = (
        all(counts[split]["cases"] == EXPECTED[split] for split in EXPECTED)
        and all(zip_counts[split] == EXPECTED[split] for split in EXPECTED)
        and (inactive_roots_zero is not False)
    )
    report.update(
        {
            "counts": counts,
            "textcnn_archive_case_counts": zip_counts,
            "inactive_test_roots_4_to_6_are_zero": inactive_roots_zero,
            "test_label_validation_deferred": not validate_test_labels,
            "sha256": hashes,
            "ready": ready,
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    report = inspect(args.root)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n")
    print(payload)
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
