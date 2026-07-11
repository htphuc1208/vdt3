from __future__ import annotations

from telco_mas.spotlight.readiness import spotlight_readiness


def test_spotlight_requires_complete_upstream_point_labels(tmp_path):
    root = tmp_path / "Anomaly Dataset"
    for index in range(16):
        case = root / "Anomaly" / f"case_{index:02d}"
        case.mkdir(parents=True)
        (case / "platform.csv").write_text("timestamp,x\n", encoding="utf-8")
        (case / "radio.csv").write_text("timestamp,y\n", encoding="utf-8")
    for scenario, count in {"1UE": 35, "5UE": 20, "7UE-Ping": 2}.items():
        path = root / "Baseline" / scenario
        path.mkdir(parents=True)
        for index in range(count):
            (path / f"{index}.zip").write_bytes(b"PK")

    report = spotlight_readiness(tmp_path)
    assert report["artifact_complete"] is True
    assert report["published_protocol_reproducible"] is False
    assert report["confirmatory_ready"] is False
    assert "Point-level anomaly labels" in report["warnings"][-1]
