"""Integration smoke test for scripts/evaluate_real.py in --mock mode."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_evaluate_real_mock_produces_csv_and_json_summary(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    out_base = tmp_path / "results"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "evaluate_real.py"),
            "--mock",
            "--methods",
            "full,monolithic",
            "--tasks",
            "clear_cluttered_plate_easy",
            "--trials",
            "2",
            "--out",
            str(out_base),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr

    run_dirs = [d for d in out_base.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    summary_json = run_dir / "summary.json"
    summary_csv = run_dir / "summary.csv"
    assert summary_json.exists()
    assert summary_csv.exists()

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert len(payload["summaries"]) == 2  # two methods x one task

    with summary_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {row["method"] for row in rows} == {"full", "monolithic"}
    assert all(row["task_id"] == "clear_cluttered_plate_easy" for row in rows)

    trial_files = list((run_dir / "trials").glob("*.json"))
    assert len(trial_files) == 4  # 2 methods x 2 trials
