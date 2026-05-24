from __future__ import annotations

import subprocess
from pathlib import Path


def test_smoke_run() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root.parent / "data" / "SBU-shadow" / "SBU-Test"
    out_dir = root / "outputs"
    cmd = [
        "python",
        "run.py",
        "--data-dir",
        str(data_dir),
        "--images-subdir",
        "ShadowImages",
        "--masks-subdir",
        "ShadowMasks",
        "--num-images",
        "1",
        "--reduction",
        "0.1",
        "--seed",
        "7",
        "--output-root",
        str(out_dir),
    ]
    res = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr
