import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vis.audit_visibility_metrics import main


def test_visibility_metrics_cli_writes_report(tmp_path, monkeypatch):
    source = tmp_path / "metrics.npz"
    np.savez(source, uv=np.array([[0., 0.]]), predicted_directions=np.array([[1., 0., 0.]]),
             target_directions=np.array([[1., 0., 0.]]), visible=np.array([True]), hair_mask=np.ones((1, 1), bool))
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["audit_visibility_metrics", "--input", str(source), "--output", str(output)])
    main()
    report = json.loads(output.read_text())
    assert report["coverage"] == 1.0
    assert report["leakage_rate"] == 0.0
