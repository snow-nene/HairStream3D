import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vis.build_volume_acceptance_matrix import main


def test_acceptance_matrix_preserves_rejection_state(tmp_path, monkeypatch):
    root = tmp_path / "runs"; run = root / "front"; run.mkdir(parents=True)
    (run / "report.json").write_text(json.dumps({"status": "rejected_unseeded_components", "domain_voxels": 10, "unseeded_voxels": 2}))
    output = tmp_path / "matrix.json"
    monkeypatch.setattr(sys, "argv", ["build_volume_acceptance_matrix", "--root", str(root), "--output", str(output), "--runs", "front"])
    main()
    report = json.loads(output.read_text())
    assert report["all_passed"] is False
    assert report["runs"][0]["unseeded_voxels"] == 2
