import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vis.audit_historical_violation_attribution import main


def test_historical_attribution_preserves_unknown_sources(tmp_path, monkeypatch):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"invalid_locations": [[3, 1], [3, 2], [7, 0]]}))
    output = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["audit", "--audit", str(audit), "--output", str(output)])
    main()
    report = json.loads(output.read_text())
    assert report["attribution_complete"] is False
    assert len(report["violating_strands"]) == 2
