import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.test.run_ablation_matrix import main


def test_ablation_matrix_cli_records_all_required_scenarios(tmp_path, monkeypatch):
    output = tmp_path / "ablation.json"
    monkeypatch.setattr(sys, "argv", ["run_ablation_matrix", "--output", str(output),
                                       "--sensitivity-seeds", "1", "2", "--sensitivity-budgets", "1", "2"])
    main()
    report = json.loads(output.read_text())
    assert len(report["records"]) == 5
    assert report["validation"]["comparable"] if "comparable" in report["validation"] else report["validation"]["fixed_seed"]
    assert any(item["metrics"]["ambiguous_samples"] > 0 for item in report["records"])
    assert len(report["parameter_sensitivity"]["grid"]) == 4
