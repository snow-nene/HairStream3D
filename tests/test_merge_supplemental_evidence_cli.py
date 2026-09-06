import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.recon_3d.merge_supplemental_evidence import main


def test_merge_cli_records_conflicts_and_unresolved(tmp_path, monkeypatch):
    base = tmp_path / "base.npz"; extra = tmp_path / "extra.npz"; output = tmp_path / "merged.npz"
    np.savez(base, partition_labels=np.array([[[1, 0]]]), domain_mask=np.array([[[True, True]]]))
    np.savez(extra, labels=np.array([[[2, 2]]]))
    monkeypatch.setattr(sys, "argv", ["merge", "--base", str(base), "--supplemental", str(extra),
                                       "--source-kind", "registered_view", "--source-view", "left", "--output", str(output)])
    main()
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["accepted_voxels"] == 1
    assert report["conflict_voxels"] == 1
    assert report["formal_bundle_allowed"] is True
