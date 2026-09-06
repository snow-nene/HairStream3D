import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.evidence_requests import build_component_requests
from lib.pipeline_snapshot import save_stage_snapshot


def test_component_requests_include_world_centers():
    plan = {"components": [{"component_id": 2, "voxels": 3, "center_index": [1, 2, 3]}]}
    report = build_component_requests(plan, [0, 0, 0], [1, 1, 1], [5, 5, 5])
    assert report["request_count"] == 1
    assert report["requests"][0]["center_world"] == [.25, .5, .75]


def test_stage_snapshot_records_hash_and_arrays(tmp_path):
    record = save_stage_snapshot(tmp_path, "integration", {"points": np.zeros((2, 3))}, {"seed": 42})
    assert len(record["sha256"]) == 64
    assert (tmp_path / "stage_manifest.json").is_file()
