import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.evidence_seed_merge import merge_evidence_seeds


def test_supplemental_seeds_fill_only_unknown_and_record_conflicts():
    existing = np.array([[[1, 0, 0]]])
    supplemental = np.array([[[2, 2, 0]]])
    result = merge_evidence_seeds(existing, supplemental, "registered_view", "left")
    assert result["labels"].tolist() == [[[1, 2, 0]]]
    assert result["accepted_voxels"] == 1
    assert result["conflict_voxels"] == 1


def test_unknown_source_is_rejected():
    with pytest.raises(ValueError):
        merge_evidence_seeds(np.zeros((1, 1, 1)), np.ones((1, 1, 1)), "unknown", "front")
