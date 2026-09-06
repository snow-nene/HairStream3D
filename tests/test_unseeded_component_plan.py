import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.unseeded_component_plan import plan_unseeded_component_evidence


def test_unseeded_component_plan_never_assigns_labels():
    domain = np.ones((4, 4, 4), bool)
    partitions = np.ones_like(domain, int)
    partitions[0, 0, 0] = 0
    seeds = np.ones_like(domain, int)
    seeds[0, 0, 0] = 0
    result = plan_unseeded_component_evidence(domain, partitions, seeds)
    assert result["unseeded_voxels"] == 1
    assert not result["formal_bundle_allowed"]
    assert result["components"][0]["status"] == "awaiting_evidence"
    assert np.all(seeds == np.where(seeds > 0, seeds, 0))


def test_seeded_domain_has_no_evidence_requests():
    result = plan_unseeded_component_evidence(np.ones((2, 2, 2), bool), np.ones((2, 2, 2), int), np.ones((2, 2, 2), int))
    assert result["formal_bundle_allowed"]
    assert result["components"] == []
